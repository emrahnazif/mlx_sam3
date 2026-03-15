import time
from functools import partial

from typing import Dict, List
import PIL
from PIL import Image
import numpy as np
import mlx.core as mx

from sam3.model import box_ops
from sam3.model.data_misc import FindStage, interpolate

# TODO: remove this, using for testing
import torch
from torchvision.transforms import v2


def transform(image_path_or_pil, resolution):
    if isinstance(image_path_or_pil, str):
        img = Image.open(image_path_or_pil).convert("RGB")
    else:
        img = image_path_or_pil.convert("RGB")
    
    img = img.resize((resolution, resolution), resample=Image.Resampling.LANCZOS)
    img_np = np.array(img).astype(np.float32) / 255.0 # [H, W, C]

    img_np = (img_np - 0.5) / 0.5

    return mx.array(img_np).transpose(2, 0, 1)  # [H, W, C] -> [C, H, W]

class Sam3Processor:
    def __init__(self, model, resolution=1008, confidence_threshold=0.5):
        self.model = model
        self.resolution = resolution
        self.confidence_threshold = confidence_threshold
        self.transform = partial(transform, resolution=self.resolution)


        self.find_stage = FindStage(
            img_ids=mx.array([0], dtype=mx.int64),
            text_ids=mx.array([0], dtype=mx.int64),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

   
    def set_image(self, image, state=None):
        if state is None:
            state = {}
        
        if isinstance(image, PIL.Image.Image):
            width, height = image.size
        # elif isinstance(image, (mx.array, np.ndarray)):
        #     height, width = image.shape[-2:]
        else:
            raise ValueError("Image must be a PIL image")
        
        image = self.transform(image)[None]

        state["original_height"] = height
        state["original_width"] = width
        import time
        start = time.perf_counter()
        state["backbone_out"] = self.model.backbone.call_image(image)
        mx.eval(state)
        second = time.perf_counter()
        print(f"Backbone pass took {second - start:.2f} Seconds")
        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_backbone_out["backbone_fpn"][0]
                )
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_backbone_out["backbone_fpn"][1]
                )
            )
            mx.eval(state["backbone_out"])
        return state

    def set_image_batch(self, iamges: List[np.ndarray], state=None):
        pass

    def set_text_prompt(self, prompt: str, state: Dict):
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")
        
        text_outputs = self.model.backbone.call_text([prompt])
        # will erase the previous text prompt if any
        state["backbone_out"].update(text_outputs)
        mx.eval(state["backbone_out"])
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()
        return self._call_grounding(state)

    def add_geometric_prompt(self, box: List, label: bool, state: Dict):
        """Adds a box prompt and run the inference.
        The image needs to be set, but not necessarily the text prompt.
        The box is assumed to be in [center_x, center_y, width, height] format and normalized in [0, 1] range.
        The label is True for a positive box, False for a negative box.
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        if "language_features" not in state["backbone_out"]:
            # Looks like we don't have a text prompt yet. This is allowed, but we need to set the text prompt to "visual" for the model to rely only on the geometric prompt
            dummy_text_outputs = self.model.backbone.call_text(
                ["visual"]
            )
            state["backbone_out"].update(dummy_text_outputs)
            mx.eval(state["backbone_out"])

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        # adding a batch and sequence dimension
        boxes = mx.array(box, dtype=mx.float32).reshape(1, 1, 4)
        labels = mx.array([label], dtype=mx.bool_).reshape(1, 1)
        state["geometric_prompt"].append_boxes(boxes, labels)

        return self._call_grounding(state)

    def add_point_prompt(self, point: List, label: bool, state: Dict):
        """Adds a point prompt and runs inference.
        The image needs to be set, but not necessarily the text prompt.
        point: [x, y] normalized in [0, 1] range (x=horizontal, y=vertical).
        label: True for a positive point (foreground), False for a negative point (background).
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before add_point_prompt")

        if "language_features" not in state["backbone_out"]:
            dummy_text_outputs = self.model.backbone.call_text(["visual"])
            state["backbone_out"].update(dummy_text_outputs)
            mx.eval(state["backbone_out"])

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        # shape: (N_points=1, batch_size=1, 2)
        points = mx.array(point, dtype=mx.float32).reshape(1, 1, 2)
        # 1 = positive (foreground), 0 = negative (background)
        # Use int32 — MLX GPU scatter doesn't support int64; model casts internally
        labels = mx.array([int(label)], dtype=mx.int32).reshape(1, 1)
        state["geometric_prompt"].append_points(points, labels)

        return self._call_grounding(state)

    def add_points_prompt(self, points: List[List[float]], labels: List[bool], state: Dict):
        """Add multiple points (positive and negative) in a single call.

        points: list of [x, y] pairs, each normalized to [0, 1]
        labels: list of True (positive/foreground) or False (negative/background)
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before add_points_prompt")

        if "language_features" not in state["backbone_out"]:
            dummy_text_outputs = self.model.backbone.call_text(["visual"])
            state["backbone_out"].update(dummy_text_outputs)
            mx.eval(state["backbone_out"])

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        # shape: (N_points, 1, 2)
        pts = mx.array(points, dtype=mx.float32).reshape(len(points), 1, 2)
        # shape: (N_points, 1) — 1=positive, 0=negative
        lbs = mx.array([int(l) for l in labels], dtype=mx.int32).reshape(len(labels), 1)
        state["geometric_prompt"].append_points(pts, lbs)

        return self._call_grounding(state)

    def reset_all_prompts(self, state: Dict):
        """Removes all the prompts and results"""
        if "backbone_out" in state:
            backbone_keys_to_del = [
                "language_features",
                "language_mask",
                "language_embeds",
            ]
            for key in backbone_keys_to_del:
                if key in state["backbone_out"]:
                    del state["backbone_out"][key]

        keys_to_del = ["geometric_prompt", "boxes", "masks", "mask_logits", "scores", "semantic_seg"]
        for key in keys_to_del:
            if key in state:
                del state[key]

    def set_confidence_threshold(self, threshold: float, state=None):
        pass

    def _call_grounding(self, state: Dict):
        # Delete previous results to allow GC before allocating new ones
        for key in ("masks", "mask_logits", "semantic_seg"):
            if key in state:
                del state[key]

        outputs = self.model.call_grounding(
            backbone_out=state["backbone_out"],
            find_input=self.find_stage,
            geometric_prompt=state["geometric_prompt"],
            find_target=None
        )

        # Phase 1: Extract needed arrays and evaluate at model resolution.
        # This materializes decoder outputs and frees the encoder/decoder
        # computation graph BEFORE the expensive full-resolution interpolation.
        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        seg_mask = outputs["semantic_seg"]
        presence_logit = outputs["presence_logit_dec"]
        mx.eval(out_bbox, out_logits, out_masks, seg_mask, presence_logit)
        del outputs
        # Free cached metal memory from model's encoder/decoder intermediates
        # before allocating large full-resolution interpolation buffers
        mx.clear_cache()

        # Phase 2: Filter by confidence (cheap ops on materialized arrays)
        out_probs = mx.sigmoid(out_logits)
        presence_score = mx.sigmoid(presence_logit)[:,None]
        out_probs = (out_probs * presence_score).squeeze(-1)

        keep = out_probs > self.confidence_threshold
        mask_np = np.array(keep[0])
        indices = mx.array(mask_np.nonzero()[0])
        out_probs = out_probs[0][indices]
        out_masks = out_masks[0][indices]
        out_bbox = out_bbox[0][indices]

        # convert box to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        img_h = state["original_height"]
        img_w = state["original_width"]
        scale_fct = mx.array([img_w, img_h, img_w, img_h])
        boxes = boxes * scale_fct[None, :]

        # Phase 3: Interpolate masks to full resolution in batches to limit
        # peak memory. Each batch allocates N×H×W×4 bytes; with batch_size=8
        # and 5000×3333 that's ~533 MB per batch instead of ~5.3 GB all at once.
        interpolator = partial(interpolate,
            size=(img_h, img_w),
            mode="bilinear",
            align_corners=False,
        )

        n_masks = out_masks.shape[0]
        batch_size = 8
        mask_chunks = []
        mask_logit_chunks = []
        for start in range(0, n_masks, batch_size):
            end = min(start + batch_size, n_masks)
            chunk = interpolator(out_masks[start:end, None])
            chunk = mx.sigmoid(chunk)
            mx.eval(chunk)
            mask_logit_chunks.append(chunk)
            mask_chunks.append(chunk > 0.5)
            mx.clear_cache()

        if mask_logit_chunks:
            out_masks = mx.concatenate(mask_logit_chunks, axis=0)
            binary_masks = mx.concatenate(mask_chunks, axis=0)
        else:
            out_masks = mx.zeros((0, 1, img_h, img_w))
            binary_masks = mx.zeros((0, 1, img_h, img_w), dtype=mx.bool_)

        seg_mask = interpolator(seg_mask)

        state["semantic_seg"] = seg_mask
        state["mask_logits"] = out_masks
        state["masks"] = binary_masks
        state["boxes"] = boxes
        state["scores"] = out_probs

        # Materialize remaining results
        mx.eval(state["boxes"], state["scores"], state["semantic_seg"])
        mx.clear_cache()
        return state