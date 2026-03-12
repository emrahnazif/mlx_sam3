import time
import mlx.core as mx
import numpy as np
from PIL import Image
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import normalize_bbox


def visualize_semantic_mask(image: Image.Image, seg_mask: mx.array, alpha: float = 0.5, 
                            color: tuple = (255, 0, 0)) -> Image.Image:
    """
    Overlay binary semantic segmentation mask on original image.
    
    Args:
        image: Original PIL image
        seg_mask: Semantic segmentation logits [B, 1, H, W] or [1, H, W]
        alpha: Transparency of the overlay (0-1)
        color: RGB color tuple for the mask overlay
    
    Returns:
        PIL Image with mask overlay
    """
    # Convert to numpy and apply sigmoid for probabilities
    seg_np = np.array(seg_mask)
    
    # Handle different shapes
    if seg_np.ndim == 4:
        seg_np = seg_np[0, 0]  # [B, C, H, W] -> [H, W]
    elif seg_np.ndim == 3:
        seg_np = seg_np[0]  # [C, H, W] -> [H, W]
    
    # Apply sigmoid to convert logits to probabilities
    seg_probs = 1 / (1 + np.exp(-seg_np))
    
    # Threshold to get binary mask
    seg_binary = (seg_probs > 0.5).astype(np.float32)
    
    # Resize mask to match image size if needed
    mask_h, mask_w = seg_binary.shape
    img_w, img_h = image.size
    
    if (mask_h, mask_w) != (img_h, img_w):
        mask_pil = Image.fromarray((seg_binary * 255).astype(np.uint8))
        mask_pil = mask_pil.resize((img_w, img_h), Image.BILINEAR)
        seg_binary = np.array(mask_pil) / 255.0
    
    # Create colored overlay
    overlay = np.zeros((img_h, img_w, 4), dtype=np.uint8)
    overlay[..., 0] = color[0]  # R
    overlay[..., 1] = color[1]  # G
    overlay[..., 2] = color[2]  # B
    overlay[..., 3] = (seg_binary * alpha * 255).astype(np.uint8)  # Alpha
    
    # Composite overlay on image
    image_rgba = image.convert("RGBA")
    overlay_img = Image.fromarray(overlay, mode="RGBA")
    result = Image.alpha_composite(image_rgba, overlay_img)
    
    return result.convert("RGB")


def save_semantic_mask(seg_mask: mx.array, output_path: str = "semantic_mask.png"):
    """
    Save the binary semantic mask as a grayscale image.
    
    Args:
        seg_mask: Semantic segmentation logits [B, 1, H, W]
        output_path: Path to save the mask
    """
    seg_np = np.array(seg_mask)
    
    # Handle shape
    if seg_np.ndim == 4:
        seg_np = seg_np[0, 0]
    elif seg_np.ndim == 3:
        seg_np = seg_np[0]
    
    # Sigmoid + threshold
    seg_probs = 1 / (1 + np.exp(-seg_np))
    seg_binary = (seg_probs > 0.5).astype(np.uint8) * 255
    
    mask_img = Image.fromarray(seg_binary, mode="L")
    mask_img.save(output_path)
    print(f"Semantic mask saved to: {output_path}")
    return mask_img


def main():
    start = time.perf_counter()
    model = build_sam3_image_model()

    second = time.perf_counter()
    print(f"Model loaded in {second - start:.2f} seconds.")

    image_path = "fietzfotos-road-7064492_1920.jpg"
    image = Image.open(image_path)
    width, height = image.size
    processor = Sam3Processor(model, confidence_threshold=0.5)
    inference_state = processor.set_image(image)
    inter = time.perf_counter()
    print(f"Image processed in {inter - second:.2f} seconds.")

    # =========================================================
    # Example 1: Text prompt
    # =========================================================
    print("\n--- Example 1: Text Prompt ---")
    processor.reset_all_prompts(inference_state)
    inference_state = processor.set_text_prompt(state=inference_state, prompt="road")
    masks, boxes, scores = inference_state["masks"], inference_state["boxes"], inference_state["scores"]
    t1 = time.perf_counter()
    print(f"Text prompt inference in {t1 - inter:.2f} seconds.")
    print(f"Objects found: {len(scores)}, Scores: {scores}")

    # =========================================================
    # Example 2: Point prompts (positive + negative)
    #
    # Points must be normalized to [0, 1] by dividing pixel
    # coordinates by image width/height:
    #   norm_x = pixel_x / width
    #   norm_y = pixel_y / height
    #
    # label=True  → positive point (object is HERE)
    # label=False → negative point (NOT this region)
    # =========================================================
    print("\n--- Example 2: Point Prompts ---")
    processor.reset_all_prompts(inference_state)

    # All points submitted in a single batch call:
    #   pos_point → positive (foreground, lower-center of road)
    #   neg_point → negative (background, upper-center sky)
    pos_point = [0.5, 0.7]   # lower-center (road)
    neg_point = [0.5, 0.1]   # upper-center (sky)
    inference_state = processor.add_points_prompt(
        points=[pos_point, neg_point],
        labels=[True, False],   # True=foreground, False=background
        state=inference_state,
    )
    t2 = time.perf_counter()
    print(f"Batch points {pos_point} (+) and {neg_point} (-):")
    print(f"  Objects found: {len(inference_state['scores'])}, Scores: {inference_state['scores']}")
    print(f"Point prompt inference in {t2 - t1:.2f} seconds.")

    # =========================================================
    # Example 3: Mixed — text prompt + point refinement
    # =========================================================
    print("\n--- Example 3: Text + Point Prompts combined ---")
    processor.reset_all_prompts(inference_state)
    inference_state = processor.set_text_prompt(state=inference_state, prompt="road")

    # Positive click to confirm the road area
    road_point = [width * 0.5 / width, height * 0.8 / height]
    inference_state = processor.add_point_prompt(
        point=road_point,
        label=True,
        state=inference_state,
    )
    t3 = time.perf_counter()
    output = inference_state
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    print(f"Text + point inference in {t3 - t2:.2f} seconds.")
    print(f"Objects found: {len(scores)}, Scores: {scores}")
    print(f"Boxes: {boxes}")

    # === Semantic Segmentation Visualization ===
    if "semantic_seg" in output:
        seg_mask = output["semantic_seg"]
        print(f"\nSemantic mask shape: {seg_mask.shape}")
        save_semantic_mask(seg_mask, "semantic_mask.png")
        overlay_img = visualize_semantic_mask(
            image,
            seg_mask,
            alpha=0.5,
            color=(0, 255, 128),
        )
        overlay_img.save("semantic_overlay.png")
        print("Semantic overlay saved to: semantic_overlay.png")


if __name__ == "__main__":
    main()
