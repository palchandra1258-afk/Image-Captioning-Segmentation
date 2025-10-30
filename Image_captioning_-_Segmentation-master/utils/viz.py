"""
Visualization utilities for masks, overlays, and results
"""
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict
import io


# COCO color palette
COCO_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0),
    (0, 255, 128), (128, 0, 255), (0, 128, 255), (192, 192, 192), (128, 128, 128)
]


def get_color_for_class(class_id: int) -> Tuple[int, int, int]:
    """Get consistent color for a class ID"""
    return COCO_COLORS[class_id % len(COCO_COLORS)]


def overlay_instance_masks(image: Image.Image, masks: np.ndarray, labels: np.ndarray, 
                          scores: np.ndarray, boxes: np.ndarray, class_names: List[str],
                          alpha: float = 0.5, show_boxes: bool = True, 
                          show_labels: bool = True) -> Image.Image:
    """
    Overlay instance segmentation masks on image
    
    Args:
        image: Original PIL Image
        masks: Masks array (N, H, W)
        labels: Class labels (N,)
        scores: Confidence scores (N,)
        boxes: Bounding boxes (N, 4)
        class_names: List of class names
        alpha: Transparency for masks (0-1)
        show_boxes: Whether to draw bounding boxes
        show_labels: Whether to show labels
        
    Returns:
        PIL Image with overlays
    """
    # Convert to numpy array
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Create overlay
    overlay = img_array.copy()
    
    # Draw each mask
    for i in range(len(masks)):
        mask = masks[i].squeeze()
        
        # Resize mask to image size if needed
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Apply threshold
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        # Get color for this class
        color = get_color_for_class(int(labels[i]))
        
        # Create colored mask
        colored_mask = np.zeros_like(img_array)
        colored_mask[binary_mask == 1] = color
        
        # Blend with overlay
        overlay = cv2.addWeighted(overlay, 1.0, colored_mask, alpha, 0)
    
    # Convert back to PIL
    result = Image.fromarray(overlay)
    draw = ImageDraw.Draw(result)
    
    # Draw bounding boxes and labels
    if show_boxes or show_labels:
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        for i in range(len(boxes)):
            box = boxes[i]
            color = get_color_for_class(int(labels[i]))
            
            if show_boxes:
                # Draw box
                draw.rectangle([(box[0], box[1]), (box[2], box[3])], 
                             outline=color, width=3)
            
            if show_labels:
                # Draw label
                label_text = f"{class_names[i]}: {scores[i]:.2f}"
                text_bbox = draw.textbbox((box[0], box[1]), label_text, font=font)
                draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], 
                             fill=color)
                draw.text((box[0], box[1]), label_text, fill=(255, 255, 255), font=font)
    
    return result


def overlay_semantic_mask(image: Image.Image, seg_map: np.ndarray, 
                          class_info: List[Tuple[int, str]], alpha: float = 0.5) -> Image.Image:
    """
    Overlay semantic segmentation mask on image
    
    Args:
        image: Original PIL Image
        seg_map: Segmentation map (H, W)
        class_info: List of (class_id, class_name) tuples
        alpha: Transparency for masks (0-1)
        
    Returns:
        PIL Image with overlay
    """
    # Convert to numpy array
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Resize segmentation map if needed
    if seg_map.shape != (h, w):
        seg_map = cv2.resize(seg_map, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Create colored overlay
    colored_mask = np.zeros_like(img_array)
    
    for class_id, _ in class_info:
        mask = (seg_map == class_id)
        color = get_color_for_class(class_id)
        colored_mask[mask] = color
    
    # Blend
    overlay = cv2.addWeighted(img_array, 1.0, colored_mask, alpha, 0)
    
    return Image.fromarray(overlay)


def create_mask_legend(class_info: List[Tuple[int, str]], max_cols: int = 4) -> Image.Image:
    """
    Create a legend image showing classes and their colors
    
    Args:
        class_info: List of (class_id, class_name) tuples
        max_cols: Maximum columns in legend
        
    Returns:
        PIL Image of the legend
    """
    # Filter out background and N/A
    filtered_classes = [(cid, cname) for cid, cname in class_info 
                       if cname not in ['__background__', 'N/A']]
    
    if not filtered_classes:
        # Return empty legend
        return Image.new('RGB', (200, 50), color=(255, 255, 255))
    
    # Calculate layout
    n_classes = len(filtered_classes)
    n_cols = min(n_classes, max_cols)
    n_rows = (n_classes + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 0.5))
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each class
    for idx, (class_id, class_name) in enumerate(filtered_classes):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        color = get_color_for_class(class_id)
        color_rgb = tuple(c / 255.0 for c in color)
        
        ax.add_patch(patches.Rectangle((0, 0), 1, 1, facecolor=color_rgb))
        ax.text(0.5, 0.5, class_name, ha='center', va='center', 
               fontsize=8, color='white', weight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_classes, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    legend_img = Image.open(buf)
    plt.close()
    
    return legend_img


def highlight_caption_objects(image: Image.Image, caption: str, 
                             detected_objects: List[str], boxes: np.ndarray,
                             labels: List[str]) -> Image.Image:
    """
    Highlight objects in image that are mentioned in the caption
    
    Args:
        image: Original PIL Image
        caption: Generated caption text
        detected_objects: List of detected object names
        boxes: Bounding boxes for detected objects
        labels: Labels for detected objects
        
    Returns:
        PIL Image with highlighted objects
    """
    result = image.copy()
    draw = ImageDraw.Draw(result)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Find which detected objects are in caption
    caption_lower = caption.lower()
    
    for i, label in enumerate(labels):
        if label.lower() in caption_lower:
            box = boxes[i]
            # Highlight with thicker green box
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], 
                         outline=(0, 255, 0), width=5)
            
            # Add label
            draw.text((box[0], box[1] - 20), label, fill=(0, 255, 0), font=font)
    
    return result


def create_comparison_grid(images: List[Image.Image], titles: List[str], 
                          n_cols: int = 2) -> Image.Image:
    """
    Create a grid of images for comparison
    
    Args:
        images: List of PIL Images
        titles: List of titles for each image
        n_cols: Number of columns
        
    Returns:
        PIL Image containing the grid
    """
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    # Get max dimensions
    max_w = max(img.width for img in images)
    max_h = max(img.height for img in images)
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, (img, title) in enumerate(zip(images, titles)):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_images, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    grid_img = Image.open(buf)
    plt.close()
    
    return grid_img
