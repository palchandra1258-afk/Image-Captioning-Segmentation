"""
Image segmentation inference pipeline
"""
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from typing import Tuple, List, Dict
import cv2


class SegmentationPipeline:
    """Pipeline for image segmentation inference"""
    
    def __init__(self, model_wrapper, device='cpu'):
        self.model_wrapper = model_wrapper
        self.device = device
        self.transform = self._get_transform()
        self.class_names = self._load_class_names()
        
    def _get_transform(self):
        """Get preprocessing transforms"""
        return T.Compose([
            T.ToTensor()
        ])
    
    def _load_class_names(self):
        """Load COCO class names"""
        # COCO 80 classes + background
        return [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def preprocess(self, image: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Preprocess image for segmentation"""
        original_size = image.size
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_tensor = self.transform(image)
        return image_tensor.unsqueeze(0).to(self.device), original_size
    
    def segment_instance(self, image: Image.Image, threshold=0.5) -> Dict:
        """
        Perform instance segmentation (e.g., Mask R-CNN)
        
        Args:
            image: PIL Image
            threshold: Confidence threshold for detections
            
        Returns:
            Dictionary with masks, boxes, labels, scores
        """
        image_tensor, original_size = self.preprocess(image)
        
        with torch.no_grad():
            predictions = self.model_wrapper.model(image_tensor)
        
        # Extract predictions
        pred = predictions[0]
        
        # Filter by threshold
        keep = pred['scores'] > threshold
        
        result = {
            'masks': pred['masks'][keep].cpu().numpy(),
            'boxes': pred['boxes'][keep].cpu().numpy(),
            'labels': pred['labels'][keep].cpu().numpy(),
            'scores': pred['scores'][keep].cpu().numpy(),
            'class_names': [self.class_names[i] for i in pred['labels'][keep].cpu().numpy()],
            'original_size': original_size
        }
        
        return result
    
    def segment_semantic(self, image: Image.Image) -> Dict:
        """
        Perform semantic segmentation (e.g., DeepLabV3+, U-Net)
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary with segmentation map and class information
        """
        image_tensor, original_size = self.preprocess(image)
        
        with torch.no_grad():
            output = self.model_wrapper.model(image_tensor)
            
            if isinstance(output, dict):
                # DeepLabV3 output
                output = output['out']
            
            # Get predicted class for each pixel
            seg_map = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        # Get unique classes present
        unique_classes = np.unique(seg_map)
        class_info = [(int(c), self.class_names[c]) for c in unique_classes if c < len(self.class_names)]
        
        result = {
            'seg_map': seg_map,
            'classes': class_info,
            'original_size': original_size
        }
        
        return result
    
    def calculate_metrics(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
        """
        Calculate segmentation metrics (mIoU, pixel accuracy, Dice)
        
        Args:
            pred_mask: Predicted segmentation mask
            gt_mask: Ground truth mask
            
        Returns:
            Dictionary with metric scores
        """
        # Flatten arrays
        pred_flat = pred_mask.flatten()
        gt_flat = gt_mask.flatten()
        
        # Pixel accuracy
        pixel_acc = np.mean(pred_flat == gt_flat)
        
        # Calculate IoU for each class
        classes = np.unique(np.concatenate([pred_flat, gt_flat]))
        ious = []
        
        for cls in classes:
            pred_cls = pred_flat == cls
            gt_cls = gt_flat == cls
            
            intersection = np.logical_and(pred_cls, gt_cls).sum()
            union = np.logical_or(pred_cls, gt_cls).sum()
            
            if union > 0:
                iou = intersection / union
                ious.append(iou)
        
        miou = np.mean(ious) if ious else 0.0
        
        # Dice coefficient (F1 score)
        intersection = np.logical_and(pred_flat > 0, gt_flat > 0).sum()
        dice = (2.0 * intersection) / (np.sum(pred_flat > 0) + np.sum(gt_flat > 0) + 1e-7)
        
        return {
            'mIoU': miou,
            'Pixel Accuracy': pixel_acc,
            'Dice': dice
        }


def segment_image(model_wrapper, image: Image.Image, threshold=0.5, device='cpu'):
    """
    High-level function to segment an image
    
    Args:
        model_wrapper: Loaded segmentation model wrapper
        image: PIL Image
        threshold: Confidence threshold
        device: Device to run inference on
        
    Returns:
        Segmentation results dictionary
    """
    pipeline = SegmentationPipeline(model_wrapper, device)
    
    model_type = model_wrapper.get_model_type()
    
    if model_type == 'instance':
        return pipeline.segment_instance(image, threshold)
    else:
        return pipeline.segment_semantic(image)


def masks_to_rle(masks: np.ndarray) -> List[Dict]:
    """
    Convert binary masks to RLE (Run-Length Encoding) format
    
    Args:
        masks: Array of binary masks
        
    Returns:
        List of RLE dictionaries
    """
    rles = []
    
    for mask in masks:
        # Ensure binary mask
        binary_mask = (mask > 0.5).astype(np.uint8).squeeze()
        
        # Flatten in column-major order
        pixels = binary_mask.T.flatten()
        
        # Compute RLE
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        
        rle = {
            'counts': runs.tolist(),
            'size': list(binary_mask.shape)
        }
        rles.append(rle)
    
    return rles
