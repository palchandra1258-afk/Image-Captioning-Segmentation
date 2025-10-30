"""
COCO dataset utilities
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


def load_coco_annotations(annotations_path: str) -> Dict:
    """
    Load COCO annotations from JSON file
    
    Args:
        annotations_path: Path to COCO annotations JSON
        
    Returns:
        Dictionary with annotations
    """
    with open(annotations_path, 'r') as f:
        return json.load(f)


def get_image_captions(coco_data: Dict, image_id: int) -> List[str]:
    """
    Get all captions for a specific image
    
    Args:
        coco_data: COCO annotations dictionary
        image_id: Image ID
        
    Returns:
        List of caption strings
    """
    if 'annotations' not in coco_data:
        return []
    
    captions = []
    for ann in coco_data['annotations']:
        if ann['image_id'] == image_id:
            captions.append(ann['caption'])
    
    return captions


def get_image_segmentations(coco_data: Dict, image_id: int) -> List[Dict]:
    """
    Get all segmentation annotations for a specific image
    
    Args:
        coco_data: COCO annotations dictionary
        image_id: Image ID
        
    Returns:
        List of segmentation annotation dictionaries
    """
    if 'annotations' not in coco_data:
        return []
    
    segmentations = []
    for ann in coco_data['annotations']:
        if ann['image_id'] == image_id and 'segmentation' in ann:
            segmentations.append(ann)
    
    return segmentations


def decode_rle(rle: Dict) -> np.ndarray:
    """
    Decode RLE (Run-Length Encoding) to binary mask
    
    Args:
        rle: RLE dictionary with 'counts' and 'size'
        
    Returns:
        Binary mask as numpy array
    """
    h, w = rle['size']
    counts = rle['counts']
    
    # Decode RLE
    mask = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    val = 0
    
    if isinstance(counts, list):
        for count in counts:
            mask[pos:pos+count] = val
            pos += count
            val = 1 - val
    
    return mask.reshape((h, w), order='F')


def create_sample_manifest() -> List[Dict]:
    """
    Create a manifest of sample COCO images
    
    Returns:
        List of sample image metadata dictionaries
    """
    return [
        {
            'id': 1,
            'file_name': 'sample1.jpg',
            'width': 640,
            'height': 480,
            'captions': [
                'A person standing on a street.',
                'A man on a city sidewalk.',
                'Someone walking down the street.'
            ]
        },
        {
            'id': 2,
            'file_name': 'sample2.jpg',
            'width': 640,
            'height': 480,
            'captions': [
                'A dog playing in the park.',
                'A brown dog running on grass.',
                'An animal outdoors.'
            ]
        },
        {
            'id': 3,
            'file_name': 'sample3.jpg',
            'width': 800,
            'height': 600,
            'captions': [
                'A car parked on the street.',
                'A vehicle near a building.',
                'An automobile in an urban setting.'
            ]
        }
    ]


class COCOEvaluator:
    """Evaluator for COCO metrics"""
    
    def __init__(self):
        self.results = []
    
    def add_prediction(self, image_id: int, prediction: Dict):
        """Add a prediction for evaluation"""
        self.results.append({
            'image_id': image_id,
            'prediction': prediction
        })
    
    def compute_metrics(self) -> Dict:
        """Compute evaluation metrics"""
        # Placeholder for actual COCO evaluation
        return {
            'mAP': 0.0,
            'mIoU': 0.0,
            'BLEU-4': 0.0
        }
