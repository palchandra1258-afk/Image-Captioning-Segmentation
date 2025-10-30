"""
Inference Package

This package contains inference pipelines for image captioning
and segmentation tasks.
"""

from .captioning import CaptioningPipeline, caption_image
from .segmentation import SegmentationPipeline, segment_image, masks_to_rle

__all__ = [
    'CaptioningPipeline',
    'caption_image',
    'SegmentationPipeline',
    'segment_image',
    'masks_to_rle'
]
