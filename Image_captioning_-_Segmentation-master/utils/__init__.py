"""
Utils Package

This package contains utility functions for visualization, COCO dataset
handling, and I/O operations.
"""

from .viz import (
    overlay_instance_masks,
    overlay_semantic_mask,
    create_mask_legend,
    highlight_caption_objects,
    create_comparison_grid
)

from .coco_utils import (
    load_coco_annotations,
    get_image_captions,
    get_image_segmentations,
    decode_rle,
    create_sample_manifest,
    COCOEvaluator
)

from .io import (
    export_caption_txt,
    export_mask_png,
    export_results_json,
    create_export_bundle,
    image_to_bytes,
    load_image_from_url,
    create_batch_export
)

__all__ = [
    # Visualization
    'overlay_instance_masks',
    'overlay_semantic_mask',
    'create_mask_legend',
    'highlight_caption_objects',
    'create_comparison_grid',
    
    # COCO utilities
    'load_coco_annotations',
    'get_image_captions',
    'get_image_segmentations',
    'decode_rle',
    'create_sample_manifest',
    'COCOEvaluator',
    
    # I/O utilities
    'export_caption_txt',
    'export_mask_png',
    'export_results_json',
    'create_export_bundle',
    'image_to_bytes',
    'load_image_from_url',
    'create_batch_export'
]
