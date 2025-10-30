"""
Unit tests for visualization utilities
"""
import pytest
import numpy as np
from PIL import Image
from utils.viz import (get_color_for_class, overlay_instance_masks, 
                       overlay_semantic_mask, create_mask_legend)


def test_get_color_for_class():
    """Test color assignment for classes"""
    color1 = get_color_for_class(0)
    color2 = get_color_for_class(1)
    
    assert len(color1) == 3
    assert len(color2) == 3
    assert color1 != color2
    
    # Test consistency
    assert get_color_for_class(0) == get_color_for_class(0)


def test_overlay_instance_masks():
    """Test instance mask overlay"""
    # Create test image
    image = Image.new('RGB', (100, 100), color='white')
    
    # Create test masks
    masks = np.random.rand(2, 1, 100, 100) > 0.5
    labels = np.array([1, 2])
    scores = np.array([0.9, 0.8])
    boxes = np.array([[10, 10, 50, 50], [60, 60, 90, 90]])
    class_names = ['person', 'car']
    
    result = overlay_instance_masks(
        image, masks, labels, scores, boxes, class_names,
        alpha=0.5, show_boxes=True, show_labels=True
    )
    
    assert isinstance(result, Image.Image)
    assert result.size == image.size


def test_overlay_semantic_mask():
    """Test semantic mask overlay"""
    # Create test image
    image = Image.new('RGB', (100, 100), color='white')
    
    # Create test segmentation map
    seg_map = np.random.randint(0, 5, (100, 100))
    class_info = [(0, 'background'), (1, 'person'), (2, 'car')]
    
    result = overlay_semantic_mask(image, seg_map, class_info, alpha=0.5)
    
    assert isinstance(result, Image.Image)
    assert result.size == image.size


def test_create_mask_legend():
    """Test mask legend creation"""
    class_info = [(1, 'person'), (2, 'car'), (3, 'dog')]
    
    legend = create_mask_legend(class_info, max_cols=2)
    
    assert isinstance(legend, Image.Image)
    assert legend.size[0] > 0
    assert legend.size[1] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
