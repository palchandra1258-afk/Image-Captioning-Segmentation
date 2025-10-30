"""
Unit tests for inference pipelines
"""
import pytest
import torch
from PIL import Image
import numpy as np
from models.wrappers import CaptioningModelWrapper, SegmentationModelWrapper
from inference.captioning import CaptioningPipeline
from inference.segmentation import SegmentationPipeline, masks_to_rle


@pytest.fixture
def test_image():
    """Create a test image"""
    return Image.new('RGB', (224, 224), color='red')


def test_captioning_preprocessing(test_image):
    """Test image preprocessing for captioning"""
    from models.wrappers import EncoderCNN
    encoder = EncoderCNN(embed_size=512, encoder_type='resnet50')
    wrapper = CaptioningModelWrapper('resnet50_lstm', device='cpu')
    wrapper.encoder = encoder
    
    pipeline = CaptioningPipeline(wrapper, device='cpu')
    
    tensor = pipeline.preprocess(test_image)
    
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape[0] == 1  # Batch size
    assert tensor.shape[1] == 3  # RGB channels
    assert tensor.device.type == 'cpu'


def test_segmentation_preprocessing(test_image):
    """Test image preprocessing for segmentation"""
    wrapper = SegmentationModelWrapper('maskrcnn', device='cpu')
    pipeline = SegmentationPipeline(wrapper, device='cpu')
    
    tensor, original_size = pipeline.preprocess(test_image)
    
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape[0] == 1  # Batch size
    assert tensor.shape[1] == 3  # RGB channels
    assert original_size == test_image.size


def test_masks_to_rle():
    """Test RLE encoding of masks"""
    # Create simple binary masks
    masks = np.array([
        [[0, 0, 1, 1], [0, 0, 1, 1]],
        [[1, 0, 0, 1], [1, 0, 0, 1]]
    ], dtype=np.uint8)
    
    rles = masks_to_rle(masks)
    
    assert isinstance(rles, list)
    assert len(rles) == 2
    assert 'counts' in rles[0]
    assert 'size' in rles[0]


def test_caption_metrics():
    """Test caption evaluation metrics"""
    from inference.captioning import CaptioningPipeline
    from models.wrappers import EncoderCNN, DecoderLSTM
    
    encoder = EncoderCNN(embed_size=512, encoder_type='resnet50')
    decoder = DecoderLSTM(embed_size=512, hidden_size=512, vocab_size=1000)
    
    wrapper = CaptioningModelWrapper('resnet50_lstm', device='cpu')
    wrapper.encoder = encoder
    wrapper.decoder = decoder
    
    pipeline = CaptioningPipeline(wrapper, device='cpu')
    
    generated = "a dog playing in the park"
    references = [
        "a brown dog running on grass",
        "a dog playing outdoors",
        "an animal in a park"
    ]
    
    metrics = pipeline.calculate_metrics(generated, references)
    
    assert 'BLEU-1' in metrics
    assert 'BLEU-2' in metrics
    assert 'BLEU-3' in metrics
    assert 'BLEU-4' in metrics
    assert 'CIDEr' in metrics
    assert all(0 <= v <= 1 for v in [metrics['BLEU-1'], metrics['BLEU-2'], 
                                      metrics['BLEU-3'], metrics['BLEU-4']])


def test_segmentation_metrics():
    """Test segmentation evaluation metrics"""
    pipeline = SegmentationPipeline(
        SegmentationModelWrapper('maskrcnn', device='cpu'),
        device='cpu'
    )
    
    # Create test masks
    pred_mask = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 2, 2]])
    gt_mask = np.array([[0, 0, 1, 1], [0, 1, 1, 1], [2, 2, 2, 2]])
    
    metrics = pipeline.calculate_metrics(pred_mask, gt_mask)
    
    assert 'mIoU' in metrics
    assert 'Pixel Accuracy' in metrics
    assert 'Dice' in metrics
    assert all(0 <= v <= 1 for v in metrics.values())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
