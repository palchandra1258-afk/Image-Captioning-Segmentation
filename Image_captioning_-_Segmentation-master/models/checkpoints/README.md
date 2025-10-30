# Model Checkpoints

This directory contains pre-trained model weights for captioning and segmentation models.

## Download Instructions

### Automatic Download (Recommended)
Some models will be automatically downloaded from PyTorch Hub on first use:
- Mask R-CNN (torchvision pretrained)

### Manual Download
For custom-trained models, download from the links in `models_manifest.json` and place them here.

### Directory Structure

```
checkpoints/
├── resnet50_lstm_coco.pth          # ResNet50 + LSTM captioning model
├── inceptionv3_transformer_coco.pth # InceptionV3 + Transformer model
├── unet_coco_semantic.pth          # U-Net segmentation model
├── deeplabv3plus_coco.pth          # DeepLabV3+ model
└── maskrcnn_coco.pth               # Mask R-CNN (or auto-downloaded)
```

## File Sizes

Expected checkpoint sizes:
- ResNet50 + LSTM: ~100-200 MB
- InceptionV3 + Transformer: ~200-300 MB
- U-Net: ~50-100 MB
- DeepLabV3+: ~200-250 MB
- Mask R-CNN: ~150-200 MB

**Note**: This directory is excluded from version control (.gitignore) due to large file sizes.

## Cloud Storage Alternative

For deployment, consider hosting checkpoints on:
- AWS S3
- Google Cloud Storage
- Azure Blob Storage
- Hugging Face Model Hub

Then update the `checkpoint_url` fields in `models_manifest.json`.
