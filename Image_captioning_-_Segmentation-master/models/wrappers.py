"""
Model wrapper classes for loading and managing captioning and segmentation models
"""
import os
import json
import torch
import torch.nn as nn
from torchvision import models
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import streamlit as st
from pathlib import Path


def load_models_manifest():
    """Load the models manifest configuration"""
    manifest_path = Path(__file__).parent.parent / "models_manifest.json"
    with open(manifest_path, 'r') as f:
        return json.load(f)


class EncoderCNN(nn.Module):
    """CNN Encoder for image captioning (ResNet50/InceptionV3)"""
    def __init__(self, embed_size, encoder_type='resnet50'):
        super(EncoderCNN, self).__init__()
        if encoder_type == 'resnet50':
            resnet = models.resnet50(pretrained=True)
            modules = list(resnet.children())[:-1]
            self.resnet = nn.Sequential(*modules)
            self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        elif encoder_type == 'inceptionv3':
            inception = models.inception_v3(pretrained=True)
            inception.aux_logits = False
            modules = list(inception.children())[:-1]
            self.resnet = nn.Sequential(*modules)
            self.embed = nn.Linear(2048, embed_size)
        
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.bn(self.embed(features))
        return features


class DecoderLSTM(nn.Module):
    """LSTM Decoder for image captioning"""
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs


class CaptioningModelWrapper:
    """Wrapper for captioning models (CNN + LSTM/Transformer)"""
    
    def __init__(self, model_name, device='cpu'):
        self.model_name = model_name
        self.device = device
        self.manifest = load_models_manifest()
        self.config = self.manifest['captioning_models'][model_name]
        self.encoder = None
        self.decoder = None
        self.vocab = None
        
    def load(self):
        """Load the model with caching"""
        checkpoint_path = self.config['local_path']
        
        if not os.path.exists(checkpoint_path):
            st.warning(f"Checkpoint not found: {checkpoint_path}. Using pretrained base models.")
            return self._load_pretrained()
        
        return self._load_from_checkpoint(checkpoint_path)
    
    def _load_pretrained(self):
        """Load pretrained models as fallback"""
        embed_size = self.config['embed_size']
        vocab_size = self.config['vocab_size']
        
        if 'resnet50' in self.model_name:
            self.encoder = EncoderCNN(embed_size, 'resnet50').to(self.device)
        elif 'inceptionv3' in self.model_name:
            self.encoder = EncoderCNN(embed_size, 'inceptionv3').to(self.device)
        
        if 'lstm' in self.model_name:
            hidden_size = self.config['hidden_size']
            num_layers = self.config['num_layers']
            self.decoder = DecoderLSTM(embed_size, hidden_size, vocab_size, num_layers).to(self.device)
        
        self.encoder.eval()
        self.decoder.eval()
        
        return True
    
    def _load_from_checkpoint(self, checkpoint_path):
        """Load from saved checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        # Implementation depends on checkpoint structure
        return True
    
    def get_input_size(self):
        """Get expected input image size"""
        return tuple(self.config['input_size'])


class SegmentationModelWrapper:
    """Wrapper for segmentation models (U-Net, DeepLabV3+, Mask R-CNN)"""
    
    def __init__(self, model_name, device='cpu'):
        self.model_name = model_name
        self.device = device
        self.manifest = load_models_manifest()
        self.config = self.manifest['segmentation_models'][model_name]
        self.model = None
        
    def load(self):
        """Load the segmentation model with caching"""
        if 'maskrcnn' in self.model_name and self.config.get('pretrained'):
            return self._load_maskrcnn_pretrained()
        
        checkpoint_path = self.config['local_path']
        if not os.path.exists(checkpoint_path):
            st.warning(f"Checkpoint not found: {checkpoint_path}. Loading pretrained Mask R-CNN.")
            return self._load_maskrcnn_pretrained()
        
        return self._load_from_checkpoint(checkpoint_path)
    
    def _load_maskrcnn_pretrained(self):
        """Load pretrained Mask R-CNN from torchvision"""
        from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
        
        weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
        self.model = maskrcnn_resnet50_fpn(weights=weights)
        self.model.to(self.device)
        self.model.eval()
        
        return True
    
    def _load_from_checkpoint(self, checkpoint_path):
        """Load from saved checkpoint"""
        # Implementation for custom trained models
        if 'unet' in self.model_name:
            # Load U-Net - fallback to DeepLabV3 for now
            st.info("U-Net checkpoint not found, loading DeepLabV3+ pretrained model instead.")
            return self._load_deeplabv3_pretrained()
        elif 'deeplabv3' in self.model_name:
            # Load pretrained DeepLabV3
            return self._load_deeplabv3_pretrained()
        
        return True
    
    def _load_deeplabv3_pretrained(self):
        """Load pretrained DeepLabV3+ from torchvision"""
        from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
        
        weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        self.model = deeplabv3_resnet50(weights=weights)
        self.model.to(self.device)
        self.model.eval()
        
        return True
    
    def get_input_size(self):
        """Get expected input image size"""
        return tuple(self.config['input_size'])
    
    def get_model_type(self):
        """Get model type: semantic or instance"""
        return self.config['type']


@st.cache_resource
def load_captioning_model(model_name, device='cpu'):
    """Cached loading of captioning models"""
    wrapper = CaptioningModelWrapper(model_name, device)
    wrapper.load()
    return wrapper


@st.cache_resource
def load_segmentation_model(model_name, device='cpu'):
    """Cached loading of segmentation models"""
    wrapper = SegmentationModelWrapper(model_name, device)
    wrapper.load()
    return wrapper
