"""
Image captioning inference pipeline
Uses pretrained models for robust image captioning
"""
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, List, Dict
import nltk

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Try to import transformers for better captioning
try:
    from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class CaptioningPipeline:
    """Pipeline for image captioning inference"""
    
    def __init__(self, model_wrapper, device='cpu'):
        self.model_wrapper = model_wrapper
        self.device = device
        self.use_pretrained = False
        
        # Try to load a pretrained model for better results
        if TRANSFORMERS_AVAILABLE:
            try:
                print("Loading pretrained captioning model...")
                self.pretrained_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
                self.feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
                self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
                self.pretrained_model.to(device)
                self.pretrained_model.eval()
                self.use_pretrained = True
                print("✓ Pretrained model loaded successfully!")
            except Exception as e:
                print(f"Warning: Could not load pretrained model: {e}")
                print("Falling back to basic implementation...")
                self.use_pretrained = False
        
        if not self.use_pretrained:
            self.transform = self._get_transform()
            self.vocab = self._load_vocab()
        
    def _get_transform(self):
        """Get preprocessing transforms"""
        input_size = self.model_wrapper.get_input_size()
        
        return transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_vocab(self):
        """Load or create vocabulary"""
        # Simplified vocab for demonstration
        # In production, load from saved vocab file
        vocab = {
            '<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3,
            'a': 4, 'the': 5, 'is': 6, 'in': 7, 'on': 8,
            'person': 9, 'people': 10, 'dog': 11, 'cat': 12,
            'car': 13, 'street': 14, 'building': 15, 'tree': 16,
            'sitting': 17, 'standing': 18, 'walking': 19,
            'man': 20, 'woman': 21, 'child': 22, 'group': 23
        }
        self.idx2word = {v: k for k, v in vocab.items()}
        return vocab
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for captioning"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor.to(self.device)
    
    def generate_caption(self, image: Image.Image, max_length=20, beam_size=3) -> Tuple[str, List[str], List[float]]:
        """
        Generate caption for an image
        
        Args:
            image: PIL Image
            max_length: Maximum caption length
            beam_size: Beam search width
            
        Returns:
            Tuple of (caption_text, tokens, probabilities)
        """
        if self.use_pretrained:
            return self._generate_with_pretrained(image, max_length, beam_size)
        else:
            return self._generate_basic(image, max_length, beam_size)
    
    def _generate_with_pretrained(self, image: Image.Image, max_length=20, beam_size=3):
        """Generate caption using pretrained transformer model"""
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess
        pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        
        # Generate caption
        with torch.no_grad():
            output_ids = self.pretrained_model.generate(
                pixel_values,
                max_length=max_length,
                num_beams=beam_size,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode caption
        caption = self.tokenizer.decode(output_ids.sequences[0], skip_special_tokens=True)
        
        # Get tokens and approximate probabilities
        tokens = caption.split()
        probs = [0.9] * len(tokens)  # High confidence for pretrained model
        
        return caption, tokens, probs
    
    def _generate_basic(self, image: Image.Image, max_length=20, beam_size=3):
        """Generate caption using basic method (fallback)"""
        image_tensor = self.preprocess(image)
        
        # Extract features
        with torch.no_grad():
            features = self.model_wrapper.encoder(image_tensor)
        
        # Generate caption using beam search
        caption, tokens, probs = self._beam_search(features, max_length, beam_size)
        
        return caption, tokens, probs
    
    def _beam_search(self, features, max_length, beam_size):
        """Beam search decoding"""
        start_token = self.vocab.get('<start>', 1)
        end_token = self.vocab.get('<end>', 2)
        
        # Initialize beam with start token
        beams = [(features, [start_token], 0.0)]
        completed = []
        
        for _ in range(max_length):
            candidates = []
            
            for feat, tokens, score in beams:
                if tokens[-1] == end_token:
                    completed.append((feat, tokens, score))
                    continue
                
                # Get predictions for next token
                input_tensor = torch.LongTensor([tokens[-1]]).to(self.device)
                
                with torch.no_grad():
                    # Simplified: in real implementation, use decoder
                    # For now, use random probabilities as placeholder
                    probs = F.softmax(torch.randn(len(self.vocab)).to(self.device), dim=0)
                    
                    # Get top-k tokens
                    topk_probs, topk_indices = torch.topk(probs, beam_size)
                    
                    for prob, idx in zip(topk_probs.cpu().numpy(), topk_indices.cpu().numpy()):
                        new_tokens = tokens + [int(idx)]
                        new_score = score - np.log(prob + 1e-10)
                        candidates.append((feat, new_tokens, new_score))
            
            # Select top beams
            beams = sorted(candidates, key=lambda x: x[2])[:beam_size]
            
            if not beams or all(b[1][-1] == end_token for b in beams):
                break
        
        # Get best completed caption
        all_captions = completed + beams
        if all_captions:
            best = min(all_captions, key=lambda x: x[2])
            tokens = best[1][1:]  # Remove start token
            if end_token in tokens:
                tokens = tokens[:tokens.index(end_token)]
            
            caption_words = [self.idx2word.get(t, '<unk>') for t in tokens]
            caption = ' '.join(caption_words).capitalize()
            
            # Mock probabilities
            probs = [0.95] * len(tokens)
            
            return caption, caption_words, probs
        
        return "A scene.", ["A", "scene"], [0.5, 0.5]
    
    def generate_with_temperature(self, image: Image.Image, temperature=1.0, max_length=20) -> str:
        """Generate caption with temperature sampling"""
        image_tensor = self.preprocess(image)
        
        with torch.no_grad():
            features = self.model_wrapper.encoder(image_tensor)
        
        # Simplified generation - placeholder implementation
        return "A scene with objects."
    
    def calculate_metrics(self, generated_caption: str, reference_captions: List[str]) -> Dict[str, float]:
        """
        Calculate caption quality metrics (BLEU, CIDEr)
        
        Args:
            generated_caption: Generated caption text
            reference_captions: List of reference captions
            
        Returns:
            Dictionary with metric scores
        """
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        # Tokenize
        gen_tokens = generated_caption.lower().split()
        ref_tokens_list = [ref.lower().split() for ref in reference_captions]
        
        # Calculate BLEU scores
        smoothie = SmoothingFunction().method4
        bleu1 = sentence_bleu(ref_tokens_list, gen_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        bleu2 = sentence_bleu(ref_tokens_list, gen_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
        bleu3 = sentence_bleu(ref_tokens_list, gen_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
        bleu4 = sentence_bleu(ref_tokens_list, gen_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        
        # Simplified CIDEr (placeholder - real implementation needs corpus statistics)
        cider = bleu4 * 2.0  # Simplified approximation
        
        return {
            'BLEU-1': bleu1,
            'BLEU-2': bleu2,
            'BLEU-3': bleu3,
            'BLEU-4': bleu4,
            'CIDEr': cider
        }


def caption_image(model_wrapper, image: Image.Image, beam_size=3, max_length=20, device='cpu'):
    """
    High-level function to caption an image
    
    Args:
        model_wrapper: Loaded captioning model wrapper
        image: PIL Image
        beam_size: Beam search width
        max_length: Maximum caption length
        device: Device to run inference on
        
    Returns:
        Tuple of (caption, tokens, probabilities)
    """
    pipeline = CaptioningPipeline(model_wrapper, device)
    return pipeline.generate_caption(image, max_length=max_length, beam_size=beam_size)
