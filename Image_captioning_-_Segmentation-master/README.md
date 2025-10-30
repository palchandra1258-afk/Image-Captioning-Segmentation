# Image Captioning & Segmentation

[![CI/CD](https://github.com/username/image-caption-seg/actions/workflows/ci.yml/badge.svg)](https://github.com/username/image-caption-seg/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Production-quality Streamlit web application for Image Captioning and Segmentation using COCO 2014 dataset with state-of-the-art deep learning models.

## 🎯 Project Overview

This project combines **image captioning** and **image segmentation** into a unified pipeline, employing deep learning models such as CNNs, LSTMs, Transformers, and Mask R-CNN for accurate object labeling and text description generation.

### Key Features

- 🖼️ **Dual Functionality**: Both image captioning and segmentation in one app
- 🤖 **Multiple Models**: Support for ResNet50+LSTM, InceptionV3+Transformer, U-Net, DeepLabV3+, Mask R-CNN
- 🔗 **Combined Pipeline**: Synchronized caption and segmentation with object linking
- 📦 **Batch Processing**: Process multiple images at once
- 🎨 **Interactive UI**: Modern Streamlit interface with real-time visualization
- 🐳 **Docker Ready**: CPU and GPU-enabled containers
- ✅ **Production Quality**: Comprehensive testing, CI/CD, and error handling

## 📊 Models

### Captioning Models
- **ResNet50 + LSTM**: CNN encoder with LSTM decoder
- **InceptionV3 + Transformer**: InceptionV3 encoder with Transformer decoder

### Segmentation Models
- **Mask R-CNN** (Instance): Pretrained on COCO 2014
- **DeepLabV3+** (Semantic): Advanced semantic segmentation
- **U-Net** (Semantic): Medical imaging architecture adapted for COCO

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip or conda
- (Optional) CUDA-capable GPU for faster inference
- (Optional) Docker for containerized deployment

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/image-caption-seg.git
cd image-caption-seg
```

2. **Create virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download NLTK data** (required for captioning):
```bash
python -c "import nltk; nltk.download('punkt')"
```

### Running the Application

**Local Development**:
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

**With Docker (CPU)**:
```bash
docker-compose up app
```

**With Docker (GPU)**:
```bash
docker-compose --profile gpu up app-gpu
```

## 📁 Project Structure

```
image-caption-seg-streamlit/
├── app.py                      # Main Streamlit application
├── app_continuation.py         # Additional render functions
├── requirements.txt            # Python dependencies
├── models_manifest.json        # Model configuration
├── Dockerfile                  # Docker config (CPU)
├── Dockerfile.gpu              # Docker config (GPU)
├── docker-compose.yml          # Docker Compose config
├── README.md                   # This file
│
├── models/
│   ├── checkpoints/            # Model weights (download separately)
│   └── wrappers.py             # Model loading wrappers
│
├── inference/
│   ├── captioning.py           # Captioning inference pipeline
│   └── segmentation.py         # Segmentation inference pipeline
│
├── utils/
│   ├── viz.py                  # Visualization utilities
│   ├── coco_utils.py           # COCO dataset helpers
│   └── io.py                   # I/O and export utilities
│
├── static/
│   ├── custom.css              # Custom styling
│   └── samples/                # Sample COCO images (optional)
│
├── tests/
│   ├── test_viz.py             # Visualization tests
│   └── test_inference.py       # Inference pipeline tests
│
└── .github/
    └── workflows/
        └── ci.yml              # GitHub Actions CI/CD
```

## 🎮 Usage Guide

### 1. Landing / About Page
- View project abstract, objectives, and dataset information
- Access GitHub repositories and documentation
- View team information and references

### 2. Image Upload
Choose from three upload methods:
- **File Upload**: PNG, JPG, JPEG (max 10MB)
- **Sample Dataset**: Select from provided COCO 2014 samples
- **URL**: Load image from web URL

### 3. Captioning Module
- Select model: ResNet50+LSTM or InceptionV3+Transformer
- Adjust beam search width (1-5)
- Set maximum caption length (10-30)
- Generate captions with confidence scores
- View token-level probabilities
- Calculate BLEU/CIDEr metrics with reference captions

### 4. Segmentation Module
- Choose model: Mask R-CNN, DeepLabV3+, or U-Net
- Adjust confidence threshold (instance segmentation)
- Control mask transparency (0-1)
- Toggle bounding boxes and labels
- View class-wise legend
- Inspect detection details and raw mask arrays

### 5. Combined Pipeline
- Run both captioning and segmentation
- Highlight objects mentioned in captions
- View synchronized results
- Export complete bundle (caption.txt, mask.png, results.json)

### 6. Batch Processing
- Upload multiple images
- Choose processing mode:
  - Captioning Only
  - Segmentation Only
  - Both
- View progress and results table
- Download batch results as ZIP

### 7. Developer Mode
- View model load times
- Monitor memory usage
- Inspect intermediate feature maps
- Analyze token probabilities
- Debug segmentation masks

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/test_viz.py -v
```

## 📦 Model Checkpoints

### Download Pre-trained Models

**Mask R-CNN** (Automatic):
- Automatically downloaded from PyTorch Hub on first use

**Custom Models** (Manual):
1. Download from the provided links in `models_manifest.json`
2. Place in `models/checkpoints/` directory

### Model Manifest Structure

```json
{
  "captioning_models": {
    "model_name": {
      "checkpoint_url": "https://...",
      "local_path": "models/checkpoints/model.pth",
      "input_size": [224, 224],
      ...
    }
  },
  "segmentation_models": {
    ...
  }
}
```

## 🐳 Docker Deployment

### Build Images

```bash
# CPU version
docker build -t image-caption-seg:latest -f Dockerfile .

# GPU version
docker build -t image-caption-seg:gpu -f Dockerfile.gpu .
```

### Run Containers

```bash
# CPU
docker run -p 8501:8501 image-caption-seg:latest

# GPU (requires nvidia-docker)
docker run --gpus all -p 8501:8501 image-caption-seg:gpu
```

### Using Docker Compose

```bash
# CPU version
docker-compose up app

# GPU version
docker-compose --profile gpu up app-gpu
```

## 🌐 Deployment to Streamlit Cloud

1. Push code to GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Create new app and link to your repository
4. Configure Python version (3.10+)
5. Set `app.py` as the main file
6. Deploy!

**Note**: For models, use cloud storage (S3, GCS) and download on startup.

## 🎨 Customization

### Custom CSS
Edit `static/custom.css` to customize the UI theme and styling.

### Adding New Models
1. Update `models_manifest.json` with new model config
2. Implement model wrapper in `models/wrappers.py`
3. Update inference pipeline if needed
4. Add model to UI dropdowns

## 📊 Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI Layer                    │
├─────────────────────────────────────────────────────────┤
│  Landing  │ Captioning │ Segmentation │ Combined │ Batch │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                  Inference Pipelines                     │
├───────────────────────────┬─────────────────────────────┤
│   Captioning Pipeline     │   Segmentation Pipeline     │
│  - Preprocessing          │  - Preprocessing            │
│  - Feature Extraction     │  - Inference                │
│  - Beam Search           │  - Postprocessing           │
│  - Metrics               │  - Metrics                  │
└───────────────────────────┴─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                    Model Wrappers                        │
├───────────────────────────┬─────────────────────────────┤
│  ResNet50+LSTM           │   Mask R-CNN                │
│  InceptionV3+Transformer │   DeepLabV3+                │
│                          │   U-Net                     │
└───────────────────────────┴─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                  Utility Modules                         │
├─────────────────┬─────────────────┬─────────────────────┤
│  Visualization  │   COCO Utils    │    I/O Utils        │
│  - Overlays     │   - Annotations │    - Exports        │
│  - Legends      │   - Metrics     │    - Bundles        │
└─────────────────┴─────────────────┴─────────────────────┘
```

## 👥 Team

- **Brunda B** - [GitHub](https://github.com/Brunda292005)
  - [Image Captioning](https://github.com/Brunda292005/Image_Captioning.git)
  - [Image Segmentation](https://github.com/Brunda292005/Image_-Segmentation.git)

- **Jaromi D** - [GitHub](https://github.com/jaromi-joe)
  - [Image Captioning](https://github.com/jaromi-joe/Image_captioning.git)
  - [Image Segmentation](https://github.com/jaromi-joe/Image_segmentation.git)

**Supervisor**: Chandan Mishra

## 📚 References

- [COCO Dataset](https://cocodataset.org/)
- [Mishra et al. (2024) - NLP Literature Review](https://www.researchgate.net/publication/381295446)
- [Anthony, G.S. (2024) - NLP Framework for Banking Skills](University of Salford MPhil thesis)
- [Sahu & Sood (2019) - CV Classification using NLP](Jaypee University B.Tech thesis)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- COCO Dataset team for providing the benchmark dataset
- PyTorch and Hugging Face teams for excellent frameworks
- Streamlit for the amazing web framework
- Open-source community for various libraries and tools

## 🐛 Issues and Support

For bugs, feature requests, or questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Contribute via pull requests

## 🚀 Future Enhancements

- [ ] Add more captioning models (BLIP, CLIP-based)
- [ ] Support for video captioning and segmentation
- [ ] Real-time webcam input
- [ ] Multi-language caption support
- [ ] Advanced evaluation metrics dashboard
- [ ] Model fine-tuning interface
- [ ] Cloud deployment guides (AWS, Azure, GCP)

---

**Built with ❤️ using Python, PyTorch, and Streamlit**
