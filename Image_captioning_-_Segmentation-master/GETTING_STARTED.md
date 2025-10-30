# Getting Started with Image Captioning & Segmentation

Welcome! This guide will help you get the app up and running in **5 minutes**.

## 🚀 Quick Start (Windows)

### Option 1: Automated Setup (Recommended)

Simply run the setup script:

```powershell
.\setup.ps1
```

This will:
- ✅ Check Python version
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Download NLTK data
- ✅ Create necessary directories
- ✅ Check for GPU support
- ✅ Optionally start the app

### Option 2: Manual Setup

1. **Create virtual environment**:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. **Install dependencies**:
```powershell
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

3. **Run the app**:
```powershell
streamlit run app.py
```

4. **Open your browser** to: http://localhost:8501

## 📋 Prerequisites

- **Python**: 3.10 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Disk Space**: 2GB for dependencies + models
- **GPU**: Optional (CUDA-capable GPU for faster inference)

## 🎮 First Time Usage

### 1. Start the App
```powershell
streamlit run app.py
```

### 2. Navigate the Interface

The sidebar shows 6 sections:

- **🏠 Landing / About**: Project overview and information
- **💬 Captioning**: Generate image captions
- **🎨 Segmentation**: Detect and segment objects
- **🔗 Combined Pipeline**: Run both together
- **📦 Batch Processing**: Process multiple images
- **🛠️ Developer Mode**: Debug and performance insights

### 3. Upload an Image

Choose from three methods:
- **File Upload**: Drag & drop or browse for PNG/JPG (max 10MB)
- **Sample Dataset**: Select from COCO 2014 samples (if available)
- **URL**: Paste an image URL from the web

### 4. Try Captioning

1. Go to **💬 Captioning** tab
2. Select a model: ResNet50+LSTM or InceptionV3+Transformer
3. Adjust beam search width (higher = more diverse captions)
4. Click **🎯 Generate Caption**
5. View the generated caption and token probabilities

### 5. Try Segmentation

1. Go to **🎨 Segmentation** tab
2. Choose a model: Mask R-CNN (instance) or DeepLabV3+ (semantic)
3. Adjust confidence threshold
4. Click **🎯 Segment Image**
5. View detected objects with colored masks

### 6. Export Results

- **Caption**: Download as .txt file
- **Mask**: Download segmentation as .png
- **JSON**: Export complete results
- **ZIP**: Bundle everything together

## 🔧 Configuration

### GPU vs CPU

The app automatically detects available hardware:
- **GPU Available**: Displays GPU name and uses CUDA
- **CPU Only**: Falls back to CPU mode

Switch manually in sidebar: **Settings → Computation Device**

### Performance Tips

**For faster processing**:
- Use GPU if available
- Close other applications to free RAM
- Use smaller images (resize to 640x480)
- Lower beam search width
- Use CPU for testing, GPU for production

**For better results**:
- Use higher quality images
- Increase beam search width
- Try different models
- Compare outputs from multiple models

## 🧪 Testing the Installation

Run the test suite to verify everything works:

```powershell
# Activate virtual environment first
.\venv\Scripts\Activate.ps1

# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=. --cov-report=html
```

View coverage report: Open `htmlcov/index.html` in browser

## 🐳 Using Docker

### CPU Version
```powershell
docker-compose up app
```
Access at: http://localhost:8501

### GPU Version (Requires NVIDIA Docker)
```powershell
docker-compose --profile gpu up app-gpu
```

## 📁 Project Structure at a Glance

```
Project1/
├── app.py                  ← Main application (START HERE)
├── requirements.txt        ← Dependencies
├── models_manifest.json    ← Model configurations
├── README.md              ← Full documentation
├── DEPLOYMENT.md          ← Deployment guide
├── setup.ps1              ← Automated setup script
│
├── models/                ← Model wrappers
│   ├── wrappers.py
│   └── checkpoints/       ← Model weights (download separately)
│
├── inference/             ← Inference pipelines
│   ├── captioning.py
│   └── segmentation.py
│
├── utils/                 ← Utility functions
│   ├── viz.py            ← Visualization
│   ├── coco_utils.py     ← COCO dataset helpers
│   └── io.py             ← Export functions
│
├── static/                ← Static assets
│   ├── custom.css
│   └── samples/          ← Sample images (add your own)
│
└── tests/                 ← Test suite
    ├── test_viz.py
    └── test_inference.py
```

## 🎯 Common Use Cases

### Use Case 1: Generate Caption for Your Photo
1. Upload your image (File Upload)
2. Go to Captioning tab
3. Select ResNet50+LSTM
4. Generate caption
5. Download caption.txt

### Use Case 2: Find Objects in Image
1. Upload image
2. Go to Segmentation tab
3. Select Mask R-CNN
4. View detected objects with bounding boxes
5. Download mask.png

### Use Case 3: Process Multiple Images
1. Go to Batch Processing tab
2. Upload multiple images
3. Select "Both" (captioning + segmentation)
4. Download results.zip

### Use Case 4: Compare Models
1. Upload image
2. Try ResNet50+LSTM → note caption
3. Try InceptionV3+Transformer → compare
4. Export both results

## 🆘 Troubleshooting

### App won't start
- Check Python version: `python --version` (needs 3.10+)
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`
- Clear cache: Delete `.streamlit/` folder

### Out of memory error
- Reduce image size
- Close other applications
- Use CPU mode instead of GPU
- Process images one at a time

### Model not loading
- Check `models/checkpoints/` exists
- Download model weights (see models/checkpoints/README.md)
- Check internet connection (for auto-download)

### Slow performance
- Use GPU if available
- Reduce beam search width
- Lower confidence threshold
- Use smaller images

## 📚 Next Steps

1. **Read Full Documentation**: See [README.md](README.md)
2. **Explore Deployment Options**: See [DEPLOYMENT.md](DEPLOYMENT.md)
3. **Add Sample Images**: Place COCO images in `static/samples/`
4. **Download Models**: Get checkpoints for `models/checkpoints/`
5. **Customize**: Edit `static/custom.css` for your theme
6. **Contribute**: Add new models or features!

## 🤝 Getting Help

- **Documentation**: README.md (comprehensive guide)
- **Deployment**: DEPLOYMENT.md (hosting instructions)
- **Code Issues**: Check GitHub Issues
- **Model Questions**: See models_manifest.json comments

## 🎓 Learning Resources

- **COCO Dataset**: https://cocodataset.org/
- **Streamlit Docs**: https://docs.streamlit.io/
- **PyTorch Tutorials**: https://pytorch.org/tutorials/
- **Object Detection**: https://github.com/pytorch/vision
- **Image Captioning**: https://huggingface.co/tasks/image-to-text

---

## ✅ Quick Checklist

Before you start, make sure you have:

- [ ] Python 3.10+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] NLTK data downloaded
- [ ] App running (`streamlit run app.py`)
- [ ] Browser open at http://localhost:8501
- [ ] Test image ready to upload

**You're all set!** 🎉 Enjoy exploring image captioning and segmentation!

---

**Need help?** Open an issue on GitHub or check the [full README](README.md).
