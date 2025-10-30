# Changelog

All notable changes to the Image Captioning & Segmentation project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned Features
- [ ] Additional captioning models (BLIP, CLIP)
- [ ] Video captioning support
- [ ] Real-time webcam input
- [ ] Multi-language captions
- [ ] Model fine-tuning interface
- [ ] Advanced metrics dashboard
- [ ] Cloud storage integration

## [1.0.0] - 2024-01-XX (Initial Release)

### Added

#### Core Features
- **Streamlit Web Application**: Production-ready web interface
- **Image Captioning Module**: 
  - ResNet50 + LSTM model
  - InceptionV3 + Transformer model
  - Beam search implementation (width 1-5)
  - Temperature sampling
  - Token probability visualization
  - BLEU and CIDEr metrics

- **Image Segmentation Module**:
  - Mask R-CNN for instance segmentation
  - DeepLabV3+ for semantic segmentation
  - U-Net for semantic segmentation
  - Configurable confidence thresholds
  - Mask transparency control
  - Bounding boxes and labels

- **Combined Pipeline**:
  - Synchronized captioning and segmentation
  - Object highlighting in captions
  - Unified result visualization

- **Batch Processing**:
  - Multiple image upload
  - Captioning, segmentation, or both
  - Progress tracking
  - Bulk export to ZIP

- **Developer Mode**:
  - Performance benchmarking
  - Memory usage monitoring
  - GPU/CPU metrics
  - Intermediate output visualization
  - Debug logging

#### User Interface
- Landing/About page with project information
- Image upload via file, URL, or sample dataset
- Responsive design with mobile support
- Custom CSS theming
- Accessibility features (ARIA labels, keyboard navigation)
- Export functionality (TXT, PNG, JSON, ZIP)

#### Backend Components
- **Model Wrappers**: Cached model loading with Streamlit
- **Inference Pipelines**: Optimized processing for both tasks
- **Visualization Utilities**: Mask overlays, legends, comparisons
- **COCO Utilities**: Dataset handling and evaluation
- **I/O Utilities**: Export and file handling

#### Infrastructure
- **Docker Support**: 
  - CPU Dockerfile with Python 3.10-slim
  - GPU Dockerfile with CUDA 11.7 and cuDNN 8
  - Docker Compose orchestration
  - Health checks and volume mounts

- **Testing**:
  - pytest test suite
  - Unit tests for all modules
  - Integration tests for pipelines
  - Code coverage reporting

- **CI/CD**:
  - GitHub Actions workflow
  - Automated testing on push/PR
  - Docker image building
  - Linting (flake8, black, isort)
  - Codecov integration

#### Documentation
- Comprehensive README.md
- Quick start guide (GETTING_STARTED.md)
- Deployment guide (DEPLOYMENT.md)
- Contributing guidelines (CONTRIBUTING.md)
- Code of conduct
- MIT License

#### Configuration
- requirements.txt with all dependencies
- models_manifest.json for model configs
- .gitignore for version control
- Setup scripts (setup.ps1 for Windows)

### Technical Stack
- Python 3.10+
- Streamlit 1.30.0+
- PyTorch 2.0.0+
- Transformers 4.30.0+
- OpenCV 4.8.0+
- NLTK 3.8.0+
- pytest 7.4.0+

### Performance
- Model caching with `@st.cache_resource`
- GPU acceleration support
- Efficient batch processing
- Optimized image preprocessing

### Accessibility
- WCAG-compliant color contrast
- Keyboard navigation support
- Screen reader friendly
- Focus indicators
- Semantic HTML

### Known Issues
- Sample images need to be added to `static/samples/`
- Model checkpoints need to be downloaded separately
- Some linting warnings remain (non-critical)

### Breaking Changes
- N/A (initial release)

---

## Version History

### [1.0.0] - Initial Release
- First production-ready version
- Complete feature set as per project requirements
- Full documentation and deployment guides

---

## Upgrade Guide

### From Development to 1.0.0
1. Run `pip install -r requirements.txt --upgrade`
2. Download model checkpoints to `models/checkpoints/`
3. Add sample images to `static/samples/`
4. Update environment variables if needed
5. Run tests: `pytest tests/ -v`
6. Restart the application

---

## Contributors

### Core Team
- **Brunda B** - Image Captioning & Segmentation Implementation
- **Jaromi D** - Image Captioning & Segmentation Implementation

### Supervisor
- **Chandan Mishra** - Project Supervisor

---

## References

- [COCO Dataset](https://cocodataset.org/)
- [PyTorch Vision](https://github.com/pytorch/vision)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## Notes

- Versions follow [Semantic Versioning](https://semver.org/)
- Dates are in YYYY-MM-DD format
- All notable changes are documented here
- See GitHub releases for binary downloads

---

**Legend**:
- `Added` - New features
- `Changed` - Changes in existing functionality
- `Deprecated` - Soon-to-be removed features
- `Removed` - Removed features
- `Fixed` - Bug fixes
- `Security` - Security fixes
