# Deployment Guide

## Quick Start (One-Click Deploy)

### Local Development

```bash
# 1. Clone and setup
git clone <your-repo-url>
cd image-caption-seg-streamlit
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"

# 3. Run the app
streamlit run app.py
```

Access at: `http://localhost:8501`

### Docker Deployment

**CPU Version:**
```bash
docker-compose up app
```

**GPU Version:**
```bash
docker-compose --profile gpu up app-gpu
```

## Streamlit Cloud Deployment

1. **Push to GitHub**:
```bash
git init
git add .
git commit -m "Initial commit: Production-ready image caption & segmentation app"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository
   - Set main file: `app.py`
   - Set Python version: `3.10`
   - Click "Deploy"

3. **Configure Secrets** (if needed):
   - Go to app settings
   - Add secrets in TOML format:
   ```toml
   [secrets]
   HUGGINGFACE_TOKEN = "your_token_here"
   ```

## Docker Hub Deployment

```bash
# Build images
docker build -t yourusername/image-caption-seg:latest -f Dockerfile .
docker build -t yourusername/image-caption-seg:gpu -f Dockerfile.gpu .

# Push to Docker Hub
docker login
docker push yourusername/image-caption-seg:latest
docker push yourusername/image-caption-seg:gpu

# Run from Docker Hub
docker run -p 8501:8501 yourusername/image-caption-seg:latest
```

## Cloud Platform Deployment

### AWS ECS

1. **Create ECR repository**:
```bash
aws ecr create-repository --repository-name image-caption-seg
```

2. **Push image to ECR**:
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag image-caption-seg:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/image-caption-seg:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/image-caption-seg:latest
```

3. **Create ECS task definition** and service

### Google Cloud Run

```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT-ID/image-caption-seg

# Deploy
gcloud run deploy image-caption-seg \
  --image gcr.io/PROJECT-ID/image-caption-seg \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2
```

### Azure Container Instances

```bash
# Create resource group
az group create --name ImageCaptionRG --location eastus

# Create container registry
az acr create --resource-group ImageCaptionRG --name imagecaptionacr --sku Basic

# Build and push
az acr build --registry imagecaptionacr --image image-caption-seg:latest .

# Deploy
az container create \
  --resource-group ImageCaptionRG \
  --name image-caption-seg-app \
  --image imagecaptionacr.azurecr.io/image-caption-seg:latest \
  --cpu 2 \
  --memory 4 \
  --registry-login-server imagecaptionacr.azurecr.io \
  --registry-username <username> \
  --registry-password <password> \
  --dns-name-label image-caption-seg \
  --ports 8501
```

## Production Checklist

- [ ] Update GitHub repository URLs in README.md
- [ ] Add actual sample images to `static/samples/`
- [ ] Download model checkpoints to `models/checkpoints/`
- [ ] Configure environment variables for production
- [ ] Set up monitoring and logging
- [ ] Configure SSL/TLS for HTTPS
- [ ] Set up CI/CD pipeline (already configured in `.github/workflows/ci.yml`)
- [ ] Run security audit: `pip install safety && safety check`
- [ ] Run dependency audit: `pip-audit`
- [ ] Test with production data
- [ ] Set up error tracking (e.g., Sentry)
- [ ] Configure autoscaling (if applicable)

## Environment Variables

Create `.env` file for local development:

```env
# Optional: API keys for cloud storage
CLOUD_STORAGE_BUCKET=your-bucket-name
HUGGINGFACE_TOKEN=your-token

# Optional: Model checkpoint URLs
RESNET50_LSTM_CHECKPOINT_URL=https://...
INCEPTIONV3_TRANSFORMER_CHECKPOINT_URL=https://...

# Optional: Performance tuning
STREAMLIT_SERVER_MAX_UPLOAD_SIZE=10
STREAMLIT_SERVER_ENABLE_CORS=false
```

## Performance Optimization

### For Production Deployment:

1. **Enable caching**: Already implemented with `@st.cache_resource`
2. **Use CDN**: Host static assets on CDN
3. **Optimize images**: Compress images before upload
4. **Use model quantization**: For faster inference
5. **Enable GPU**: Use GPU-enabled instances for faster processing

### Recommended Instance Sizes:

- **CPU-only**: 4 vCPU, 8GB RAM minimum
- **GPU**: 1x NVIDIA T4 or better, 16GB RAM minimum

## Monitoring

### Health Check Endpoint

The Docker containers include health checks:
```bash
curl http://localhost:8501/_stcore/health
```

### Metrics to Monitor:

- Response time
- Memory usage
- GPU utilization (if applicable)
- Error rate
- Request count

### Logging

View Streamlit logs:
```bash
# Docker
docker logs -f <container-id>

# Cloud Run
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=image-caption-seg" --limit 50
```

## Troubleshooting

### Common Issues:

1. **Out of Memory**:
   - Reduce batch size
   - Clear GPU cache regularly
   - Use CPU mode for testing

2. **Slow Loading**:
   - Preload models at startup
   - Use smaller model variants
   - Enable caching

3. **Port Already in Use**:
   ```bash
   streamlit run app.py --server.port 8502
   ```

4. **CUDA Out of Memory**:
   ```python
   torch.cuda.empty_cache()
   ```

## Support

For issues and questions:
- Open an issue on GitHub
- Check the README.md
- Review existing issues

---

**Deployment Status**: ✅ Ready for Production
