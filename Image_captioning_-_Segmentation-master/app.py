"""
Main Streamlit Application for Image Captioning & Segmentation
Production-ready web interface for COCO-based deep learning models
"""

import streamlit as st
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import sys
import time
from io import BytesIO

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import project modules
from models.wrappers import load_captioning_model, load_segmentation_model
from inference.captioning import caption_image, CaptioningPipeline
from inference.segmentation import segment_image, SegmentationPipeline, masks_to_rle
from utils.viz import (overlay_instance_masks, overlay_semantic_mask, create_mask_legend,
                       highlight_caption_objects, create_comparison_grid)
from utils.coco_utils import create_sample_manifest, get_image_captions
from utils.io import (export_caption_txt, export_mask_png, export_results_json,
                     create_export_bundle, image_to_bytes, load_image_from_url)

# Page navigation constants
PAGE_LANDING = "🏠 Landing / About"
PAGE_CAPTIONING = "💬 Captioning"
PAGE_SEGMENTATION = "🎨 Segmentation"
PAGE_COMBINED = "🔗 Combined Pipeline"
PAGE_BATCH = "📦 Batch Processing"
PAGE_DEVELOPER = "🛠️ Developer Mode"

# Page configuration
st.set_page_config(
    page_title="Image Captioning & Segmentation",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    css_file = project_root / "static" / "custom.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Initialize session state
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'captioning_result' not in st.session_state:
    st.session_state.captioning_result = None
if 'segmentation_result' not in st.session_state:
    st.session_state.segmentation_result = None
if 'device' not in st.session_state:
    st.session_state.device = 'cuda' if torch.cuda.is_available() else 'cpu'


def render_header():
    """Render application header"""
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        st.markdown("### 🖼️")
    
    with col2:
        st.markdown("# Image Captioning & Segmentation")
        st.markdown("*Deep Learning on COCO 2014 Dataset*")
    
    with col3:
        st.markdown("")


def render_landing_page():
    """Render Landing/About page"""
    st.markdown("## About This Project")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Project Abstract
        
        The "Image Captioning and Segmentation" project uses the MS COCO 2014 dataset to establish 
        sophisticated models that combine image captioning and segmentation, employing deep learning 
        models such as CNNs, LSTMs, Transformers, and Mask R-CNN for accurate object labeling and 
        text description generation.
        
        ### Objectives
        
        - 🔍 Investigate theoretical and practical foundations of image captioning and segmentation
        - 🤖 Implement deep learning architectures for generating text descriptions
        - 🎨 Perform semantic and instance-level segmentation using state-of-the-art neural networks
        - 🔗 Integrate both systems into a unified pipeline
        - 🚀 Deploy the final model using an interactive UI
        
        ### Dataset: COCO 2014
        
        The COCO (Common Objects in Context) 2014 dataset is a gold standard benchmark featuring:
        - **330K+ images** with 200K+ labeled
        - **5 human-written captions** per image
        - Segmentation masks for object boundaries
        - **80 object categories** and 91 stuff categories
        
        [Learn more about COCO Dataset →](https://cocodataset.org/)
        """)
    
    with col2:
        st.info("""
        **Tech Stack**
        
        - Python 3.10+
        - PyTorch
        - Streamlit
        - OpenCV
        - Transformers
        - NLTK
        """)
        
        st.success("""
        **Models**
        
        **Captioning:**
        - ResNet50 + LSTM
        - InceptionV3 + Transformer
        
        **Segmentation:**
        - U-Net (Semantic)
        - DeepLabV3+ (Semantic)
        - Mask R-CNN (Instance)
        """)
        
        st.markdown("""
        **Submitted to:** Chandan Mishra
        """)
    
    st.markdown("---")


def render_image_upload():
    """Render image upload section"""
    st.markdown("### 📤 Upload Image")
    
    upload_method = st.radio(
        "Select upload method:",
        ["File Upload", "Sample Dataset", "URL"],
        horizontal=True,
        key="upload_method"
    )
    
    uploaded_image = None
    
    if upload_method == "File Upload":
        uploaded_file = st.file_uploader(
            "Choose an image (PNG, JPG, JPEG)",
            type=['png', 'jpg', 'jpeg'],
            help="Max file size: 10MB"
        )
        
        if uploaded_file:
            # Check file size (10MB limit)
            if uploaded_file.size > 10 * 1024 * 1024:
                st.error("File size exceeds 10MB limit. Please upload a smaller image.")
                return None
            
            uploaded_image = Image.open(uploaded_file)
    
    elif upload_method == "Sample Dataset":
        samples = create_sample_manifest()
        sample_names = [f"{s['file_name']} - {s['captions'][0][:50]}..." for s in samples]
        
        selected_sample = st.selectbox("Select a sample image:", sample_names)
        
        if selected_sample:
            # Sample images would be loaded from static/samples/ directory in production
            st.info("📌 Sample images would be loaded from static/samples/ directory")
            # Future implementation:
            # sample_path = f"static/samples/{samples[0]['file_name']}"
            # if Path(sample_path).exists():
            #     uploaded_image = Image.open(sample_path)
    
    elif upload_method == "URL":
        image_url = st.text_input("Enter image URL:")
        
        if image_url:
            try:
                with st.spinner("Loading image from URL..."):
                    uploaded_image = load_image_from_url(image_url)
                st.success("Image loaded successfully!")
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
    
    if uploaded_image:
        st.session_state.current_image = uploaded_image
        
        # Display image preview
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(uploaded_image, caption="Uploaded Image")
        with col2:
            st.markdown("**Image Metadata:**")
            st.write(f"- Size: {uploaded_image.size}")
            st.write(f"- Mode: {uploaded_image.mode}")
            st.write(f"- Format: {uploaded_image.format if hasattr(uploaded_image, 'format') else 'N/A'}")
    
    return uploaded_image


def render_captioning_tab():
    """Render Captioning module"""
    st.markdown("## 💬 Image Captioning")
    
    if st.session_state.current_image is None:
        st.warning("Please upload an image first!")
        return
    
    # Model selection
    col1, col2 = st.columns([1, 1])
    
    with col1:
        model_choice = st.selectbox(
            "Select Captioning Model:",
            ["resnet50_lstm", "inceptionv3_transformer"],
            format_func=lambda x: "ResNet50 + LSTM" if "resnet50" in x else "InceptionV3 + Transformer"
        )
    
    with col2:
        beam_size = st.slider("Beam Search Width:", 1, 5, 3)
        max_length = st.slider("Max Caption Length:", 10, 30, 20)
    
    # Generation settings
    with st.expander("Advanced Settings"):
        # These settings are placeholders for future enhancements
        st.slider("Temperature (sampling randomness):", 0.1, 2.0, 1.0, 0.1, 
                 help="Reserved for future use", disabled=True)
        st.number_input("Number of caption variations:", 1, 5, 1,
                       help="Reserved for future use", disabled=True)
    
    # Generate caption
    if st.button("🎯 Generate Caption", type="primary"):
        with st.spinner("Generating caption..."):
            try:
                # Load model
                model_wrapper = load_captioning_model(model_choice, st.session_state.device)
                
                # Progress bar
                progress_bar = st.progress(0)
                progress_bar.progress(30)
                
                # Generate caption
                caption, tokens, probs = caption_image(
                    model_wrapper,
                    st.session_state.current_image,
                    beam_size=beam_size,
                    max_length=max_length,
                    device=st.session_state.device
                )
                
                progress_bar.progress(100)
                
                # Store result
                st.session_state.captioning_result = {
                    'caption': caption,
                    'tokens': tokens,
                    'probabilities': probs,
                    'model': model_choice
                }
                
                st.success("Caption generated successfully!")
                
            except Exception as e:
                st.error(f"Error generating caption: {str(e)}")
                return
    
    # Display results
    if st.session_state.captioning_result:
        result = st.session_state.captioning_result
        
        st.markdown("### 📝 Generated Caption")
        st.markdown(f"#### *\"{result['caption']}\"*")
        
        # Token analysis
        with st.expander("Token Analysis"):
            st.markdown("**Tokens and Probabilities:**")
            for token, prob in zip(result['tokens'], result['probabilities']):
                st.write(f"- **{token}**: {prob:.3f}")
        
        # Metrics (if reference captions available)
        reference_captions = st.text_area(
            "Enter reference captions (one per line) for evaluation:",
            height=100
        )
        
        if reference_captions:
            refs = [cap.strip() for cap in reference_captions.split('\n') if cap.strip()]
            if refs:
                pipeline = CaptioningPipeline(
                    load_captioning_model(model_choice, st.session_state.device),
                    st.session_state.device
                )
                metrics = pipeline.calculate_metrics(result['caption'], refs)
                
                st.markdown("### 📊 Evaluation Metrics")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("BLEU-1", f"{metrics['BLEU-1']:.3f}")
                col2.metric("BLEU-2", f"{metrics['BLEU-2']:.3f}")
                col3.metric("BLEU-3", f"{metrics['BLEU-3']:.3f}")
                col4.metric("BLEU-4", f"{metrics['BLEU-4']:.3f}")


def render_developer_tab():
    """Render Developer/Debug mode"""
    st.markdown("## 🛠️ Developer Mode")
    
    st.info("This section provides insights into model performance, memory usage, and debug outputs.")
    
    # System info
    st.markdown("### 📊 System Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Device", st.session_state.device.upper())
    with col2:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            st.metric("GPU", gpu_name[:20] + "..." if len(gpu_name) > 20 else gpu_name)
        else:
            st.metric("GPU", "N/A")
    with col3:
        st.metric("PyTorch Version", torch.__version__)
    
    # Model performance metrics
    st.markdown("### ⏱️ Model Performance")
    
    if st.button("Run Performance Benchmark", type="secondary"):
        if st.session_state.current_image is None:
            st.warning("Please upload an image first!")
            return
        
        with st.spinner("Running benchmark..."):
            results = {}
            
            # Captioning models
            st.markdown("#### Captioning Models")
            for model_name in ["resnet50_lstm", "inceptionv3_transformer"]:
                try:
                    start_time = time.time()
                    model = load_captioning_model(model_name, st.session_state.device)
                    load_time = time.time() - start_time
                    
                    start_time = time.time()
                    _ = caption_image(model, st.session_state.current_image, device=st.session_state.device)
                    inference_time = time.time() - start_time
                    
                    results[model_name] = {
                        'load_time': load_time,
                        'inference_time': inference_time
                    }
                    
                    col1, col2 = st.columns(2)
                    col1.metric(f"{model_name} - Load Time", f"{load_time:.2f}s")
                    col2.metric(f"{model_name} - Inference Time", f"{inference_time:.2f}s")
                    
                except Exception as e:
                    st.error(f"Error benchmarking {model_name}: {str(e)}")
            
            # Segmentation models
            st.markdown("#### Segmentation Models")
            for model_name in ["maskrcnn", "deeplabv3plus"]:
                try:
                    start_time = time.time()
                    model = load_segmentation_model(model_name, st.session_state.device)
                    load_time = time.time() - start_time
                    
                    start_time = time.time()
                    _ = segment_image(model, st.session_state.current_image, device=st.session_state.device)
                    inference_time = time.time() - start_time
                    
                    results[model_name] = {
                        'load_time': load_time,
                        'inference_time': inference_time
                    }
                    
                    col1, col2 = st.columns(2)
                    col1.metric(f"{model_name} - Load Time", f"{load_time:.2f}s")
                    col2.metric(f"{model_name} - Inference Time", f"{inference_time:.2f}s")
                    
                except Exception as e:
                    st.error(f"Error benchmarking {model_name}: {str(e)}")
    
    # Memory usage
    st.markdown("### 💾 Memory Usage")
    
    if torch.cuda.is_available():
        col1, col2, col3 = st.columns(3)
        with col1:
            allocated = torch.cuda.memory_allocated() / 1024**3
            st.metric("GPU Memory Allocated", f"{allocated:.2f} GB")
        with col2:
            reserved = torch.cuda.memory_reserved() / 1024**3
            st.metric("GPU Memory Reserved", f"{reserved:.2f} GB")
        with col3:
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            st.metric("Max GPU Memory", f"{max_allocated:.2f} GB")
        
        if st.button("Clear GPU Cache"):
            torch.cuda.empty_cache()
            st.success("GPU cache cleared!")
    else:
        st.info("GPU not available. Memory tracking is limited.")
    
    # Debug outputs
    st.markdown("### 🔍 Debug Outputs")
    
    debug_mode = st.checkbox("Enable verbose debugging")
    
    if debug_mode and st.session_state.current_image:
        st.markdown("#### Image Preprocessing")
        
        # Show preprocessed image
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        preprocessed = transform(st.session_state.current_image.convert('RGB'))
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state.current_image, caption="Original")
        with col2:
            # Denormalize for visualization
            denorm = preprocessed.clone()
            for t, m, s in zip(denorm, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]):
                t.mul_(s).add_(m)
            denorm = denorm.permute(1, 2, 0).numpy()
            denorm = np.clip(denorm, 0, 1)
            st.image(denorm, caption="Preprocessed (Normalized)")
        
        st.write(f"Tensor shape: {preprocessed.shape}")
        st.write(f"Min value: {preprocessed.min():.3f}, Max value: {preprocessed.max():.3f}")
    
    # Feature map visualization
    if debug_mode and st.session_state.captioning_result:
        st.markdown("#### Intermediate Outputs")
        
        with st.expander("Token Probabilities (Full Distribution)"):
            if 'probabilities' in st.session_state.captioning_result:
                probs = st.session_state.captioning_result['probabilities']
                st.bar_chart(dict(zip(st.session_state.captioning_result['tokens'], probs)))
    
    if debug_mode and st.session_state.segmentation_result:
        with st.expander("Segmentation Mask Arrays"):
            if 'masks' in st.session_state.segmentation_result:
                masks = st.session_state.segmentation_result['masks']
                st.write(f"Number of masks: {len(masks)}")
                for i, mask in enumerate(masks[:3]):  # Show first 3
                    st.write(f"Mask {i+1} shape: {mask.shape}")
                    st.write(f"Unique values: {np.unique(mask)}")
                    if st.checkbox(f"Show Mask {i+1} heatmap"):
                        st.image(mask, caption=f"Mask {i+1}", clamp=True)


def main():
    """Main application entry point"""
    render_header()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to:",
        [PAGE_LANDING, PAGE_CAPTIONING, PAGE_SEGMENTATION, 
         PAGE_COMBINED, PAGE_BATCH, PAGE_DEVELOPER],
        label_visibility="collapsed"
    )
    
    # Settings
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Settings")
    
    # Device selection
    if torch.cuda.is_available():
        device_option = st.sidebar.radio(
            "Computation Device:",
            ["GPU (CUDA)", "CPU"],
            index=0 if st.session_state.device == 'cuda' else 1
        )
        st.session_state.device = 'cuda' if 'GPU' in device_option else 'cpu'
    else:
        st.sidebar.info("GPU not available. Using CPU.")
        st.session_state.device = 'cpu'
    
    # Image upload (always available except on landing page)
    if page != PAGE_LANDING:
        st.sidebar.markdown("---")
        render_image_upload()
    
    # Render selected page
    st.markdown("---")
    
    if page == PAGE_LANDING:
        render_landing_page()
    
    elif page == PAGE_CAPTIONING:
        render_captioning_tab()
    
    elif page == PAGE_SEGMENTATION:
        # Import from continuation file
        from app_continuation import render_segmentation_tab
        render_segmentation_tab()
    
    elif page == PAGE_COMBINED:
        from app_continuation import render_combined_tab
        render_combined_tab()
    
    elif page == PAGE_BATCH:
        from app_continuation import render_batch_tab
        render_batch_tab()
    
    elif page == PAGE_DEVELOPER:
        render_developer_tab()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.8em;'>
    <p>Image Captioning & Segmentation</p>
    <p>Built with ❤️ using Streamlit</p>
    <p>© 2024 Chandra Pal D</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
