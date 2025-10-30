"""
Continuation of app.py - Additional rendering functions
"""
import streamlit as st
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import sys
import traceback

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import project modules
from models.wrappers import load_segmentation_model, load_captioning_model
from inference.segmentation import segment_image, SegmentationPipeline
from inference.captioning import caption_image, CaptioningPipeline
from utils.viz import overlay_instance_masks, overlay_semantic_mask, create_comparison_grid, create_mask_legend, highlight_caption_objects
from utils.io import create_export_bundle, create_batch_export, export_mask_png, export_caption_txt

# Constants for batch processing modes
CAPTION_MODE = "Captioning Only"
SEG_MODE = "Segmentation Only"
BOTH_MODE = "Both"

def render_segmentation_tab():
    """Render Segmentation module"""
    st.markdown("## 🎨 Image Segmentation")
    
    if st.session_state.current_image is None:
        st.warning("Please upload an image first!")
        return
    
    # Model selection
    col1, col2 = st.columns([1, 1])
    
    with col1:
        model_choice = st.selectbox(
            "Select Segmentation Model:",
            ["maskrcnn", "deeplabv3plus", "unet"],
            format_func=lambda x: {
                "maskrcnn": "Mask R-CNN (Instance)",
                "deeplabv3plus": "DeepLabV3+ (Semantic)",
                "unet": "U-Net (Semantic)"
            }.get(x, x)
        )
    
    with col2:
        if model_choice == "maskrcnn":
            threshold = st.slider("Confidence Threshold:", 0.1, 0.9, 0.5, 0.05)
        else:
            threshold = 0.5
    
    # Visualization settings
    with st.expander("Visualization Settings"):
        alpha = st.slider("Mask Transparency:", 0.0, 1.0, 0.5, 0.05)
        show_boxes = st.checkbox("Show Bounding Boxes", value=True)
        show_labels = st.checkbox("Show Labels", value=True)
    
    # Perform segmentation
    if st.button("🎯 Perform Segmentation", type="primary"):
        with st.spinner("Performing segmentation..."):
            try:
                # Load model
                model_wrapper = load_segmentation_model(model_choice, st.session_state.device)
                
                # Progress bar
                progress_bar = st.progress(0)
                progress_bar.progress(30)
                
                # Perform segmentation
                seg_result = segment_image(
                    model_wrapper,
                    st.session_state.current_image,
                    threshold=threshold,
                    device=st.session_state.device
                )
                
                progress_bar.progress(70)
                
                # Create visualization
                if model_wrapper.get_model_type() == 'instance':
                    overlay = overlay_instance_masks(
                        st.session_state.current_image,
                        seg_result['masks'],
                        seg_result['labels'],
                        seg_result['scores'],
                        seg_result['boxes'],
                        seg_result['class_names'],
                        alpha=alpha,
                        show_boxes=show_boxes,
                        show_labels=show_labels
                    )
                else:
                    overlay = overlay_semantic_mask(
                        st.session_state.current_image,
                        seg_result['seg_map'],
                        seg_result['classes'],
                        alpha=alpha
                    )
                
                progress_bar.progress(100)
                
                # Store result
                st.session_state.segmentation_result = {
                    'seg_result': seg_result,
                    'overlay': overlay,
                    'model': model_choice
                }
                
                st.success("Segmentation completed successfully!")
                
            except Exception as e:
                st.error(f"Error performing segmentation: {str(e)}")
                st.error("**Debug Info:**")
                st.code(traceback.format_exc())
                return
    
    # Display results
    if st.session_state.segmentation_result:
        result = st.session_state.segmentation_result
        
        # Display overlay
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Original Image")
            st.image(st.session_state.current_image)
        
        with col2:
            st.markdown("### Segmentation Overlay")
            st.image(result['overlay'])
        
        # Legend
        st.markdown("### 🎨 Class Legend")
        if result['model'] == 'maskrcnn':
            class_info = [(int(label), name) for label, name in 
                         zip(result['seg_result']['labels'], result['seg_result']['class_names'])]
        else:
            class_info = result['seg_result']['classes']
        
        legend_img = create_mask_legend(class_info)
        st.image(legend_img)
        
        # Detection details
        if result['model'] == 'maskrcnn':
            st.markdown("### 📋 Detection Details")
            
            for i, (label, score, box) in enumerate(zip(
                result['seg_result']['class_names'],
                result['seg_result']['scores'],
                result['seg_result']['boxes']
            )):
                with st.expander(f"{i+1}. {label} ({score:.2%})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Confidence:** {score:.4f}")
                        st.write(f"**Bounding Box:** [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
                    with col2:
                        st.write(f"**Width:** {box[2] - box[0]:.1f}px")
                        st.write(f"**Height:** {box[3] - box[1]:.1f}px")
        
        # Raw mask array
        if st.checkbox("Show Raw Mask Arrays"):
            if result['model'] == 'maskrcnn':
                st.markdown("**Mask Shapes:**")
                for i, mask in enumerate(result['seg_result']['masks']):
                    st.text(f"  Mask {i+1}: {mask.shape}")
                st.markdown(f"**Total Detections:** {len(result['seg_result']['masks'])}")
            else:
                st.markdown(f"**Segmentation Map Shape:** {result['seg_result']['seg_map'].shape}")
                st.markdown(f"**Unique Classes:** {len(result['seg_result']['classes'])}")


def render_combined_tab():
    """Render Combined Pipeline"""
    st.markdown("## 🔗 Combined Pipeline")
    st.markdown("Run both captioning and segmentation, with caption-to-object linking")
    
    if st.session_state.current_image is None:
        st.warning("Please upload an image first!")
        return
    
    # Model selections
    col1, col2 = st.columns(2)
    
    with col1:
        caption_model = st.selectbox(
            "Captioning Model:",
            ["resnet50_lstm", "inceptionv3_transformer"],
            key="combined_caption_model"
        )
    
    with col2:
        seg_model = st.selectbox(
            "Segmentation Model:",
            ["maskrcnn", "deeplabv3plus", "unet"],
            key="combined_seg_model"
        )
    
    if st.button("🚀 Run Combined Pipeline", type="primary"):
        with st.spinner("Running combined pipeline..."):
            try:
                # Run captioning
                st.info("Step 1/2: Generating caption...")
                caption_wrapper = load_captioning_model(caption_model, st.session_state.device)
                caption, tokens, _ = caption_image(
                    caption_wrapper,
                    st.session_state.current_image,
                    device=st.session_state.device
                )
                
                # Run segmentation
                st.info("Step 2/2: Performing segmentation...")
                seg_wrapper = load_segmentation_model(seg_model, st.session_state.device)
                seg_result = segment_image(
                    seg_wrapper,
                    st.session_state.current_image,
                    device=st.session_state.device
                )
                
                # Create visualizations
                if seg_wrapper.get_model_type() == 'instance':
                    overlay = overlay_instance_masks(
                        st.session_state.current_image,
                        seg_result['masks'],
                        seg_result['labels'],
                        seg_result['scores'],
                        seg_result['boxes'],
                        seg_result['class_names'],
                        alpha=0.5
                    )
                    
                    # Highlight objects mentioned in caption
                    highlighted = highlight_caption_objects(
                        st.session_state.current_image,
                        caption,
                        seg_result['class_names'],
                        seg_result['boxes'],
                        seg_result['class_names']
                    )
                else:
                    overlay = overlay_semantic_mask(
                        st.session_state.current_image,
                        seg_result['seg_map'],
                        seg_result['classes'],
                        alpha=0.5
                    )
                    highlighted = overlay
                
                # Display results
                st.success("Combined pipeline completed!")
                
                # Caption
                st.markdown("### 💬 Generated Caption")
                st.markdown(f"#### *\"{caption}\"*")
                
                # Visualizations
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Original**")
                    st.image(st.session_state.current_image)
                
                with col2:
                    st.markdown("**Segmentation**")
                    st.image(overlay)
                
                with col3:
                    st.markdown("**Caption-Linked**")
                    st.image(highlighted)
                
                # Export section
                st.markdown("### 📦 Export Results")
                
                if st.button("Download Results Bundle"):
                    # Create exports
                    output_dir = Path("outputs")
                    bundle_path = create_export_bundle(
                        caption,
                        overlay,
                        {
                            'caption': caption,
                            'tokens': tokens,
                            'segmentation': {
                                'model': seg_model,
                                'num_detections': len(seg_result.get('masks', [])) if 'masks' in seg_result else 0
                            }
                        },
                        output_dir
                    )
                    
                    with open(bundle_path, 'rb') as f:
                        st.download_button(
                            "📥 Download ZIP",
                            f.read(),
                            file_name="results_bundle.zip",
                            mime="application/zip"
                        )
                
            except Exception as e:
                st.error(f"Error in combined pipeline: {str(e)}")
                st.error("**Debug Info:**")
                st.code(traceback.format_exc())


def render_batch_tab():
    """Render Batch Processing mode"""
    st.markdown("## 📚 Batch Processing")
    st.markdown("Process multiple images at once")
    
    st.info("Batch processing allows you to upload multiple images and process them sequentially.")
    
    # File uploader for multiple files
    uploaded_files = st.file_uploader(
        "Upload multiple images:",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.write(f"**{len(uploaded_files)} images uploaded**")
        
        # Model selection
        CAPTION_MODE = "Captioning Only"
        SEG_MODE = "Segmentation Only"
        BOTH_MODE = "Both"
        
        col1, col2 = st.columns(2)
        with col1:
            batch_mode = st.radio("Processing Mode:", [CAPTION_MODE, SEG_MODE, BOTH_MODE])
        with col2:
            caption_model = None
            seg_model = None
            if batch_mode in [CAPTION_MODE, BOTH_MODE]:
                caption_model = st.selectbox("Caption Model:", ["resnet50_lstm", "inceptionv3_transformer"])
            if batch_mode in [SEG_MODE, BOTH_MODE]:
                seg_model = st.selectbox("Segmentation Model:", ["maskrcnn", "deeplabv3plus", "unet"])
        
        if st.button("🚀 Start Batch Processing", type="primary"):
            results_list = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {idx+1}/{len(uploaded_files)}: {uploaded_file.name}")
                
                try:
                    image = Image.open(uploaded_file)
                    result = {'filename': uploaded_file.name}
                    
                    # Process based on mode
                    if batch_mode in [CAPTION_MODE, BOTH_MODE]:
                        if caption_model:
                            wrapper = load_captioning_model(caption_model, st.session_state.device)
                            caption, _, _ = caption_image(wrapper, image, device=st.session_state.device)
                            result['caption'] = caption
                    
                    if batch_mode in [SEG_MODE, BOTH_MODE]:
                        if seg_model:
                            wrapper = load_segmentation_model(seg_model, st.session_state.device)
                            seg_result = segment_image(wrapper, image, device=st.session_state.device)
                            result['segmentation'] = {
                                'num_objects': len(seg_result.get('masks', [])) if 'masks' in seg_result else 0
                            }
                    
                    results_list.append(result)
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.text("✅ Batch processing complete!")
            
            # Display results table
            st.markdown("### Results Summary")
            st.table(results_list)
            
            # Export
            if st.button("Download Batch Results"):
                from utils.io import create_batch_export
                output_dir = Path("outputs")
                zip_path = create_batch_export(results_list, output_dir)
                
                with open(zip_path, 'rb') as f:
                    st.download_button(
                        "📥 Download Results",
                        f.read(),
                        file_name="batch_results.zip",
                        mime="application/zip"
                    )


# (Additional functions: render_developer_tab, main would be in final app.py)
