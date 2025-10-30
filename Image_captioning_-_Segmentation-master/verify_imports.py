"""
Quick verification script to check if all imports work correctly
Run this before starting the Streamlit app to verify no import errors
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("🔍 Verifying imports...")

try:
    print("  ✓ Importing app.py modules...")
    from models.wrappers import load_captioning_model, load_segmentation_model
    from inference.captioning import caption_image, CaptioningPipeline
    from inference.segmentation import segment_image, SegmentationPipeline
    from utils.viz import (overlay_instance_masks, overlay_semantic_mask, create_mask_legend,
                           highlight_caption_objects, create_comparison_grid)
    from utils.coco_utils import create_sample_manifest, get_image_captions
    from utils.io import (export_caption_txt, export_mask_png, export_results_json,
                         create_export_bundle, image_to_bytes, load_image_from_url)
    print("    ✅ app.py imports successful")
    
    print("  ✓ Importing app_continuation.py modules...")
    from models.wrappers import load_segmentation_model, load_captioning_model
    from inference.segmentation import segment_image, SegmentationPipeline
    from inference.captioning import caption_image, CaptioningPipeline
    from utils.viz import overlay_instance_masks, overlay_semantic_mask, create_comparison_grid, create_mask_legend, highlight_caption_objects
    from utils.io import create_export_bundle, create_batch_export, export_mask_png, export_caption_txt
    print("    ✅ app_continuation.py imports successful")
    
    print("\n✅ All imports verified successfully!")
    print("\n🚀 You can now run the app with:")
    print("   streamlit run app.py")
    
except ImportError as e:
    print(f"\n❌ Import error detected: {e}")
    print("\n📋 Please check:")
    print("  1. All required packages are installed (requirements.txt)")
    print("  2. All module files exist in the correct directories")
    print("  3. __init__.py files exist in all package directories")
    sys.exit(1)
    
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
