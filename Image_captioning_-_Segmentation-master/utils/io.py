"""
I/O utilities for exports and file handling
"""
import json
import zipfile
from pathlib import Path
from typing import Dict, List
from PIL import Image
import numpy as np
import io


def export_caption_txt(caption: str, output_path: str):
    """
    Export caption to text file
    
    Args:
        caption: Caption text
        output_path: Output file path
    """
    with open(output_path, 'w') as f:
        f.write(caption)


def export_mask_png(mask_image: Image.Image, output_path: str):
    """
    Export mask overlay as PNG
    
    Args:
        mask_image: PIL Image with mask overlay
        output_path: Output file path
    """
    mask_image.save(output_path, 'PNG')


def export_results_json(results: Dict, output_path: str):
    """
    Export results as JSON
    
    Args:
        results: Results dictionary
        output_path: Output file path
    """
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(item) for item in obj]
        return obj
    
    serializable_results = convert_numpy(results)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)


def create_export_bundle(caption: str, mask_image: Image.Image, 
                         results: Dict, output_dir: Path, 
                         image_name: str = 'result') -> str:
    """
    Create a ZIP bundle with all exports
    
    Args:
        caption: Caption text
        mask_image: PIL Image with mask overlay
        results: Results dictionary
        output_dir: Output directory path
        image_name: Base name for output files
        
    Returns:
        Path to created ZIP file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary files
    txt_path = output_dir / f"{image_name}_caption.txt"
    png_path = output_dir / f"{image_name}_mask.png"
    json_path = output_dir / f"{image_name}_results.json"
    
    export_caption_txt(caption, str(txt_path))
    export_mask_png(mask_image, str(png_path))
    export_results_json(results, str(json_path))
    
    # Create ZIP
    zip_path = output_dir / f"{image_name}_bundle.zip"
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(txt_path, txt_path.name)
        zipf.write(png_path, png_path.name)
        zipf.write(json_path, json_path.name)
    
    return str(zip_path)


def create_batch_export(results_list: List[Dict], output_dir: Path) -> str:
    """
    Create batch export with multiple results
    
    Args:
        results_list: List of result dictionaries
        output_dir: Output directory path
        
    Returns:
        Path to created ZIP file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = output_dir / "batch_results.zip"
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for idx, result in enumerate(results_list):
            # Export each result
            prefix = f"image_{idx:03d}"
            
            if 'caption' in result:
                txt_content = result['caption']
                zipf.writestr(f"{prefix}_caption.txt", txt_content)
            
            if 'results_dict' in result:
                json_content = json.dumps(result['results_dict'], indent=2)
                zipf.writestr(f"{prefix}_results.json", json_content)
    
    return str(zip_path)


def image_to_bytes(image: Image.Image, format='PNG') -> bytes:
    """
    Convert PIL Image to bytes
    
    Args:
        image: PIL Image
        format: Image format (PNG, JPEG, etc.)
        
    Returns:
        Image as bytes
    """
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=format)
    img_byte_arr.seek(0)
    return img_byte_arr.getvalue()


def load_image_from_url(url: str) -> Image.Image:
    """
    Load image from URL
    
    Args:
        url: Image URL
        
    Returns:
        PIL Image
    """
    import requests
    
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    
    return Image.open(io.BytesIO(response.content))
