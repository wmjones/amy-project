#!/usr/bin/env python3
"""
Analyze image quality for OCR performance
"""

from PIL import Image, ImageStat
import numpy as np
import os


def analyze_image(image_path):
    """Analyze image properties and quality metrics"""
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    # Open the image
    img = Image.open(image_path)
    
    print(f"Analyzing image: {image_path}")
    print("=" * 50)
    
    # Basic properties
    print(f"Format: {img.format}")
    print(f"Mode: {img.mode}")
    print(f"Size: {img.size[0]} x {img.size[1]} pixels")
    print(f"Resolution (DPI): {img.info.get('dpi', 'Not specified')}")
    
    # Calculate megapixels
    megapixels = (img.size[0] * img.size[1]) / 1000000
    print(f"Megapixels: {megapixels:.2f}")
    
    # Image statistics
    stat = ImageStat.Stat(img)
    
    # Brightness (mean pixel values)
    if img.mode == 'RGB':
        r_mean, g_mean, b_mean = stat.mean
        brightness = (r_mean + g_mean + b_mean) / 3
        print(f"\nBrightness (RGB mean): {brightness:.2f}")
        print(f"  Red channel mean: {r_mean:.2f}")
        print(f"  Green channel mean: {g_mean:.2f}")
        print(f"  Blue channel mean: {b_mean:.2f}")
    else:
        brightness = stat.mean[0]
        print(f"\nBrightness (mean): {brightness:.2f}")
    
    # Contrast (standard deviation)
    if img.mode == 'RGB':
        r_std, g_std, b_std = stat.stddev
        contrast = (r_std + g_std + b_std) / 3
        print(f"\nContrast (RGB stddev): {contrast:.2f}")
        print(f"  Red channel stddev: {r_std:.2f}")
        print(f"  Green channel stddev: {g_std:.2f}")
        print(f"  Blue channel stddev: {b_std:.2f}")
    else:
        contrast = stat.stddev[0]
        print(f"\nContrast (stddev): {contrast:.2f}")
    
    # Check for common OCR issues
    print("\nOCR Quality Analysis:")
    print("-" * 20)
    
    # Resolution check
    if img.size[0] < 1000 or img.size[1] < 1000:
        print("⚠️  Low resolution: OCR works best with images > 1000px")
    else:
        print("✓  Resolution: Good for OCR")
    
    # DPI check
    dpi = img.info.get('dpi')
    if dpi and (dpi[0] < 300 or dpi[1] < 300):
        print(f"⚠️  Low DPI: {dpi}. OCR works best with >= 300 DPI")
    elif dpi:
        print(f"✓  DPI: {dpi} - Good for OCR")
    else:
        print("⚠️  DPI not specified in image metadata")
    
    # Brightness check
    if brightness < 100:
        print("⚠️  Image may be too dark (brightness < 100)")
    elif brightness > 200:
        print("⚠️  Image may be too bright (brightness > 200)")
    else:
        print("✓  Brightness: Good range for OCR")
    
    # Contrast check
    if contrast < 40:
        print("⚠️  Low contrast - may affect OCR accuracy")
    else:
        print("✓  Contrast: Sufficient for OCR")
    
    # Color mode check
    if img.mode != 'L' and img.mode != '1':
        print(f"⚠️  Color mode '{img.mode}': Consider converting to grayscale for better OCR")
    else:
        print("✓  Grayscale/Binary mode: Optimal for OCR")
    
    # Check for potential issues
    print("\nAdditional Analysis:")
    print("-" * 20)
    
    # File size
    file_size = os.path.getsize(image_path) / 1024 / 1024  # MB
    print(f"File size: {file_size:.2f} MB")
    
    # Aspect ratio
    aspect_ratio = img.size[0] / img.size[1]
    print(f"Aspect ratio: {aspect_ratio:.2f}")
    
    # Quick blur detection (using variance of Laplacian)
    try:
        # Convert to grayscale if needed
        gray_img = img.convert('L')
        # Convert to numpy array
        img_array = np.array(gray_img)
        # Calculate variance of Laplacian
        laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        from scipy.ndimage import convolve
        filtered = convolve(img_array, laplacian)
        variance = np.var(filtered)
        print(f"Blur metric (Laplacian variance): {variance:.2f}")
        if variance < 100:
            print("⚠️  Image may be blurry (low Laplacian variance)")
        else:
            print("✓  Image sharpness: Acceptable")
    except ImportError:
        print("Note: Install scipy for blur detection")
    except Exception as e:
        print(f"Could not perform blur detection: {e}")
    
    # Recommendations
    print("\nRecommendations for better OCR:")
    print("-" * 30)
    
    recommendations = []
    
    if img.size[0] < 1000 or img.size[1] < 1000:
        recommendations.append("- Increase image resolution (rescan at higher DPI)")
    
    if img.mode != 'L':
        recommendations.append("- Convert to grayscale before OCR")
    
    if brightness < 100 or brightness > 200:
        recommendations.append("- Adjust brightness to 100-200 range")
    
    if contrast < 40:
        recommendations.append("- Increase contrast")
    
    if not recommendations:
        print("Image appears suitable for OCR")
    else:
        for rec in recommendations:
            print(rec)


if __name__ == "__main__":
    image_path = "/workspaces/amy-project/workspace/downloads/hansman/100_4247.JPG"
    analyze_image(image_path)