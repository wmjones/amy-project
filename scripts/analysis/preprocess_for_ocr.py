#!/usr/bin/env python3
"""
Preprocess image for better OCR results
"""

from PIL import Image, ImageEnhance, ImageOps
import os


def preprocess_image(input_path, output_dir="/workspaces/amy-project/workspace/processed/"):
    """Apply preprocessing steps to improve OCR quality"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the image
    img = Image.open(input_path)
    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)
    
    print(f"Processing: {filename}")
    print("-" * 40)
    
    # Step 1: Convert to grayscale
    gray_img = img.convert('L')
    gray_path = os.path.join(output_dir, f"{name}_1_grayscale{ext}")
    gray_img.save(gray_path)
    print(f"✓ Converted to grayscale: {gray_path}")
    
    # Step 2: Enhance contrast
    enhancer = ImageEnhance.Contrast(gray_img)
    contrast_img = enhancer.enhance(1.5)  # Increase contrast by 50%
    contrast_path = os.path.join(output_dir, f"{name}_2_contrast{ext}")
    contrast_img.save(contrast_path)
    print(f"✓ Enhanced contrast: {contrast_path}")
    
    # Step 3: Auto-adjust levels (equalize)
    equalized_img = ImageOps.equalize(contrast_img)
    equalized_path = os.path.join(output_dir, f"{name}_3_equalized{ext}")
    equalized_img.save(equalized_path)
    print(f"✓ Equalized histogram: {equalized_path}")
    
    # Step 4: Apply adaptive thresholding (simple version)
    # Convert to binary based on mean threshold
    threshold = 128
    binary_img = contrast_img.point(lambda x: 0 if x < threshold else 255, '1')
    binary_path = os.path.join(output_dir, f"{name}_4_binary{ext}")
    binary_img.save(binary_path)
    print(f"✓ Applied binary threshold: {binary_path}")
    
    # Step 5: Resize if needed (optional)
    # If DPI is too high, we might want to resize
    if img.size[0] > 3000:
        scale_factor = 3000 / img.size[0]
        new_size = (int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))
        resized_img = contrast_img.resize(new_size, Image.Resampling.LANCZOS)
        resized_path = os.path.join(output_dir, f"{name}_5_resized{ext}")
        resized_img.save(resized_path)
        print(f"✓ Resized to {new_size}: {resized_path}")
    
    # Save final optimized version
    final_img = contrast_img  # Using contrast-enhanced grayscale as final
    final_path = os.path.join(output_dir, f"{name}_final_optimized{ext}")
    final_img.save(final_path, dpi=(300, 300))  # Save with standard OCR DPI
    print(f"\n✓ Final optimized image: {final_path}")
    
    return final_path


def compare_ocr_results(original_path, processed_path):
    """Compare OCR results between original and processed images"""
    try:
        import pytesseract
        
        print("\nComparing OCR results:")
        print("-" * 40)
        
        # OCR on original
        original_img = Image.open(original_path)
        original_text = pytesseract.image_to_string(original_img)
        
        # OCR on processed
        processed_img = Image.open(processed_path)
        processed_text = pytesseract.image_to_string(processed_img)
        
        print(f"Original OCR length: {len(original_text)} characters")
        print(f"Processed OCR length: {len(processed_text)} characters")
        
        if len(processed_text) > len(original_text):
            print("✓ Preprocessing improved OCR results!")
        else:
            print("⚠️  Preprocessing did not improve results")
            
        # Save text results
        output_dir = os.path.dirname(processed_path)
        with open(os.path.join(output_dir, "ocr_comparison.txt"), "w") as f:
            f.write("ORIGINAL IMAGE OCR:\n")
            f.write("-" * 30 + "\n")
            f.write(original_text)
            f.write("\n\nPROCESSED IMAGE OCR:\n")
            f.write("-" * 30 + "\n")
            f.write(processed_text)
            
        print("OCR comparison saved to ocr_comparison.txt")
        
    except ImportError:
        print("Note: Install pytesseract to compare OCR results")
    except Exception as e:
        print(f"Error comparing OCR results: {e}")


if __name__ == "__main__":
    input_image = "/workspaces/amy-project/workspace/downloads/hansman/100_4247.JPG"
    output_dir = "/workspaces/amy-project/workspace/processed/hansman_preprocessing/"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process the image
    final_image = preprocess_image(input_image, output_dir)
    
    # Compare OCR results if tesseract is available
    compare_ocr_results(input_image, final_image)