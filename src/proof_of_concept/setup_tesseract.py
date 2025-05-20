#!/usr/bin/env python3
"""
Setup script for Tesseract OCR installation and verification.
Part of Task 18 - OCR Proof of Concept for Hansman Syracuse Collection.
"""

import subprocess
import sys
import os
import platform
from pathlib import Path


def check_tesseract_installed():
    """Check if Tesseract is already installed."""
    try:
        result = subprocess.run(
            ["tesseract", "--version"], capture_output=True, text=True
        )
        if result.returncode == 0:
            print("✓ Tesseract is already installed:")
            print(result.stdout)
            return True
    except FileNotFoundError:
        print("✗ Tesseract is not installed")
        return False
    return False


def install_tesseract():
    """Install Tesseract based on the operating system."""
    system = platform.system()

    if system == "Linux":
        print("Installing Tesseract on Linux...")
        try:
            # For Ubuntu/Debian
            subprocess.run(["sudo", "apt-get", "update"], check=True)
            subprocess.run(
                [
                    "sudo",
                    "apt-get",
                    "install",
                    "-y",
                    "tesseract-ocr",
                    "tesseract-ocr-eng",
                ],
                check=True,
            )
            print("✓ Tesseract installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("Error: Could not install Tesseract. Please install manually:")
            print("  sudo apt-get install tesseract-ocr tesseract-ocr-eng")
            return False

    elif system == "Darwin":  # macOS
        print("Installing Tesseract on macOS...")
        try:
            subprocess.run(["brew", "install", "tesseract"], check=True)
            print("✓ Tesseract installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("Error: Could not install Tesseract. Please install manually:")
            print("  brew install tesseract")
            return False

    elif system == "Windows":
        print("Windows detected. Please install Tesseract manually:")
        print("1. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("2. Run the installer")
        print("3. Add Tesseract to your PATH")
        return False

    else:
        print(f"Unsupported operating system: {system}")
        return False


def verify_tesseract():
    """Verify Tesseract installation with a simple test."""
    print("\nVerifying Tesseract installation...")

    # Create a simple test image with text
    try:
        from PIL import Image, ImageDraw, ImageFont

        # Create a test image
        width, height = 200, 50
        image = Image.new("RGB", (width, height), color="white")
        draw = ImageDraw.Draw(image)

        # Add text
        text = "Test OCR"
        draw.text((10, 10), text, fill="black")

        # Save test image
        test_image_path = Path("test_ocr.png")
        image.save(test_image_path)

        # Run OCR on test image
        result = subprocess.run(
            ["tesseract", str(test_image_path), "stdout"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            extracted_text = result.stdout.strip()
            print(f"✓ OCR test successful")
            print(f"  Original text: '{text}'")
            print(f"  Extracted text: '{extracted_text}'")

            # Clean up
            test_image_path.unlink()
            return True
        else:
            print("✗ OCR test failed")
            return False

    except ImportError:
        print("Note: PIL (Pillow) not installed. Skipping visual test.")
        print("You can install it with: pip install Pillow")

        # Basic command test
        result = subprocess.run(
            ["tesseract", "--version"], capture_output=True, text=True
        )
        return result.returncode == 0


def check_language_packs():
    """Check available language packs."""
    print("\nChecking available language packs...")
    try:
        result = subprocess.run(
            ["tesseract", "--list-langs"], capture_output=True, text=True
        )
        if result.returncode == 0:
            languages = result.stdout.strip().split("\n")[1:]  # Skip header
            print(f"✓ Available languages: {', '.join(languages)}")

            # Check for recommended languages
            recommended = ["eng", "fra", "deu", "spa"]
            missing = [lang for lang in recommended if lang not in languages]

            if missing:
                print(f"\nRecommended languages missing: {', '.join(missing)}")
                print("You can install them with:")
                for lang in missing:
                    print(f"  sudo apt-get install tesseract-ocr-{lang}")
        else:
            print("✗ Could not list language packs")
    except Exception as e:
        print(f"Error checking languages: {e}")


def setup_project_structure():
    """Create necessary directories for the proof of concept."""
    directories = [
        "data/hansman_samples",
        "output/ocr_results",
        "output/summaries",
        "output/reports",
        "logs",
    ]

    print("\nSetting up project structure...")
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}")


def main():
    """Main setup function."""
    print("=== Tesseract OCR Setup for Hansman Syracuse Collection ===\n")

    # Check if already installed
    if not check_tesseract_installed():
        if install_tesseract():
            print("\nTesseract installed successfully!")
        else:
            print("\nPlease install Tesseract manually before proceeding.")
            sys.exit(1)

    # Verify installation
    if verify_tesseract():
        print("\n✓ Tesseract is working correctly")
    else:
        print("\n✗ Tesseract verification failed")
        sys.exit(1)

    # Check language packs
    check_language_packs()

    # Setup project structure
    setup_project_structure()

    print("\n=== Setup Complete ===")
    print(
        "Tesseract OCR is ready for the Hansman Syracuse Collection proof of concept!"
    )


if __name__ == "__main__":
    main()
