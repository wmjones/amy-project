#!/usr/bin/env python3
"""
Demo script showing how to use the file processing engine.
"""

import os
import sys
import logging
from pathlib import Path
from src.file_access.processor import FileProcessor
from src.file_access.ocr_processor import OCRProcessor
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def demo_file_processor():
    """Demonstrate file processing capabilities."""
    print("=== File Processor Demo ===\n")

    # Initialize processor
    processor = FileProcessor(max_chunk_size=5000, enable_ocr=True)

    # Demo files (create some if they don't exist)
    demo_dir = Path("demo_files")
    demo_dir.mkdir(exist_ok=True)

    # Create a sample text file
    text_file = demo_dir / "sample.txt"
    if not text_file.exists():
        text_file.write_text(
            """
        This is a sample document for demonstrating file processing.

        It contains multiple paragraphs and some structure to show
        how the processor handles different content types.

        Key features:
        - Text extraction
        - Encoding detection
        - Content chunking for large files
        """
        )
        print(f"Created sample text file: {text_file}")

    # Process files
    files_to_process = [
        text_file,
        # Add more files here as they exist
    ]

    for file_path in files_to_process:
        if not file_path.exists():
            continue

        print(f"\nProcessing: {file_path}")
        result = processor.process_file(str(file_path))

        if result.success:
            print(f"✅ Success!")
            print(f"Format: {result.format}")
            print(f"Content length: {len(result.content)} characters")
            print(f"Metadata: {json.dumps(result.metadata, indent=2)}")

            if result.chunks:
                print(f"Chunks: {len(result.chunks)}")

            # Show preview
            preview = (
                result.content[:200] + "..."
                if len(result.content) > 200
                else result.content
            )
            print(f"\nContent preview:")
            print(preview)
        else:
            print(f"❌ Error: {result.error}")

    print("\n=== Demo Complete ===")


def demo_ocr_processor():
    """Demonstrate OCR processing."""
    print("\n=== OCR Processor Demo ===\n")

    # Initialize OCR processor
    ocr_processor = OCRProcessor(
        language="eng", enhance_contrast=True, denoise=True, deskew=True
    )

    # Demo with an image (if exists)
    demo_dir = Path("demo_files")
    image_files = list(demo_dir.glob("*.jpg")) + list(demo_dir.glob("*.png"))

    if not image_files:
        print("No image files found in demo_files directory")
        print("Add some image files to test OCR functionality")
        return

    for image_file in image_files[:2]:  # Process first 2 images
        print(f"\nProcessing image: {image_file}")

        result = ocr_processor.process_image(str(image_file))

        if result["success"]:
            print(f"✅ OCR Success!")
            print(f"Confidence: {result['confidence']:.2f}%")
            print(f"Metadata: {json.dumps(result['metadata'], indent=2)}")

            # Show text preview
            text = result["text"]
            preview = text[:300] + "..." if len(text) > 300 else text
            print(f"\nExtracted text:")
            print(preview)
        else:
            print(f"❌ OCR Error: {result['error']}")

    print("\n=== OCR Demo Complete ===")


def demo_batch_processing():
    """Demonstrate batch processing."""
    print("\n=== Batch Processing Demo ===\n")

    processor = FileProcessor()
    demo_dir = Path("demo_files")

    # Get all files in demo directory
    files = list(demo_dir.glob("*.*"))

    if not files:
        print("No files found in demo_files directory")
        return

    print(f"Processing {len(files)} files...")

    def progress_callback(current, total):
        print(f"Progress: {current}/{total} ({current/total*100:.0f}%)")

    results = processor.batch_process(
        [str(f) for f in files], progress_callback=progress_callback
    )

    # Summary
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful

    print(f"\nBatch processing complete:")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")

    # Show results
    for i, (file, result) in enumerate(zip(files, results)):
        status = "✅" if result.success else "❌"
        print(f"{status} {file.name} - {result.format}")


def main():
    """Run all demos."""
    demos = {
        "1": ("File Processor Demo", demo_file_processor),
        "2": ("OCR Processor Demo", demo_ocr_processor),
        "3": ("Batch Processing Demo", demo_batch_processing),
    }

    if len(sys.argv) > 1 and sys.argv[1] in demos:
        name, func = demos[sys.argv[1]]
        print(f"Running {name}...")
        func()
    else:
        print("File Processing Engine Demo")
        print("===========================")
        print("\nAvailable demos:")
        for key, (name, _) in demos.items():
            print(f"{key}. {name}")
        print("\nUsage: python processor_demo.py [demo_number]")
        print("\nRunning all demos...")

        for _, func in demos.values():
            func()
            print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()
