#!/usr/bin/env python3
"""
Compare OCR quality metrics between original and preprocessed images
"""

import os
import json
from collections import Counter
import re


def analyze_ocr_text(text):
    """Analyze OCR text quality metrics"""
    metrics = {
        "total_characters": len(text),
        "total_words": len(text.split()),
        "lines": len(text.strip().split("\n")),
        "avg_word_length": 0,
        "gibberish_score": 0,
        "punctuation_ratio": 0,
        "uppercase_ratio": 0,
        "special_char_ratio": 0,
        "coherence_score": 0,
    }

    # Word analysis
    words = text.split()
    if words:
        metrics["avg_word_length"] = sum(len(w) for w in words) / len(words)

        # Count gibberish (words with unusual character patterns)
        gibberish_patterns = [
            r"^[^aeiouAEIOU]{4,}$",  # No vowels
            r"^[aeiouAEIOU]{4,}$",  # Only vowels
            r"(.)\1{3,}",  # Repeated characters
            r"^[^a-zA-Z]+$",  # No letters
        ]

        gibberish_count = 0
        for word in words:
            for pattern in gibberish_patterns:
                if re.match(pattern, word):
                    gibberish_count += 1
                    break

        metrics["gibberish_score"] = gibberish_count / len(words) if words else 0

    # Character analysis
    if text:
        char_counts = Counter(text)
        total_chars = len(text)

        # Punctuation ratio
        punctuation = sum(char_counts[c] for c in ".,;:!?\"'")
        metrics["punctuation_ratio"] = punctuation / total_chars

        # Uppercase ratio
        uppercase = sum(1 for c in text if c.isupper())
        metrics["uppercase_ratio"] = uppercase / total_chars

        # Special character ratio
        special = sum(1 for c in text if not c.isalnum() and not c.isspace())
        metrics["special_char_ratio"] = special / total_chars

        # Simple coherence score (based on common English patterns)
        common_words = {"the", "of", "and", "to", "in", "a", "is", "that", "it", "was"}
        text_lower = text.lower()
        coherence_count = sum(1 for word in common_words if word in text_lower.split())
        metrics["coherence_score"] = coherence_count / 10  # Normalize to 0-1

    return metrics


def load_ocr_results(file_path):
    """Load OCR results from file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return ""


def compare_ocr_quality(original_path, processed_path):
    """Compare OCR quality between original and processed images"""
    print("OCR Quality Comparison")
    print("=" * 50)

    # Load OCR results
    original_text = load_ocr_results(original_path)
    processed_text = load_ocr_results(processed_path)

    # Analyze both texts
    original_metrics = analyze_ocr_text(original_text)
    processed_metrics = analyze_ocr_text(processed_text)

    # Display comparison
    print(f"\n{'Metric':<20} {'Original':<15} {'Processed':<15} {'Change':<10}")
    print("-" * 60)

    for metric in original_metrics:
        orig_val = original_metrics[metric]
        proc_val = processed_metrics[metric]

        if isinstance(orig_val, float):
            change = ((proc_val - orig_val) / orig_val * 100) if orig_val > 0 else 0
            print(f"{metric:<20} {orig_val:<15.3f} {proc_val:<15.3f} {change:>+8.1f}%")
        else:
            change = proc_val - orig_val
            print(f"{metric:<20} {orig_val:<15d} {proc_val:<15d} {change:>+8d}")

    # Overall quality assessment
    print("\nQuality Assessment:")
    print("-" * 30)

    improvements = 0
    if processed_metrics["gibberish_score"] < original_metrics["gibberish_score"]:
        print("✓ Reduced gibberish")
        improvements += 1

    if processed_metrics["coherence_score"] > original_metrics["coherence_score"]:
        print("✓ Improved coherence")
        improvements += 1

    if processed_metrics["total_words"] > original_metrics["total_words"]:
        print("✓ More words detected")
        improvements += 1

    if (
        processed_metrics["avg_word_length"] > 2
        and processed_metrics["avg_word_length"] < 10
    ):
        print("✓ Reasonable word lengths")
        improvements += 1

    if improvements >= 2:
        print(
            f"\n✓ Overall: Preprocessing improved OCR quality ({improvements}/4 metrics improved)"
        )
    else:
        print(f"\n⚠️  Overall: Limited improvement ({improvements}/4 metrics improved)")

    return original_metrics, processed_metrics


if __name__ == "__main__":
    # Compare the original OCR with a hypothetical processed version
    original_ocr = (
        "/workspaces/amy-project/hansman_results/ocr_results/100_4247_ocr.txt"
    )

    # For this example, we'll use the same file since we don't have actual processed OCR yet
    # In a real scenario, this would be the OCR result from a preprocessed image
    processed_ocr = original_ocr

    compare_ocr_quality(original_ocr, processed_ocr)

    # Also analyze just the original to show current state
    print("\n\nDetailed Analysis of Current OCR Results:")
    print("=" * 50)

    original_text = load_ocr_results(original_ocr)
    metrics = analyze_ocr_text(original_text)

    print("Current OCR metrics:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.3f}")
        else:
            print(f"  {metric}: {value}")

    # Show sample of the OCR text
    print("\nSample of OCR output (first 500 chars):")
    print("-" * 40)
    print(original_text[:500] + "..." if len(original_text) > 500 else original_text)
