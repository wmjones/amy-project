#!/usr/bin/env python3
"""
Demo script showing how to use the Claude API client for document analysis.
"""

import os
import json
import logging
from pathlib import Path
from src.claude_integration.client import ClaudeClient
from src.claude_integration.prompts import PromptTemplates

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def demo_analyze_document():
    """Demonstrate single document analysis."""
    print("=== Claude Document Analysis Demo ===\n")

    # Initialize client
    try:
        client = ClaudeClient()
        print("✅ Claude client initialized successfully")

        # Validate connection
        if client.validate_connection():
            print("✅ API connection validated")
        else:
            print("❌ API connection failed")
            return

    except ValueError as e:
        print(f"❌ Error: {e}")
        print("Please set ANTHROPIC_API_KEY environment variable")
        return

    # Example document content
    sample_documents = {
        "invoice": """
        INVOICE #INV-2024-001
        Date: March 15, 2024

        Bill To:
        Acme Corporation
        123 Business St
        New York, NY 10001

        Services Rendered:
        - Consulting Services: $5,000
        - Software Development: $15,000
        - Support Package: $2,000

        Total Due: $22,000
        Payment Terms: Net 30
        """,
        "email": """
        From: john.doe@example.com
        To: jane.smith@company.com
        Date: March 20, 2024
        Subject: Project Update - Q1 Milestones

        Hi Jane,

        I wanted to give you an update on our Q1 milestones:

        1. Phase 1 development - Completed
        2. Testing environment - 90% complete
        3. Documentation - In progress

        We're on track to deliver by the end of the month.

        Best regards,
        John
        """,
        "report": """
        Annual Performance Report 2023

        Executive Summary:
        This report provides a comprehensive overview of our company's
        performance during the fiscal year 2023. Key highlights include:

        - Revenue growth of 25% year-over-year
        - Successful launch of 3 new products
        - Expansion into 2 new markets
        - Employee satisfaction score of 4.5/5

        The report contains detailed analysis of each department's
        contributions and recommendations for 2024.
        """,
    }

    # Analyze each document
    for doc_type, content in sample_documents.items():
        print(f"\n--- Analyzing {doc_type} ---")

        try:
            # Select appropriate prompt
            if doc_type == "invoice":
                prompt_template = PromptTemplates.get_financial_analysis_prompt()
            elif doc_type == "email":
                prompt_template = PromptTemplates.get_email_analysis_prompt()
            else:
                prompt_template = PromptTemplates.get_default_analysis_prompt()

            # Format prompt
            formatted_prompt = PromptTemplates.format_prompt(
                prompt_template, content=content
            )

            # Analyze document
            result = client.analyze_document(
                content=content,
                file_name=f"sample_{doc_type}.txt",
                file_type="txt",
                custom_prompt=formatted_prompt,
            )

            print(f"✅ Analysis complete")
            print(f"Tokens used: {result.tokens_used}")
            print(f"Confidence: {result.confidence_score}")

            # Try to parse and display metadata
            try:
                metadata = json.loads(result.content)
                print(f"Category: {metadata.get('category', 'Unknown')}")
                print(
                    f"Suggested folder: {metadata.get('suggested_folder', 'Unknown')}"
                )
                if "summary" in metadata:
                    print(f"Summary: {metadata['summary'][:100]}...")
            except json.JSONDecodeError:
                print("Metadata: Could not parse as JSON")
                print(f"Raw response: {result.content[:200]}...")

        except Exception as e:
            print(f"❌ Error analyzing {doc_type}: {e}")

    print("\n=== Demo Complete ===")


def demo_batch_analysis():
    """Demonstrate batch document analysis."""
    print("\n=== Batch Analysis Demo ===\n")

    # Initialize client
    try:
        client = ClaudeClient()
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Create sample batch
    documents = [
        {
            "content": "Meeting notes from January 15, 2024...",
            "file_name": "meeting_notes.txt",
            "file_type": "txt",
        },
        {
            "content": "Contract agreement between parties...",
            "file_name": "contract.pdf",
            "file_type": "pdf",
        },
        {
            "content": "Monthly expense report for March 2024...",
            "file_name": "expenses.xlsx",
            "file_type": "xlsx",
        },
    ]

    print(f"Analyzing batch of {len(documents)} documents...")

    # Define progress callback
    def progress_update(completed, total):
        print(f"Progress: {completed}/{total} documents")

    # Analyze batch
    try:
        results = client.analyze_batch(
            documents, batch_size=2, progress_callback=progress_update
        )

        print(f"\n✅ Batch analysis complete")
        print(
            f"Successfully analyzed: {sum(1 for r in results if r.confidence_score > 0)}"
        )
        print(f"Errors: {sum(1 for r in results if r.metadata.get('error', False))}")

        # Display results
        for i, result in enumerate(results):
            doc = documents[i]
            print(f"\n{doc['file_name']}:")
            if result.metadata.get("error"):
                print(f"  ❌ Error: {result.metadata['error_message']}")
            else:
                print(f"  ✅ Confidence: {result.confidence_score}")
                print(f"  Tokens: {result.tokens_used}")

    except Exception as e:
        print(f"❌ Error in batch analysis: {e}")


if __name__ == "__main__":
    # Run demos
    demo_analyze_document()
    print("\n" + "=" * 50 + "\n")
    demo_batch_analysis()
