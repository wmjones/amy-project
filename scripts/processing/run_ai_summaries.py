#!/usr/bin/env python3
"""Run AI summaries on OCR-processed files."""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import logging
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.metadata_extraction.ai_summarizer_fixed import AISummarizer
from src.claude_integration.client_fixed import ClaudeClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    output_dir = Path("./hansman_full_processing")
    state_file = output_dir / ".state" / "processing_state.json"

    # Load state
    with open(state_file, "r") as f:
        state = json.load(f)

    # Initialize AI summarizer
    claude_client = ClaudeClient()
    ai_summarizer = AISummarizer(claude_client=claude_client)

    # Find files that have OCR but no AI summary
    files_to_process = []
    for filename, ocr_path in state["ocr_processed"].items():
        if filename not in state["ai_processed"]:
            files_to_process.append(filename)

    print(f"Found {len(files_to_process)} files to process through AI")

    # Process each file
    ai_results = {}
    ai_processed = 0

    with tqdm(total=len(files_to_process), desc="AI Processing") as pbar:
        for filename in files_to_process:
            try:
                # Load OCR result
                ocr_path = state["ocr_processed"][filename]
                with open(ocr_path, "r") as f:
                    ocr_data = json.load(f)

                ocr_text = ocr_data.get("text", "")

                # Process AI summary
                logger.info(f"Processing AI summary for: {filename}")

                # Get the image file path
                file_path = output_dir / "downloads" / filename

                summary_result = ai_summarizer.summarize_document(
                    ocr_text=ocr_text,
                    file_path=file_path,
                    additional_context={
                        "collection": "Hansman Syracuse photo docs July 2015",
                        "processing_date": datetime.now().isoformat(),
                    },
                )

                if summary_result:
                    # Save AI summary
                    summaries_dir = output_dir / "summaries"
                    summaries_dir.mkdir(exist_ok=True)

                    summary_file = summaries_dir / f"{Path(filename).stem}_summary.json"

                    # Convert to dict and save
                    from dataclasses import asdict

                    summary_data = asdict(summary_result)

                    # Convert datetime objects
                    if "created_at" in summary_data and hasattr(
                        summary_data["created_at"], "isoformat"
                    ):
                        summary_data["created_at"] = summary_data[
                            "created_at"
                        ].isoformat()

                    summary_data["processed_at"] = datetime.now().isoformat()

                    with open(summary_file, "w") as f:
                        json.dump(summary_data, f, indent=2, default=str)

                    # Update state
                    state["ai_processed"][filename] = str(summary_file)
                    state["last_run"] = datetime.now().isoformat()

                    # Save state after each file
                    with open(state_file, "w") as f:
                        json.dump(state, f, indent=2)

                    ai_processed += 1

            except Exception as e:
                logger.error(f"Error processing AI for {filename}: {e}")
                state["errors"].append(
                    {"stage": "ai", "file": filename, "error": str(e)}
                )

            pbar.update(1)

    print(f"\nAI Processing complete: {ai_processed} files processed")

    # Save final state
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


if __name__ == "__main__":
    main()
