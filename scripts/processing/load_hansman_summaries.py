#!/usr/bin/env python3
"""
Script to load all JSON summaries from the Hansman processing folder into a CSV file.
"""

import json
import glob
import pandas as pd
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_summaries_to_csv(
    summaries_dir: str = "/workspaces/amy-project/hansman_full_processing/summaries",
    output_file: str = "/workspaces/amy-project/workspace/reports/hansman_summaries.csv",
) -> pd.DataFrame:
    """
    Load all JSON summary files from the specified directory and save to CSV.

    Args:
        summaries_dir: Path to the directory containing JSON summary files
        output_file: Path for the output CSV file

    Returns:
        pd.DataFrame: DataFrame containing the loaded summary data
    """
    logger.info(f"Loading summaries from: {summaries_dir}")

    # Find all JSON files in the directory
    json_files = glob.glob(f"{summaries_dir}/*.json")
    logger.info(f"Found {len(json_files)} JSON files")

    # List to store all the data
    all_summaries = []

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract all fields from the JSON
            summary_data = {
                "file_name": Path(json_file).name,
                "file_path": data.get("file_path", ""),
                "category": data.get("category", ""),
                "summary": data.get("summary", ""),
                "confidence_score": data.get("confidence_score", ""),
                "content_type": data.get("content_type", ""),
                "historical_period": data.get("historical_period", ""),
                "suggested_folder_path": data.get("suggested_folder_path", ""),
                "processing_time": data.get("processing_time", ""),
                "created_at": data.get("created_at", ""),
                "processed_at": data.get("processed_at", ""),
                "ocr_text": data.get("ocr_text", ""),
                "error_message": data.get("error_message", ""),
            }

            # Extract nested key_entities
            key_entities = data.get("key_entities", {})
            if isinstance(key_entities, dict):
                summary_data["people"] = key_entities.get("people", [])
                summary_data["organizations"] = key_entities.get("organizations", [])
                summary_data["locations"] = key_entities.get("locations", [])
                summary_data["dates"] = key_entities.get("dates", [])
            else:
                summary_data["people"] = []
                summary_data["organizations"] = []
                summary_data["locations"] = []
                summary_data["dates"] = []

            # Extract lists
            summary_data["location_references"] = data.get("location_references", [])
            summary_data["classification_tags"] = data.get("classification_tags", [])
            summary_data["photo_subjects"] = data.get("photo_subjects", [])
            summary_data["related_themes"] = data.get("related_themes", [])
            summary_data["related_documents"] = data.get("related_documents", [])
            summary_data["date_references"] = data.get("date_references", [])

            # Extract quality indicators
            quality_indicators = data.get("quality_indicators", {})
            if isinstance(quality_indicators, dict):
                summary_data["text_clarity"] = quality_indicators.get(
                    "text_clarity", ""
                )
                summary_data["historical_value"] = quality_indicators.get(
                    "historical_value", ""
                )
                summary_data["preservation_priority"] = quality_indicators.get(
                    "preservation_priority", ""
                )
            else:
                summary_data["text_clarity"] = ""
                summary_data["historical_value"] = ""
                summary_data["preservation_priority"] = ""

            # Extract claude metadata if present
            claude_metadata = data.get("claude_metadata", {})
            if isinstance(claude_metadata, dict):
                summary_data["claude_model"] = claude_metadata.get("model", "")
                summary_data["claude_tokens_used"] = claude_metadata.get(
                    "tokens_used", ""
                )
            else:
                summary_data["claude_model"] = ""
                summary_data["claude_tokens_used"] = ""

            all_summaries.append(summary_data)

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON file {json_file}: {e}")
        except Exception as e:
            logger.error(f"Error processing file {json_file}: {e}")

    # Create DataFrame
    df = pd.DataFrame(all_summaries)
    logger.info(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")

    # Convert list columns to string for CSV
    list_columns = [
        "people",
        "organizations",
        "locations",
        "dates",
        "classification_tags",
        "location_references",
        "photo_subjects",
        "related_themes",
        "related_documents",
        "date_references",
    ]

    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: ", ".join(x) if isinstance(x, list) else str(x)
            )

    # Sort by file name
    df = df.sort_values("file_name").reset_index(drop=True)

    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(output_file, index=False)
    logger.info(f"Saved CSV to: {output_file}")

    return df


def main():
    """Main function to load summaries and save to CSV."""
    df = load_summaries_to_csv()

    print(f"\nSuccessfully loaded {len(df)} documents")
    print(f"Columns in the CSV: {list(df.columns)}")
    print(
        f"\nCSV file saved to: /workspaces/amy-project/workspace/reports/hansman_summaries.csv"
    )


if __name__ == "__main__":
    main()
