#!/usr/bin/env python3
"""
Advanced script to load and group Hansman documents based on various criteria.
Provides functions for grouping by category, time period, themes, and quality.
"""

import json
import glob
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HansmanDocumentGrouper:
    """Class for loading and grouping Hansman document summaries."""

    def __init__(
        self,
        summaries_dir: str = "/workspaces/amy-project/hansman_full_processing/summaries",
    ):
        self.summaries_dir = summaries_dir
        self.df = None

    def load_summaries(self) -> pd.DataFrame:
        """Load all JSON summaries into a DataFrame."""
        logger.info(f"Loading summaries from: {self.summaries_dir}")

        json_files = glob.glob(f"{self.summaries_dir}/*.json")
        logger.info(f"Found {len(json_files)} JSON files")

        all_summaries = []

        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Extract all fields
                summary_data = {
                    "file_name": Path(json_file).name,
                    "file_path": data.get("file_path", ""),
                    "category": data.get("category", "unknown"),
                    "summary": data.get("summary", ""),
                    "confidence_score": data.get("confidence_score", 0),
                    "content_type": data.get("content_type", ""),
                    "historical_period": data.get("historical_period", "Unknown"),
                    "suggested_folder_path": data.get("suggested_folder_path", ""),
                    "processing_time": data.get("processing_time", 0),
                    "created_at": data.get("created_at", ""),
                }

                # Extract nested fields
                key_entities = data.get("key_entities", {})
                summary_data["people"] = key_entities.get("people", [])
                summary_data["organizations"] = key_entities.get("organizations", [])
                summary_data["locations"] = key_entities.get("locations", [])
                summary_data["dates"] = key_entities.get("dates", [])

                summary_data["location_references"] = data.get(
                    "location_references", []
                )
                summary_data["classification_tags"] = data.get(
                    "classification_tags", []
                )
                summary_data["photo_subjects"] = data.get("photo_subjects", [])
                summary_data["related_themes"] = data.get("related_themes", [])

                quality_indicators = data.get("quality_indicators", {})
                summary_data["text_clarity"] = quality_indicators.get("text_clarity", 0)
                summary_data["historical_value"] = quality_indicators.get(
                    "historical_value", 0
                )
                summary_data["preservation_priority"] = quality_indicators.get(
                    "preservation_priority", 0
                )

                all_summaries.append(summary_data)

            except Exception as e:
                logger.error(f"Error processing file {json_file}: {e}")

        self.df = pd.DataFrame(all_summaries)
        logger.info(f"Created DataFrame with {len(self.df)} rows")

        # Convert datetime strings to datetime objects
        if "created_at" in self.df.columns:
            self.df["created_at"] = pd.to_datetime(
                self.df["created_at"], errors="coerce"
            )

        return self.df

    def group_by_category(self) -> Dict[str, pd.DataFrame]:
        """Group documents by category."""
        groups = {}
        for category, group_df in self.df.groupby("category"):
            groups[category] = group_df
            logger.info(f"Category '{category}': {len(group_df)} documents")
        return groups

    def group_by_historical_period(self) -> Dict[str, pd.DataFrame]:
        """Group documents by historical period."""
        groups = {}
        for period, group_df in self.df.groupby("historical_period"):
            groups[period] = group_df
            logger.info(f"Period '{period}': {len(group_df)} documents")
        return groups

    def group_by_quality(
        self, quality_threshold: float = 0.7
    ) -> Dict[str, pd.DataFrame]:
        """Group documents by quality indicators."""
        # Calculate average quality score
        quality_cols = ["text_clarity", "historical_value", "preservation_priority"]
        self.df["avg_quality"] = self.df[quality_cols].mean(axis=1)

        high_quality = self.df[self.df["avg_quality"] >= quality_threshold]
        medium_quality = self.df[
            (self.df["avg_quality"] >= 0.4)
            & (self.df["avg_quality"] < quality_threshold)
        ]
        low_quality = self.df[self.df["avg_quality"] < 0.4]

        groups = {
            "high_quality": high_quality,
            "medium_quality": medium_quality,
            "low_quality": low_quality,
        }

        for group_name, group_df in groups.items():
            logger.info(f"{group_name}: {len(group_df)} documents")

        return groups

    def group_by_themes(self) -> Dict[str, List[str]]:
        """Group documents by themes and classification tags."""
        theme_groups = {}

        # Combine classification tags and related themes
        for idx, row in self.df.iterrows():
            all_themes = []
            if isinstance(row["classification_tags"], list):
                all_themes.extend(row["classification_tags"])
            elif row["classification_tags"]:
                all_themes.extend(row["classification_tags"].split(", "))

            if "related_themes" in row and isinstance(row["related_themes"], list):
                all_themes.extend(row["related_themes"])

            for theme in all_themes:
                if theme not in theme_groups:
                    theme_groups[theme] = []
                theme_groups[theme].append(row["file_name"])

        # Sort by number of documents
        sorted_themes = sorted(
            theme_groups.items(), key=lambda x: len(x[1]), reverse=True
        )

        logger.info(f"Found {len(theme_groups)} unique themes")
        for theme, files in sorted_themes[:10]:  # Show top 10 themes
            logger.info(f"Theme '{theme}': {len(files)} documents")

        return dict(sorted_themes)

    def find_related_documents(
        self, file_name: str, similarity_threshold: float = 0.5
    ) -> pd.DataFrame:
        """Find documents related to a specific document based on common entities and themes."""
        source_doc = self.df[self.df["file_name"] == file_name]
        if source_doc.empty:
            logger.warning(f"Document {file_name} not found")
            return pd.DataFrame()

        source_doc = source_doc.iloc[0]
        related_scores = []

        for idx, row in self.df.iterrows():
            if row["file_name"] == file_name:
                continue

            score = 0

            # Check common people
            if source_doc["people"] and row["people"]:
                common_people = set(source_doc["people"]) & set(row["people"])
                score += len(common_people) * 0.3

            # Check common organizations
            if source_doc["organizations"] and row["organizations"]:
                common_orgs = set(source_doc["organizations"]) & set(
                    row["organizations"]
                )
                score += len(common_orgs) * 0.3

            # Check common locations
            if source_doc["locations"] and row["locations"]:
                common_locs = set(source_doc["locations"]) & set(row["locations"])
                score += len(common_locs) * 0.2

            # Check common themes
            source_themes = set(source_doc.get("classification_tags", []))
            row_themes = set(row.get("classification_tags", []))
            common_themes = source_themes & row_themes
            score += len(common_themes) * 0.2

            if score >= similarity_threshold:
                related_scores.append((row["file_name"], score))

        # Sort by score
        related_scores.sort(key=lambda x: x[1], reverse=True)

        # Return as DataFrame
        related_files = [f for f, s in related_scores]
        return self.df[self.df["file_name"].isin(related_files)]

    def create_visualization(
        self, output_dir: str = "/workspaces/amy-project/workspace/reports"
    ):
        """Create visualizations of the document collection."""
        plt.style.use("seaborn-v0_8")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Category distribution
        category_counts = self.df["category"].value_counts()
        ax1.pie(category_counts.values, labels=category_counts.index, autopct="%1.1f%%")
        ax1.set_title("Document Distribution by Category")

        # Historical period distribution
        period_counts = self.df["historical_period"].value_counts()
        ax2.bar(range(len(period_counts)), period_counts.values)
        ax2.set_xticks(range(len(period_counts)))
        ax2.set_xticklabels(period_counts.index, rotation=45, ha="right")
        ax2.set_title("Documents by Historical Period")
        ax2.set_ylabel("Count")

        # Quality distribution
        quality_cols = ["text_clarity", "historical_value", "preservation_priority"]
        quality_data = self.df[quality_cols].mean()
        ax3.bar(quality_data.index, quality_data.values)
        ax3.set_title("Average Quality Indicators")
        ax3.set_ylabel("Average Score")
        ax3.set_ylim(0, 1)

        # Top themes
        all_themes = []
        for tags in self.df["classification_tags"]:
            if isinstance(tags, list):
                all_themes.extend(tags)
            elif tags:
                all_themes.extend(tags.split(", "))

        theme_counts = pd.Series(all_themes).value_counts().head(10)
        ax4.barh(range(len(theme_counts)), theme_counts.values)
        ax4.set_yticks(range(len(theme_counts)))
        ax4.set_yticklabels(theme_counts.index)
        ax4.set_title("Top 10 Classification Tags")
        ax4.set_xlabel("Count")

        plt.tight_layout()
        output_path = f"{output_dir}/hansman_document_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved visualization to: {output_path}")
        plt.close()

    def export_groups(
        self, output_dir: str = "/workspaces/amy-project/workspace/reports"
    ):
        """Export various groupings to Excel files."""
        output_path = f"{output_dir}/hansman_document_groups.xlsx"

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # Export full dataset
            self.df.to_excel(writer, sheet_name="All Documents", index=False)

            # Export by category
            category_groups = self.group_by_category()
            for category, group_df in category_groups.items():
                sheet_name = f"Category_{category[:25]}"  # Limit sheet name length
                group_df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Export by period
            period_groups = self.group_by_historical_period()
            for period, group_df in period_groups.items():
                sheet_name = f"Period_{period[:25]}"
                group_df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Export quality groups
            quality_groups = self.group_by_quality()
            for quality_level, group_df in quality_groups.items():
                group_df.to_excel(writer, sheet_name=quality_level, index=False)

        logger.info(f"Exported groups to: {output_path}")


def main():
    """Main function to demonstrate document grouping capabilities."""
    grouper = HansmanDocumentGrouper()

    # Load the data
    df = grouper.load_summaries()
    print(f"\nLoaded {len(df)} documents")

    # Create various groupings
    print("\n" + "=" * 50)
    print("GROUPING BY CATEGORY")
    print("=" * 50)
    category_groups = grouper.group_by_category()

    print("\n" + "=" * 50)
    print("GROUPING BY HISTORICAL PERIOD")
    print("=" * 50)
    period_groups = grouper.group_by_historical_period()

    print("\n" + "=" * 50)
    print("GROUPING BY QUALITY")
    print("=" * 50)
    quality_groups = grouper.group_by_quality()

    print("\n" + "=" * 50)
    print("GROUPING BY THEMES")
    print("=" * 50)
    theme_groups = grouper.group_by_themes()

    # Create visualizations
    grouper.create_visualization()

    # Export groups
    grouper.export_groups()

    # Example: Find related documents
    example_file = df.iloc[0]["file_name"]
    print(f"\n" + "=" * 50)
    print(f"DOCUMENTS RELATED TO: {example_file}")
    print("=" * 50)
    related = grouper.find_related_documents(example_file)
    if not related.empty:
        print(related[["file_name", "category", "summary"]].head())

    return grouper


if __name__ == "__main__":
    grouper = main()
