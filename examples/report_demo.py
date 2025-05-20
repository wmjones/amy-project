"""
Demo script showing report generation capabilities.
"""

import sys
import time
import shutil
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.report_generator import ReportGenerator
from src.utils.folder_visualizer import FolderVisualizer


def create_demo_data(temp_dir: Path):
    """Create demo directory structure and files."""
    # Create directories
    (temp_dir / "source").mkdir(exist_ok=True)
    (temp_dir / "organized").mkdir(exist_ok=True)

    # Create source files
    source_files = [
        ("invoice_2023_001.pdf", "Invoices", 1024 * 150),  # 150KB
        ("invoice_2023_002.pdf", "Invoices", 1024 * 200),  # 200KB
        ("receipt_store1.jpg", "Receipts", 1024 * 500),  # 500KB
        ("receipt_store2.jpg", "Receipts", 1024 * 800),  # 800KB
        ("photo_vacation.jpg", "Photos", 1024 * 1024 * 3),  # 3MB
        ("contract_client.pdf", "Legal", 1024 * 1024 * 2),  # 2MB
        ("report_2023.docx", "Reports", 1024 * 1024 * 5),  # 5MB
    ]

    for filename, _, size in source_files:
        file_path = temp_dir / "source" / filename
        file_path.write_text("x" * size)  # Create file with specific size

    # Create organized structure
    organized_dirs = [
        "Invoices/2023/10",
        "Receipts/Store1/2023-10",
        "Receipts/Store2/2023-10",
        "Photos/2023/2023-08-15",
        "Legal/Contracts/ClientCorp/2023",
        "Reports/Annual/2023",
    ]

    for dir_path in organized_dirs:
        (temp_dir / "organized" / dir_path).mkdir(parents=True, exist_ok=True)

    # Simulate file organization
    organized_files = [
        ("Invoices/2023/10/invoice_2023_001.pdf", source_files[0]),
        ("Invoices/2023/10/invoice_2023_002.pdf", source_files[1]),
        ("Receipts/Store1/2023-10/receipt_store1.jpg", source_files[2]),
        ("Receipts/Store2/2023-10/receipt_store2.jpg", source_files[3]),
        ("Photos/2023/2023-08-15/photo_vacation.jpg", source_files[4]),
        ("Legal/Contracts/ClientCorp/2023/contract_client.pdf", source_files[5]),
        ("Reports/Annual/2023/report_2023.docx", source_files[6]),
    ]

    for dest_path, (source_name, _, size) in organized_files:
        dest_file = temp_dir / "organized" / dest_path
        dest_file.write_text("x" * size)


def demonstrate_report_generation():
    """Demonstrate report generation features."""
    print("=== Report Generation Demo ===\n")

    # Create temporary directory for demo
    temp_dir = Path("report_demo_temp")
    temp_dir.mkdir(exist_ok=True)

    try:
        # Create demo data
        print("1. Creating demo data...")
        create_demo_data(temp_dir)

        # Initialize report generator
        report_gen = ReportGenerator()

        # Simulate file processing
        print("2. Simulating file processing...")

        # Process files and record movements
        source_files = [
            ("invoice_2023_001.pdf", "invoice", "2023-10-01"),
            ("invoice_2023_002.pdf", "invoice", "2023-10-15"),
            ("receipt_store1.jpg", "receipt", "2023-10-05"),
            ("receipt_store2.jpg", "receipt", "2023-10-12"),
            ("photo_vacation.jpg", "photo", "2023-08-15"),
            ("contract_client.pdf", "contract", "2023-01-10"),
            ("report_2023.docx", "report", "2023-12-01"),
        ]

        rules = [
            "Invoices by client and year",
            "Receipts by store and month",
            "Photos by date taken",
            "Contracts by party",
            "Reports by type and year",
        ]

        for i, (filename, doc_type, date) in enumerate(source_files):
            source = temp_dir / "source" / filename

            # Determine destination based on type
            if doc_type == "invoice":
                dest = temp_dir / "organized" / f"Invoices/2023/10/{filename}"
                rule = rules[0]
            elif doc_type == "receipt":
                store = "Store1" if "store1" in filename else "Store2"
                dest = temp_dir / "organized" / f"Receipts/{store}/2023-10/{filename}"
                rule = rules[1]
            elif doc_type == "photo":
                dest = temp_dir / "organized" / f"Photos/2023/2023-08-15/{filename}"
                rule = rules[2]
            elif doc_type == "contract":
                dest = (
                    temp_dir
                    / "organized"
                    / f"Legal/Contracts/ClientCorp/2023/{filename}"
                )
                rule = rules[3]
            else:
                dest = temp_dir / "organized" / f"Reports/Annual/2023/{filename}"
                rule = rules[4]

            # Record the movement
            metadata = {
                "document_type": doc_type,
                "dates": {"document_date": date},
                "entities": {"organizations": ["Test Corp"]},
            }

            report_gen.record_file_movement(source, dest, metadata, rule)
            report_gen.record_operation("copy")

            # Simulate processing time
            time.sleep(0.1)

        # Add some simulated errors
        report_gen.record_error(
            Path("/missing/file.pdf"),
            FileNotFoundError("File not found"),
            "file_access",
        )
        report_gen.record_error(
            Path("/corrupt/file.pdf"), ValueError("Invalid PDF format"), "processing"
        )

        # Generate reports
        print("\n3. Generating reports...")

        # Text report
        text_report = report_gen.generate_summary_report(
            output_format="text",
            total_files=10,
            processed_files=7,
            successful_files=7,
            failed_files=0,
            skipped_files=3,
        )
        print("\nText Report:")
        print("-" * 40)
        print(text_report)

        # Save all reports
        reports_dir = temp_dir / "reports"
        report_gen.save_all_reports(reports_dir)
        print(f"\nReports saved to: {reports_dir}")

        # Demonstrate folder visualization
        print("\n4. Creating folder visualizations...")
        visualizer = FolderVisualizer()

        # Visualize source structure
        print("\nSource Directory Structure:")
        print(
            visualizer.visualize_directory_tree(
                temp_dir / "source", max_depth=2, show_files=True
            )
        )

        # Visualize organized structure
        print("\n\nOrganized Directory Structure:")
        print(
            visualizer.visualize_directory_tree(
                temp_dir / "organized", max_depth=3, show_files=True
            )
        )

        # Create summary visualization
        print("\n\nOrganization Summary:")
        print(visualizer.create_summary_visualization(temp_dir / "organized"))

        # Generate size map
        size_map = visualizer.generate_size_map(temp_dir / "organized")
        print("\n\nDirectory Size Information:")
        for path, info in sorted(size_map.items())[:5]:  # Show top 5
            print(
                f"{path:30s} {info['size_formatted']:>10s} ({info['file_count']} files)"
            )

        # Show HTML report preview
        html_report = report_gen.generate_summary_report("html")
        html_path = reports_dir / "summary.html"
        print(f"\n\nHTML report saved to: {html_path}")
        print("Open in browser to view formatted report.")

        # Show movement report
        print("\n\nSample movement records:")
        movements = report_gen.file_movements[:3]  # Show first 3
        for movement in movements:
            print(f"  {Path(movement['source']).name} -> {movement['rule']}")

        # Show visualization data
        viz_data = report_gen.generate_visualization_data()
        print("\n\nVisualization data summary:")
        print(f"  File types: {len(viz_data['file_types'])} types")
        print(f"  Rules used: {len(viz_data['rules'])} rules")
        print(f"  Operations: {viz_data['operations']}")

    finally:
        # Cleanup
        print("\n\nCleaning up demo files...")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        print("Demo completed!")


if __name__ == "__main__":
    demonstrate_report_generation()
