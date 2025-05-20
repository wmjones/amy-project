"""
Unit tests for the report generator module.
"""

import pytest
import json
import time
from pathlib import Path
from datetime import datetime

from src.utils.report_generator import ReportGenerator


class TestReportGenerator:
    """Test the ReportGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.report_gen = ReportGenerator()
        
        # Add some test data
        self.test_file = Path("/test/document.pdf")
        self.test_metadata = {
            'document_type': 'invoice',
            'dates': {'document_date': '2023-10-15'},
            'entities': {'organizations': ['Test Corp']}
        }
    
    def test_record_file_movement(self, tmp_path):
        """Test recording file movements."""
        # Create actual file to get real stat
        source = tmp_path / "source" / "file.pdf"
        source.parent.mkdir(exist_ok=True)
        source.write_text("test content")
        
        dest = tmp_path / "dest" / "file.pdf"
        
        self.report_gen.record_file_movement(
            source, dest, self.test_metadata, "Test Rule"
        )
        
        assert len(self.report_gen.file_movements) == 1
        movement = self.report_gen.file_movements[0]
        assert movement['source'] == str(source)
        assert movement['destination'] == str(dest)
        assert movement['type'] == 'invoice'
        assert movement['rule'] == "Test Rule"
        
        # Check statistics updated
        assert self.report_gen.rule_usage["Test Rule"] == 1
        assert self.report_gen.type_distribution['invoice'] == 1
    
    def test_record_error(self):
        """Test recording errors."""
        error = ValueError("Test error")
        self.report_gen.record_error(self.test_file, error, "processing")
        
        assert len(self.report_gen.errors) == 1
        error_info = self.report_gen.errors[0]
        assert error_info['file'] == str(self.test_file)
        assert error_info['error'] == "Test error"
        assert error_info['type'] == "ValueError"
        assert error_info['context'] == "processing"
    
    def test_record_operation(self):
        """Test recording operations."""
        self.report_gen.record_operation("copy")
        self.report_gen.record_operation("copy")
        self.report_gen.record_operation("move")
        
        assert self.report_gen.operation_stats["copy"] == 2
        assert self.report_gen.operation_stats["move"] == 1
    
    def test_size_distribution(self):
        """Test size distribution categorization."""
        # Test various file sizes
        sizes = [
            (50 * 1024, '< 100KB'),
            (500 * 1024, '100KB - 1MB'),
            (5 * 1024 * 1024, '1MB - 10MB'),
            (50 * 1024 * 1024, '10MB - 100MB'),
            (200 * 1024 * 1024, '> 100MB')
        ]
        
        for size, expected_category in sizes:
            self.report_gen._update_size_distribution(size)
        
        assert self.report_gen.size_distribution['< 100KB'] == 1
        assert self.report_gen.size_distribution['100KB - 1MB'] == 1
        assert self.report_gen.size_distribution['1MB - 10MB'] == 1
        assert self.report_gen.size_distribution['10MB - 100MB'] == 1
        assert self.report_gen.size_distribution['> 100MB'] == 1
    
    def test_date_distribution(self):
        """Test date distribution categorization."""
        dates = [
            '2023-10-15',
            '2023-10-20',
            '2023-11-01',
            'invalid-date'
        ]
        
        for date in dates:
            self.report_gen._update_date_distribution(date)
        
        assert self.report_gen.date_distribution['2023-10'] == 2
        assert self.report_gen.date_distribution['2023-11'] == 1
        assert self.report_gen.date_distribution['unknown'] == 1
    
    def test_generate_text_report(self):
        """Test text report generation."""
        # Add some test data
        self.report_gen.record_operation("copy")
        self.report_gen.record_operation("move")
        self.report_gen.record_error(self.test_file, ValueError("Test"), "test")
        
        report = self.report_gen.generate_summary_report(
            output_format='text',
            total_files=100,
            processed_files=90,
            successful_files=85,
            failed_files=5,
            skipped_files=10
        )
        
        assert "File Organization Report" in report
        assert "Total files:      100" in report
        assert "Processed:        90" in report
        assert "Successful:       85" in report
        assert "Failed:           5" in report
        assert "copy" in report
        assert "move" in report
    
    def test_generate_json_report(self):
        """Test JSON report generation."""
        report_json = self.report_gen.generate_summary_report(
            output_format='json',
            total_files=100,
            processed_files=90,
            successful_files=85,
            failed_files=5,
            skipped_files=10
        )
        
        report = json.loads(report_json)
        assert report['summary']['total_files'] == 100
        assert report['summary']['processed_files'] == 90
        assert report['summary']['successful_files'] == 85
        assert report['summary']['failed_files'] == 5
        assert 'performance' in report
        assert 'operations' in report
    
    def test_generate_html_report(self):
        """Test HTML report generation."""
        report = self.report_gen.generate_summary_report(
            output_format='html',
            total_files=100,
            processed_files=90,
            successful_files=85,
            failed_files=5,
            skipped_files=10
        )
        
        assert '<html>' in report
        assert '<h1>File Organization Report</h1>' in report
        assert 'Total files:' in report
        assert '100' in report
    
    def test_performance_metrics(self):
        """Test performance metrics calculation."""
        # Create report generator with specific start time
        start_time = time.time() - 60  # 60 seconds ago
        report_gen = ReportGenerator(start_time=start_time)
        
        report_json = report_gen.generate_summary_report(
            output_format='json',
            total_files=100,
            processed_files=90,
            successful_files=85,
            failed_files=5,
            skipped_files=10
        )
        
        report = json.loads(report_json)
        assert report['performance']['elapsed_time'] >= 60
        assert report['performance']['files_per_second'] <= 90/60
    
    def test_save_all_reports(self, tmp_path):
        """Test saving all report types."""
        # Add some test data
        self.report_gen.record_operation("copy")
        
        # Create actual file to get real stat
        source = tmp_path / "source" / "file.pdf"
        source.parent.mkdir(exist_ok=True)
        source.write_text("test content")
        
        dest = tmp_path / "dest" / "file.pdf"
        
        self.report_gen.record_file_movement(
            source, dest, self.test_metadata, "Test Rule"
        )
        self.report_gen.record_error(self.test_file, ValueError("Test"), "test")
        
        # Save reports
        output_dir = tmp_path / "reports"
        # Override save_all_reports to use proper parameters
        original_method = self.report_gen.generate_summary_report
        def mock_summary_report(format):
            return original_method(
                format,
                total_files=10,
                processed_files=8,
                successful_files=7,
                failed_files=1,
                skipped_files=2
            )
        self.report_gen.generate_summary_report = mock_summary_report
        self.report_gen.save_all_reports(output_dir)
        
        # Check files were created
        assert (output_dir / "summary.txt").exists()
        assert (output_dir / "summary.html").exists()
        assert (output_dir / "summary.json").exists()
        assert (output_dir / "movements.json").exists()
        assert (output_dir / "movements.csv").exists()
        assert (output_dir / "errors.json").exists()
        assert (output_dir / "visualization_data.json").exists()
    
    def test_movement_report_csv(self, tmp_path):
        """Test CSV movement report generation."""
        # Create actual file to get real stat
        source = tmp_path / "source" / "file.pdf"
        source.parent.mkdir(exist_ok=True)
        source.write_text("test content")
        
        dest = tmp_path / "dest" / "file.pdf"
        
        self.report_gen.record_file_movement(
            source, dest, self.test_metadata, "Test Rule"
        )
        
        # Generate CSV report
        csv_path = tmp_path / "movements.csv"
        self.report_gen.generate_movement_report(csv_path, format='csv')
        
        assert csv_path.exists()
        
        # Check CSV content
        with open(csv_path) as f:
            content = f.read()
            assert "source,destination" in content
            assert "source" in content
            assert "dest" in content
    
    def test_error_report(self, tmp_path):
        """Test error report generation."""
        # Add various errors
        self.report_gen.record_error(
            Path("/file1.pdf"), ValueError("Value error"), "processing"
        )
        self.report_gen.record_error(
            Path("/file2.pdf"), TypeError("Type error"), "processing"
        )
        self.report_gen.record_error(
            Path("/file3.pdf"), IOError("IO error"), "file_access"
        )
        
        # Generate error report
        error_path = tmp_path / "errors.json"
        self.report_gen.generate_error_report(error_path)
        
        assert error_path.exists()
        
        # Check report content
        with open(error_path) as f:
            report = json.load(f)
        
        assert report['total_errors'] == 3
        assert report['errors_by_type']['ValueError'] == 1
        assert report['errors_by_type']['TypeError'] == 1
        assert report['errors_by_type']['OSError'] == 1  # IOError is now OSError in Python 3
        assert report['errors_by_context']['processing'] == 2
        assert report['errors_by_context']['file_access'] == 1
    
    def test_visualization_data(self):
        """Test visualization data generation."""
        # Add some data
        self.report_gen.type_distribution['pdf'] = 10
        self.report_gen.type_distribution['docx'] = 5
        self.report_gen.rule_usage['Rule1'] = 8
        self.report_gen.rule_usage['Rule2'] = 7
        
        viz_data = self.report_gen.generate_visualization_data()
        
        assert viz_data['file_types']['pdf'] == 10
        assert viz_data['file_types']['docx'] == 5
        assert viz_data['rules']['Rule1'] == 8
        assert viz_data['rules']['Rule2'] == 7


if __name__ == "__main__":
    pytest.main([__file__])