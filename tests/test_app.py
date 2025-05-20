"""
Test module for the main application controller.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json
import os

from src.app import FileOrganizerApp, create_argument_parser, main


class TestFileOrganizerApp(unittest.TestCase):
    """Test cases for FileOrganizerApp class."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_config.json")

        # Create test configuration
        self.test_config = {
            "api": {
                "anthropic_api_key": "test_key",
                "claude_model": "test_model",
                "rate_limit": 10,
            },
            "source": {"directory": self.temp_dir},
            "organization": {
                "base_directory": os.path.join(self.temp_dir, "organized"),
                "mode": "copy",
            },
            "processing": {"batch_size": 5, "enable_ocr": False},
            "logging": {
                "level": "INFO",
                "file": os.path.join(self.temp_dir, "test.log"),
            },
        }

        with open(self.config_file, "w") as f:
            json.dump(self.test_config, f)

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_app_initialization(self):
        """Test app initialization."""
        app = FileOrganizerApp(config_file=self.config_file)
        self.assertFalse(app._is_initialized)

        with patch("src.app.ConfigManager") as mock_config_manager:
            mock_config_manager.return_value.config = self.test_config
            app.initialize()

        self.assertTrue(app._is_initialized)

    @patch("src.app.ConfigManager")
    @patch("src.app.FileSystemAccessor")
    @patch("src.app.FileProcessor")
    @patch("src.app.ClaudeClient")
    @patch("src.app.MetadataExtractor")
    @patch("src.app.OrganizationEngine")
    @patch("src.app.FileManipulator")
    @patch("src.app.ErrorHandler")
    @patch("src.app.SimpleBatchProcessor")
    def test_component_initialization(self, *mocks):
        """Test that all components are initialized correctly."""
        # Unpack mocks
        (
            mock_batch_processor,
            mock_error_handler,
            mock_file_manipulator,
            mock_org_engine,
            mock_metadata_extractor,
            mock_claude_client,
            mock_file_processor,
            mock_file_accessor,
            mock_config_manager,
        ) = mocks

        # Set up config manager
        mock_config_manager.return_value.config = self.test_config

        app = FileOrganizerApp(config_file=self.config_file)
        app.initialize()

        # Verify all components were created
        self.assertIn("file_accessor", app.components)
        self.assertIn("file_processor", app.components)
        self.assertIn("claude_client", app.components)
        self.assertIn("metadata_extractor", app.components)
        self.assertIn("organization_engine", app.components)
        self.assertIn("file_manipulator", app.components)
        self.assertIn("error_handler", app.components)
        self.assertIn("batch_processor", app.components)

    @patch("src.app.ConfigManager")
    def test_run_with_no_files(self, mock_config_manager):
        """Test running the app when no files are found."""
        mock_config_manager.return_value.config = self.test_config

        app = FileOrganizerApp(config_file=self.config_file)

        with patch.object(app, "_scan_files", return_value=[]):
            result = app.run()

        self.assertTrue(result)

    def test_argument_parser(self):
        """Test command-line argument parser."""
        parser = create_argument_parser()

        # Test basic arguments
        args = parser.parse_args(["--config", "test.json", "--dry-run"])
        self.assertEqual(args.config, "test.json")
        self.assertTrue(args.dry_run)

        # Test source and target
        args = parser.parse_args(["--source", "/source", "--target", "/target"])
        self.assertEqual(args.source, "/source")
        self.assertEqual(args.target, "/target")

        # Test mode
        args = parser.parse_args(["--mode", "move"])
        self.assertEqual(args.mode, "move")

    @patch("src.app.FileOrganizerApp")
    @patch("sys.argv", ["app.py", "--dry-run"])
    def test_main_function(self, mock_app_class):
        """Test the main function."""
        mock_app = Mock()
        mock_app.run.return_value = True
        mock_app_class.return_value = mock_app

        with patch("sys.exit") as mock_exit:
            main()
            mock_exit.assert_called_once_with(0)

    def test_dry_run_mode(self):
        """Test dry-run mode configuration."""
        app = FileOrganizerApp(config_file=self.config_file)

        # Initialize app config and components to prevent NoneType error
        app.config = self.test_config.copy()
        app.components = {"file_manipulator": Mock(dry_run=False)}
        app._is_initialized = True

        # Test dry-run override
        with patch.object(app, "_scan_files", return_value=[]), patch.object(
            app, "_cleanup"
        ):

            app.run(dry_run_override=True)
            self.assertTrue(app.config.get("dry_run"))

    def test_source_and_target_override(self):
        """Test source and target directory overrides."""
        app = FileOrganizerApp(config_file=self.config_file)

        new_source = "/new/source"
        new_target = "/new/target"

        # Initialize app
        app.config = self.test_config.copy()
        app.components = {}
        app._is_initialized = True

        with patch.object(app, "_scan_files", return_value=[]), patch.object(
            app, "_cleanup"
        ), patch("src.app.FileSystemAccessor") as mock_accessor, patch(
            "src.app.FileManipulator"
        ) as mock_manipulator:

            app.run(source_override=new_source, target_override=new_target)

            # Verify overrides were applied
            self.assertEqual(app.config["source"]["directory"], new_source)
            self.assertEqual(app.config["organization"]["base_directory"], new_target)

    @patch("src.app.ConfigManager")
    def test_error_handling(self, mock_config_manager):
        """Test error handling in the run method."""
        mock_config_manager.return_value.config = self.test_config

        app = FileOrganizerApp(config_file=self.config_file)

        # Simulate an error during scanning
        with patch.object(app, "_scan_files", side_effect=Exception("Test error")):
            result = app.run()

        self.assertFalse(result)

    def test_file_processing_workflow(self):
        """Test the complete file processing workflow."""
        app = FileOrganizerApp(config_file=self.config_file)

        # Create mock components
        mock_file_accessor = Mock()
        mock_batch_processor = Mock()
        mock_org_engine = Mock()
        mock_file_manipulator = Mock()
        mock_report_generator = Mock()

        app.components = {
            "file_accessor": mock_file_accessor,
            "batch_processor": mock_batch_processor,
            "organization_engine": mock_org_engine,
            "file_manipulator": mock_file_manipulator,
        }
        app.config = self.test_config
        app._is_initialized = True

        # Mock file objects
        mock_files = [Mock(path="/test/file1.txt"), Mock(path="/test/file2.pdf")]
        mock_file_accessor.get_supported_files.return_value = mock_files

        # Mock batch processing results
        mock_batch_processor.process_batch.return_value = [
            ("/test/file1.txt", {"metadata": Mock()}, None),
            ("/test/file2.pdf", {"metadata": Mock()}, None),
        ]

        # Mock organization results
        mock_org_engine.determine_target_location.return_value = (
            "target/path",
            "test_rule",
        )
        mock_file_manipulator.organize_file.return_value = True

        # Mock report generation
        with patch("src.app.ReportGenerator", return_value=mock_report_generator):
            mock_report_generator.generate_summary_report.return_value = "Test report"

            # Run the workflow
            result = app.run()

        self.assertTrue(result)

        # Verify method calls
        mock_file_accessor.get_supported_files.assert_called_once()
        mock_batch_processor.process_batch.assert_called()
        mock_org_engine.determine_target_location.assert_called()
        mock_file_manipulator.organize_file.assert_called()


if __name__ == "__main__":
    unittest.main()
