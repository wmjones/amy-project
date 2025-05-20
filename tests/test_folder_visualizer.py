"""
Unit tests for the folder visualizer module.
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from src.utils.folder_visualizer import FolderVisualizer


class TestFolderVisualizer:
    """Test the FolderVisualizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.visualizer = FolderVisualizer()
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create a test directory structure
        self.create_test_structure()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def create_test_structure(self):
        """Create a test directory structure."""
        # Create directories
        (self.temp_dir / "docs").mkdir()
        (self.temp_dir / "docs" / "invoices").mkdir()
        (self.temp_dir / "docs" / "receipts").mkdir()
        (self.temp_dir / "images").mkdir()
        (self.temp_dir / "images" / "photos").mkdir()
        
        # Create files
        (self.temp_dir / "readme.txt").write_text("Test readme")
        (self.temp_dir / "docs" / "summary.pdf").write_text("Test summary")
        (self.temp_dir / "docs" / "invoices" / "inv001.pdf").write_text("Invoice 1")
        (self.temp_dir / "docs" / "invoices" / "inv002.pdf").write_text("Invoice 2")
        (self.temp_dir / "docs" / "receipts" / "receipt1.jpg").write_text("Receipt 1")
        (self.temp_dir / "images" / "cover.png").write_text("Cover image")
        (self.temp_dir / "images" / "photos" / "photo1.jpg").write_text("Photo 1")
        (self.temp_dir / "images" / "photos" / "photo2.jpg").write_text("Photo 2")
    
    def test_visualize_directory_tree(self):
        """Test basic directory tree visualization."""
        tree = self.visualizer.visualize_directory_tree(self.temp_dir, max_depth=2)
        
        # Check that main directories are shown
        assert "docs" in tree
        assert "images" in tree
        assert "readme.txt" not in tree  # Files not shown by default
        
        # Check tree structure characters
        assert "├" in tree or "└" in tree
        assert "─" in tree
    
    def test_visualize_with_files(self):
        """Test directory tree with files shown."""
        tree = self.visualizer.visualize_directory_tree(
            self.temp_dir, max_depth=3, show_files=True
        )
        
        # Check that files are now shown
        assert "readme.txt" in tree
        assert "summary.pdf" in tree
        assert "inv001.pdf" in tree
    
    def test_visualize_with_file_filter(self):
        """Test directory tree with file extension filter."""
        tree = self.visualizer.visualize_directory_tree(
            self.temp_dir, 
            max_depth=3, 
            show_files=True,
            file_extensions={'.pdf'}
        )
        
        # Check that only PDF files are shown
        assert "summary.pdf" in tree
        assert "inv001.pdf" in tree
        assert "receipt1.jpg" not in tree
        assert "photo1.jpg" not in tree
    
    def test_max_depth_limit(self):
        """Test max depth limiting."""
        tree = self.visualizer.visualize_directory_tree(
            self.temp_dir, max_depth=1, show_files=True
        )
        
        # Check that subdirectories are not expanded
        assert "docs" in tree
        assert "images" in tree
        assert "invoices" not in tree  # Should not show subdirectories
        assert "photos" not in tree
    
    def test_compare_structures(self):
        """Test directory structure comparison."""
        # Create a second directory structure
        other_dir = self.temp_dir / "other"
        other_dir.mkdir()
        (other_dir / "docs").mkdir()
        (other_dir / "docs" / "new_file.pdf").write_text("New file")
        (other_dir / "archive").mkdir()
        
        comparison = self.visualizer.compare_structures(
            self.temp_dir, other_dir, max_depth=2
        )
        
        assert 'before' in comparison
        assert 'after' in comparison
        assert 'diff' in comparison
        
        # Check that the diff shows changes
        assert "Added files:" in comparison['diff'] or "Removed files:" in comparison['diff']
    
    def test_generate_size_map(self):
        """Test size map generation."""
        size_map = self.visualizer.generate_size_map(self.temp_dir, max_depth=2)
        
        # Check that directories are in the map
        assert "." in size_map  # Root directory
        assert "docs" in size_map
        assert "images" in size_map
        
        # Check that size information is included
        assert size_map["."]["total_size"] > 0
        assert size_map["."]["file_count"] >= 0
        assert size_map["."]["subdir_count"] >= 2
        assert "size_formatted" in size_map["."]
    
    def test_format_size(self):
        """Test size formatting."""
        test_cases = [
            (512, "512.0 B"),
            (1024, "1.0 KB"),
            (1024 * 1024, "1.0 MB"),
            (5.5 * 1024 * 1024, "5.5 MB"),
            (1024 * 1024 * 1024, "1.0 GB")
        ]
        
        for size_bytes, expected in test_cases:
            formatted = self.visualizer._format_size(int(size_bytes))
            assert formatted == expected
    
    def test_create_summary_visualization(self):
        """Test summary visualization creation."""
        summary = self.visualizer.create_summary_visualization(self.temp_dir)
        
        # Check that summary contains expected information
        assert "Organization Summary" in summary
        assert "Total files organized:" in summary
        assert "Directory breakdown:" in summary
        
        # Check that directories are listed
        assert "docs/" in summary
        assert "images/" in summary
    
    def test_export_structure_as_json(self):
        """Test JSON export of directory structure."""
        structure = self.visualizer.export_structure_as_json(
            self.temp_dir, max_depth=2
        )
        
        # Check root structure
        assert structure['name'] == self.temp_dir.name
        assert structure['type'] == 'directory'
        assert 'children' in structure
        
        # Check children
        children_names = [child['name'] for child in structure['children']]
        assert 'docs' in children_names
        assert 'images' in children_names
        
        # Find docs directory
        docs_dir = next(child for child in structure['children'] 
                       if child['name'] == 'docs')
        assert docs_dir['type'] == 'directory'
        assert 'children' in docs_dir
        
        # Check that files have size and extension
        readme = next((child for child in structure['children'] 
                      if child['name'] == 'readme.txt'), None)
        if readme:
            assert readme['type'] == 'file'
            assert 'size' in readme
            assert readme['extension'] == '.txt'
    
    def test_permission_error_handling(self):
        """Test handling of permission errors."""
        # This test is platform-specific and may not work on all systems
        # Create a directory with no read permissions
        no_access_dir = self.temp_dir / "no_access"
        no_access_dir.mkdir()
        
        try:
            import os
            os.chmod(no_access_dir, 0o000)
            
            # Should handle permission error gracefully
            tree = self.visualizer.visualize_directory_tree(self.temp_dir)
            # Tree should still be generated, possibly with error indicator
            assert str(self.temp_dir) in tree
            
        finally:
            # Restore permissions for cleanup
            os.chmod(no_access_dir, 0o755)


if __name__ == "__main__":
    pytest.main([__file__])