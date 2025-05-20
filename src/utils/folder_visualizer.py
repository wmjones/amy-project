"""
Folder structure visualization utilities for report generation.
Creates visual representations of directory hierarchies.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import defaultdict


class FolderVisualizer:
    """Create visual representations of folder structures."""
    
    def __init__(self):
        """Initialize folder visualizer."""
        self.tree_chars = {
            'vertical': '│',
            'horizontal': '─',
            'branch': '├',
            'last_branch': '└'
        }
    
    def visualize_directory_tree(self, root_path: Path, 
                               max_depth: int = 3,
                               show_files: bool = False,
                               file_extensions: Optional[Set[str]] = None) -> str:
        """
        Create a text-based tree visualization of directory structure.
        
        Args:
            root_path: Root directory to visualize
            max_depth: Maximum depth to traverse
            show_files: Whether to show files in addition to directories
            file_extensions: If provided, only show files with these extensions
            
        Returns:
            String representation of directory tree
        """
        lines = [str(root_path)]
        self._build_tree(root_path, lines, "", max_depth, show_files, file_extensions)
        return '\n'.join(lines)
    
    def _build_tree(self, path: Path, lines: List[str], prefix: str,
                   max_depth: int, show_files: bool = False,
                   file_extensions: Optional[Set[str]] = None, 
                   current_depth: int = 0):
        """Recursively build tree visualization."""
        if current_depth >= max_depth:
            return
        
        try:
            entries = list(path.iterdir())
            entries.sort(key=lambda p: (not p.is_dir(), p.name.lower()))
            
            for i, entry in enumerate(entries):
                is_last = i == len(entries) - 1
                
                # Skip files if not showing them
                if entry.is_file() and not show_files:
                    continue
                
                # Filter by extension if specified
                if entry.is_file() and file_extensions:
                    if entry.suffix.lower() not in file_extensions:
                        continue
                
                # Create the branch line
                if is_last:
                    connector = self.tree_chars['last_branch']
                    extension = "    "
                else:
                    connector = self.tree_chars['branch']
                    extension = self.tree_chars['vertical'] + "   "
                
                # Add entry to tree
                lines.append(f"{prefix}{connector}{self.tree_chars['horizontal']*2} {entry.name}")
                
                # Recurse for directories
                if entry.is_dir():
                    self._build_tree(entry, lines, prefix + extension,
                                   max_depth, show_files, file_extensions,
                                   current_depth + 1)
        except PermissionError:
            lines.append(f"{prefix}{self.tree_chars['branch']}{self.tree_chars['horizontal']*2} [Permission Denied]")
    
    def compare_structures(self, before_path: Path, after_path: Path,
                          max_depth: int = 3) -> Dict[str, str]:
        """
        Compare two directory structures and highlight differences.
        
        Args:
            before_path: Original directory structure
            after_path: Directory structure after organization
            max_depth: Maximum depth to compare
            
        Returns:
            Dictionary with 'before', 'after', and 'diff' visualizations
        """
        before_tree = self.visualize_directory_tree(before_path, max_depth, show_files=True)
        after_tree = self.visualize_directory_tree(after_path, max_depth, show_files=True)
        
        # Create a simple diff visualization
        before_files = self._get_all_files(before_path, max_depth)
        after_files = self._get_all_files(after_path, max_depth)
        
        added = after_files - before_files
        removed = before_files - after_files
        
        diff_lines = []
        if added:
            diff_lines.append("Added files:")
            for file in sorted(added):
                diff_lines.append(f"  + {file}")
        
        if removed:
            diff_lines.append("\nRemoved files:")
            for file in sorted(removed):
                diff_lines.append(f"  - {file}")
        
        return {
            'before': before_tree,
            'after': after_tree,
            'diff': '\n'.join(diff_lines) if diff_lines else "No differences found"
        }
    
    def _get_all_files(self, root_path: Path, max_depth: int) -> Set[str]:
        """Get all file paths relative to root."""
        files = set()
        
        def _scan(path: Path, depth: int = 0):
            if depth >= max_depth:
                return
            
            try:
                for entry in path.iterdir():
                    if entry.is_file():
                        files.add(str(entry.relative_to(root_path)))
                    elif entry.is_dir():
                        _scan(entry, depth + 1)
            except PermissionError:
                pass
        
        _scan(root_path)
        return files
    
    def generate_size_map(self, root_path: Path, max_depth: int = 3) -> Dict[str, Dict]:
        """
        Generate a size map of directory structure.
        
        Args:
            root_path: Root directory to analyze
            max_depth: Maximum depth to traverse
            
        Returns:
            Dictionary with size information for each directory
        """
        size_map = {}
        
        def _calculate_size(path: Path, depth: int = 0) -> int:
            if depth > max_depth:
                return 0
            
            total_size = 0
            file_count = 0
            subdir_count = 0
            
            try:
                for entry in path.iterdir():
                    if entry.is_file():
                        total_size += entry.stat().st_size
                        file_count += 1
                    elif entry.is_dir():
                        subdir_count += 1
                        subdir_size = _calculate_size(entry, depth + 1)
                        total_size += subdir_size
            except PermissionError:
                pass
            
            # Store information for this directory
            rel_path = str(path.relative_to(root_path)) if path != root_path else "."
            size_map[rel_path] = {
                'total_size': total_size,
                'file_count': file_count,
                'subdir_count': subdir_count,
                'size_formatted': self._format_size(total_size)
            }
            
            return total_size
        
        _calculate_size(root_path)
        return size_map
    
    def _format_size(self, size_bytes: int) -> str:
        """Format byte size as human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"
    
    def create_summary_visualization(self, organized_path: Path) -> str:
        """
        Create a summary visualization of the organized structure.
        
        Args:
            organized_path: Path to organized directory
            
        Returns:
            Text visualization of the organization summary
        """
        # Count files by type in each top-level directory
        type_distribution = defaultdict(lambda: defaultdict(int))
        total_files = 0
        
        try:
            for top_dir in organized_path.iterdir():
                if top_dir.is_dir():
                    for root, _, files in os.walk(top_dir):
                        for file in files:
                            file_path = Path(root) / file
                            file_type = file_path.suffix.lower()
                            type_distribution[top_dir.name][file_type] += 1
                            total_files += 1
        except Exception as e:
            return f"Error creating visualization: {e}"
        
        # Build summary
        lines = [
            "Organization Summary",
            "=" * 20,
            f"Total files organized: {total_files}",
            "",
            "Directory breakdown:"
        ]
        
        for dir_name, types in sorted(type_distribution.items()):
            dir_total = sum(types.values())
            lines.append(f"\n{dir_name}/ ({dir_total} files)")
            
            # Show top 5 file types
            sorted_types = sorted(types.items(), key=lambda x: x[1], reverse=True)[:5]
            for file_type, count in sorted_types:
                type_name = file_type if file_type else "(no extension)"
                lines.append(f"  {type_name:15s} {count:5d} files")
            
            if len(types) > 5:
                remaining = sum(count for _, count in sorted_types[5:])
                lines.append(f"  {'(others)':15s} {remaining:5d} files")
        
        return '\n'.join(lines)
    
    def export_structure_as_json(self, root_path: Path, max_depth: int = 3) -> Dict:
        """
        Export directory structure as JSON for further processing.
        
        Args:
            root_path: Root directory to export
            max_depth: Maximum depth to traverse
            
        Returns:
            Dictionary representing the directory structure
        """
        def _build_structure(path: Path, depth: int = 0) -> Dict:
            if depth > max_depth:
                return None
            
            structure = {
                'name': path.name,
                'type': 'directory' if path.is_dir() else 'file',
                'path': str(path)
            }
            
            if path.is_file():
                structure['size'] = path.stat().st_size
                structure['extension'] = path.suffix
            else:
                children = []
                try:
                    for entry in sorted(path.iterdir()):
                        child = _build_structure(entry, depth + 1)
                        if child:
                            children.append(child)
                except PermissionError:
                    structure['error'] = 'Permission denied'
                
                if children:
                    structure['children'] = children
            
            return structure
        
        return _build_structure(root_path)