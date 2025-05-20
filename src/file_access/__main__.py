"""
Command line interface for file access module.
"""

import click
import logging
from pathlib import Path
from .local_accessor import FileSystemAccessor
from ..utils.file_utils import human_readable_size

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@click.command()
@click.argument("directory", type=click.Path(exists=True))
@click.option("--recursive/--no-recursive", default=True, help="Scan subdirectories")
@click.option("--supported-only", is_flag=True, help="Show only supported files")
@click.option("--stats", is_flag=True, help="Show directory statistics")
def scan(directory: str, recursive: bool, supported_only: bool, stats: bool):
    """Scan a directory and list files."""
    accessor = FileSystemAccessor(directory)

    if stats:
        # Show statistics
        dir_stats = accessor.get_directory_stats()
        click.echo(f"\nDirectory Statistics for: {directory}")
        click.echo(f"Total files: {dir_stats['total_files']}")
        click.echo(f"Supported files: {dir_stats['supported_files']}")
        click.echo(f"Total size: {human_readable_size(dir_stats['total_size'])}")

        click.echo("\nFiles by extension:")
        for ext, count in sorted(dir_stats["by_extension"].items()):
            click.echo(f"  {ext}: {count}")

    else:
        # List files
        if supported_only:
            files = accessor.get_supported_files(recursive=recursive)
        else:
            files = accessor.scan_directory(recursive=recursive)

        click.echo(f"\nFiles in {directory}:")
        for file in files:
            size = human_readable_size(file.size)
            support = "[SUPPORTED]" if file.is_supported else "[unsupported]"
            click.echo(f"  {file.name} - {size} {support}")

        click.echo(f"\nTotal: {len(files)} files")


if __name__ == "__main__":
    scan()
