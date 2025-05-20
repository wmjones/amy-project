#!/usr/bin/env python3
"""
Generate architecture diagram for the OCR + AI pipeline implementation.
Creates a visual representation of the system components and data flow.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_architecture_diagram():
    """Create a comprehensive architecture diagram."""
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define colors
    colors = {
        'input': '#E8F4FD',
        'processing': '#BBE1FA',
        'ai': '#3282B8',
        'storage': '#0F4C75',
        'integration': '#1B262C',
        'output': '#C8E6C9'
    }
    
    # Component dimensions
    box_width = 2.5
    box_height = 1.5
    
    # Layer 1: Input Sources
    y_level = 10
    
    # Dropbox
    dropbox_box = FancyBboxPatch(
        (1, y_level), box_width, box_height,
        boxstyle="round,pad=0.1",
        facecolor=colors['input'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(dropbox_box)
    ax.text(2.25, y_level + 0.75, 'Dropbox\nStorage', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Local Files
    local_box = FancyBboxPatch(
        (5, y_level), box_width, box_height,
        boxstyle="round,pad=0.1",
        facecolor=colors['input'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(local_box)
    ax.text(6.25, y_level + 0.75, 'Local\nFiles', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Scanner
    scanner_box = FancyBboxPatch(
        (9, y_level), box_width, box_height,
        boxstyle="round,pad=0.1",
        facecolor=colors['input'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(scanner_box)
    ax.text(10.25, y_level + 0.75, 'Scanner\nInput', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Layer 2: File Processing
    y_level = 7.5
    
    # File Ingestion
    ingestion_box = FancyBboxPatch(
        (3, y_level), box_width * 1.5, box_height,
        boxstyle="round,pad=0.1",
        facecolor=colors['processing'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(ingestion_box)
    ax.text(4.875, y_level + 0.75, 'File Ingestion\n& Validation', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Document Queue
    queue_box = FancyBboxPatch(
        (7.5, y_level), box_width, box_height,
        boxstyle="round,pad=0.1",
        facecolor=colors['processing'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(queue_box)
    ax.text(8.75, y_level + 0.75, 'Document\nQueue', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Layer 3: OCR Processing
    y_level = 5
    
    # Preprocessing
    preprocess_box = FancyBboxPatch(
        (1, y_level), box_width, box_height,
        boxstyle="round,pad=0.1",
        facecolor=colors['processing'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(preprocess_box)
    ax.text(2.25, y_level + 0.75, 'Image\nPreprocessing', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # OCR Engine
    ocr_box = FancyBboxPatch(
        (4.5, y_level), box_width * 1.5, box_height,
        boxstyle="round,pad=0.1",
        facecolor=colors['processing'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(ocr_box)
    ax.text(6.375, y_level + 0.75, 'OCR Engine\n(Tesseract/Google)', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Cache
    cache_box = FancyBboxPatch(
        (8.5, y_level), box_width, box_height,
        boxstyle="round,pad=0.1",
        facecolor=colors['storage'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(cache_box)
    ax.text(9.75, y_level + 0.75, 'Redis\nCache', 
            ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # Performance Monitor
    monitor_box = FancyBboxPatch(
        (12, y_level), box_width, box_height,
        boxstyle="round,pad=0.1",
        facecolor=colors['integration'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(monitor_box)
    ax.text(13.25, y_level + 0.75, 'Performance\nMonitor', 
            ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # Layer 4: AI Processing
    y_level = 2.5
    
    # Claude API
    claude_box = FancyBboxPatch(
        (2, y_level), box_width * 1.5, box_height,
        boxstyle="round,pad=0.1",
        facecolor=colors['ai'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(claude_box)
    ax.text(3.875, y_level + 0.75, 'Claude AI\nSummarization', 
            ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # Metadata Integration
    metadata_box = FancyBboxPatch(
        (6, y_level), box_width * 1.5, box_height,
        boxstyle="round,pad=0.1",
        facecolor=colors['integration'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(metadata_box)
    ax.text(7.875, y_level + 0.75, 'Metadata\nIntegration', 
            ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # Organization Engine
    org_box = FancyBboxPatch(
        (10, y_level), box_width * 1.5, box_height,
        boxstyle="round,pad=0.1",
        facecolor=colors['integration'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(org_box)
    ax.text(11.875, y_level + 0.75, 'Organization\nEngine', 
            ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # Layer 5: Storage & Output
    y_level = 0.5
    
    # Metadata Storage
    storage_box = FancyBboxPatch(
        (1, y_level), box_width, box_height,
        boxstyle="round,pad=0.1",
        facecolor=colors['storage'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(storage_box)
    ax.text(2.25, y_level + 0.75, 'Metadata\nStorage', 
            ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # File System
    filesystem_box = FancyBboxPatch(
        (4.5, y_level), box_width, box_height,
        boxstyle="round,pad=0.1",
        facecolor=colors['output'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(filesystem_box)
    ax.text(5.75, y_level + 0.75, 'Organized\nFile System', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Reports
    reports_box = FancyBboxPatch(
        (8, y_level), box_width, box_height,
        boxstyle="round,pad=0.1",
        facecolor=colors['output'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(reports_box)
    ax.text(9.25, y_level + 0.75, 'Analysis\nReports', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Dashboard
    dashboard_box = FancyBboxPatch(
        (11.5, y_level), box_width, box_height,
        boxstyle="round,pad=0.1",
        facecolor=colors['output'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(dashboard_box)
    ax.text(12.75, y_level + 0.75, 'Monitoring\nDashboard', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Add connections
    connections = [
        # Input to Ingestion
        ((2.25, 10), (4.875, 8.5)),
        ((6.25, 10), (4.875, 8.5)),
        ((10.25, 10), (4.875, 8.5)),
        
        # Ingestion to Queue
        ((4.875, 7.5), (8.75, 7.5)),
        
        # Queue to Preprocessing
        ((8.75, 7.5), (2.25, 6.5)),
        
        # Preprocessing to OCR
        ((2.25, 5), (4.5, 5.75)),
        
        # OCR to Cache
        ((7.25, 5.75), (8.5, 5.75)),
        
        # OCR to Claude
        ((5.75, 5), (3.875, 4)),
        
        # Claude to Metadata
        ((3.875, 2.5), (6, 3.25)),
        
        # Metadata to Organization
        ((7.875, 2.5), (10, 3.25)),
        
        # To Storage/Output
        ((6, 2.5), (2.25, 2)),
        ((7.875, 2.5), (5.75, 2)),
        ((10, 2.5), (9.25, 2)),
        ((13.25, 5), (12.75, 2)),
    ]
    
    for start, end in connections:
        arrow = ConnectionPatch(
            start, end, "data", "data",
            arrowstyle="->", 
            shrinkA=5, shrinkB=5,
            mutation_scale=20,
            fc="gray",
            linewidth=2
        )
        ax.add_artist(arrow)
    
    # Add title and labels
    ax.text(8, 11.5, 'OCR + AI Pipeline Architecture', 
            ha='center', va='center', fontsize=20, fontweight='bold')
    
    # Add component counts
    ax.text(0.5, 10.75, '400+ Files', ha='left', va='center', fontsize=11)
    ax.text(13.5, 5.75, 'Real-time', ha='center', va='center', fontsize=11)
    ax.text(1.5, 3.25, 'SQLite/JSON', ha='center', va='center', fontsize=10, color='white')
    
    # Add data flow legend
    legend_elements = [
        mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['input'], label='Input Sources'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['processing'], label='Processing'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['ai'], label='AI Services'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['integration'], label='Integration'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['storage'], label='Storage'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['output'], label='Output')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, fontsize=10)
    
    plt.tight_layout()
    
    # Save the diagram
    output_path = 'docs/pipeline_architecture.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Architecture diagram saved to: {output_path}")
    
    # Also create a simplified flow diagram
    create_flow_diagram()
    
def create_flow_diagram():
    """Create a simplified data flow diagram."""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Define positions
    steps = [
        (2, 6.5, "1. File Input\n(Dropbox/Local)"),
        (6, 6.5, "2. Validation\n& Queuing"),
        (10, 6.5, "3. Preprocessing\n(Enhance/Deskew)"),
        (2, 4, "4. OCR\n(Tesseract/Google)"),
        (6, 4, "5. AI Analysis\n(Claude API)"),
        (10, 4, "6. Metadata\nIntegration"),
        (4, 1.5, "7. Organization\nRules"),
        (8, 1.5, "8. Final Output\n(Reports/Files)")
    ]
    
    # Draw boxes and text
    for x, y, text in steps:
        box = FancyBboxPatch(
            (x-1.2, y-0.6), 2.4, 1.2,
            boxstyle="round,pad=0.1",
            facecolor='lightblue',
            edgecolor='darkblue',
            linewidth=2
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Draw arrows
    arrows = [
        ((3.2, 6.5), (4.8, 6.5)),
        ((7.2, 6.5), (8.8, 6.5)),
        ((10, 5.9), (10, 4.6)),
        ((8.8, 4), (7.2, 4)),
        ((4.8, 4), (3.2, 4)),
        ((2, 3.4), (2, 2.1)),
        ((6, 3.4), (4, 2.1)),
        ((10, 3.4), (8, 2.1)),
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(
            start, end, "data", "data",
            arrowstyle="->", 
            mutation_scale=20,
            fc="darkblue",
            linewidth=2
        )
        ax.add_artist(arrow)
    
    # Add title
    ax.text(6, 7.5, 'Hansman Syracuse Collection Processing Flow', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Add metrics
    ax.text(0.5, 0.5, 'Processing: 400+ files | Time: 4-7 hours | Cost: $6.60', 
            ha='left', va='center', fontsize=12, style='italic')
    
    plt.tight_layout()
    
    # Save the flow diagram
    output_path = 'docs/processing_flow.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Flow diagram saved to: {output_path}")

if __name__ == "__main__":
    create_architecture_diagram()