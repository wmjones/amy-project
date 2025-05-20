"""
Document segmentation and structural analysis.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np
import cv2
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class DocumentRegionType(Enum):
    """Types of document regions."""
    HEADER = "header"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE = "figure"
    FOOTER = "footer"
    CAPTION = "caption"
    LIST = "list"
    UNKNOWN = "unknown"


@dataclass
class DocumentRegion:
    """Represents a segmented region of a document."""
    region_type: DocumentRegionType
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    text_density: float = 0.0
    is_vertical: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def area(self) -> int:
        """Calculate region area."""
        return self.bbox[2] * self.bbox[3]
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get region center point."""
        return (
            self.bbox[0] + self.bbox[2] // 2,
            self.bbox[1] + self.bbox[3] // 2
        )


class DocumentSegmenter:
    """Analyzes document structure and segments into regions."""
    
    def __init__(
        self,
        min_region_area: int = 1000,
        text_density_threshold: float = 0.3,
        line_detection_threshold: int = 150
    ):
        """Initialize the document segmenter.
        
        Args:
            min_region_area: Minimum area for a valid region
            text_density_threshold: Threshold for considering region as text
            line_detection_threshold: Threshold for line detection
        """
        self.min_region_area = min_region_area
        self.text_density_threshold = text_density_threshold
        self.line_detection_threshold = line_detection_threshold
        
    def segment_document(self, image: np.ndarray) -> List[DocumentRegion]:
        """Segment document into structural regions.
        
        Args:
            image: Document image (can be color or grayscale)
            
        Returns:
            List of detected document regions
        """
        logger.info("Segmenting document structure")
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Detect basic regions
        regions = []
        
        # Find contours of text blocks
        text_regions = self._find_text_regions(gray)
        regions.extend(text_regions)
        
        # Detect tables
        table_regions = self._detect_tables(gray)
        regions.extend(table_regions)
        
        # Classify regions by type
        classified_regions = self._classify_regions(regions, gray)
        
        # Sort regions by position (top to bottom, left to right)
        classified_regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        
        return classified_regions
    
    def _find_text_regions(self, image: np.ndarray) -> List[DocumentRegion]:
        """Find regions containing text blocks.
        
        Args:
            image: Grayscale document image
            
        Returns:
            List of text regions
        """
        logger.debug("Finding text regions")
        
        # Apply morphological operations to merge text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilated = cv2.dilate(image, kernel, iterations=2)
        
        # Apply threshold
        _, binary = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            if area < self.min_region_area:
                continue
                
            # Calculate text density
            roi = image[y:y+h, x:x+w]
            text_density = self._calculate_text_density(roi)
            
            region = DocumentRegion(
                region_type=DocumentRegionType.UNKNOWN,
                bbox=(x, y, w, h),
                confidence=0.8,
                text_density=text_density
            )
            regions.append(region)
            
        return regions
    
    def _detect_tables(self, image: np.ndarray) -> List[DocumentRegion]:
        """Detect table regions using line detection.
        
        Args:
            image: Grayscale document image
            
        Returns:
            List of table regions
        """
        logger.debug("Detecting tables")
        
        # Apply edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Detect horizontal and vertical lines
        horizontal_lines = self._detect_lines(edges, horizontal=True)
        vertical_lines = self._detect_lines(edges, horizontal=False)
        
        # Find intersections and table boundaries
        table_regions = []
        intersections = self._find_line_intersections(horizontal_lines, vertical_lines)
        
        if len(intersections) > 4:  # Minimum for a table
            # Group intersections into table regions
            tables = self._group_intersections_to_tables(intersections)
            
            for table_bbox in tables:
                region = DocumentRegion(
                    region_type=DocumentRegionType.TABLE,
                    bbox=table_bbox,
                    confidence=0.9,
                    metadata={'line_count': len(intersections)}
                )
                table_regions.append(region)
                
        return table_regions
    
    def _detect_lines(self, edges: np.ndarray, horizontal: bool = True) -> List[Tuple[int, int, int, int]]:
        """Detect horizontal or vertical lines.
        
        Args:
            edges: Edge-detected image
            horizontal: Whether to detect horizontal lines
            
        Returns:
            List of lines as (x1, y1, x2, y2) tuples
        """
        # Create appropriate kernel
        if horizontal:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
        # Apply morphology
        lines_img = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Use HoughLinesP for line detection
        lines = cv2.HoughLinesP(
            lines_img,
            rho=1,
            theta=np.pi/180,
            threshold=self.line_detection_threshold,
            minLineLength=100,
            maxLineGap=10
        )
        
        if lines is None:
            return []
            
        # Convert to list of tuples
        line_list = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_list.append((x1, y1, x2, y2))
            
        return line_list
    
    def _find_line_intersections(
        self,
        horizontal_lines: List[Tuple[int, int, int, int]],
        vertical_lines: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int]]:
        """Find intersections between horizontal and vertical lines.
        
        Args:
            horizontal_lines: List of horizontal lines
            vertical_lines: List of vertical lines
            
        Returns:
            List of intersection points
        """
        intersections = []
        
        for h_line in horizontal_lines:
            hx1, hy1, hx2, hy2 = h_line
            
            for v_line in vertical_lines:
                vx1, vy1, vx2, vy2 = v_line
                
                # Check if lines intersect
                if (min(hx1, hx2) <= vx1 <= max(hx1, hx2) and
                    min(vy1, vy2) <= hy1 <= max(vy1, vy2)):
                    intersections.append((vx1, hy1))
                    
        return intersections
    
    def _group_intersections_to_tables(
        self,
        intersections: List[Tuple[int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """Group intersection points into table bounding boxes.
        
        Args:
            intersections: List of intersection points
            
        Returns:
            List of table bounding boxes
        """
        if not intersections:
            return []
            
        # Simple clustering - find bounding box of nearby intersections
        tables = []
        
        # Sort intersections by position
        intersections.sort()
        
        # Find clusters of intersections
        clusters = []
        current_cluster = [intersections[0]]
        
        for i in range(1, len(intersections)):
            x, y = intersections[i]
            last_x, last_y = current_cluster[-1]
            
            # If points are close, add to current cluster
            if abs(x - last_x) < 100 and abs(y - last_y) < 100:
                current_cluster.append((x, y))
            else:
                if len(current_cluster) > 4:  # Minimum for table
                    clusters.append(current_cluster)
                current_cluster = [(x, y)]
                
        if len(current_cluster) > 4:
            clusters.append(current_cluster)
            
        # Convert clusters to bounding boxes
        for cluster in clusters:
            xs = [p[0] for p in cluster]
            ys = [p[1] for p in cluster]
            
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            tables.append((min_x, min_y, max_x - min_x, max_y - min_y))
            
        return tables
    
    def _calculate_text_density(self, region: np.ndarray) -> float:
        """Calculate text density in a region.
        
        Args:
            region: Region of interest
            
        Returns:
            Text density score (0-1)
        """
        # Apply threshold
        _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Calculate ratio of foreground pixels
        total_pixels = region.size
        text_pixels = cv2.countNonZero(binary)
        
        return text_pixels / total_pixels if total_pixels > 0 else 0
    
    def _classify_regions(
        self,
        regions: List[DocumentRegion],
        image: np.ndarray
    ) -> List[DocumentRegion]:
        """Classify regions by type based on characteristics.
        
        Args:
            regions: List of regions to classify
            image: Document image
            
        Returns:
            List of classified regions
        """
        image_height, image_width = image.shape[:2]
        
        for i, region in enumerate(regions):
            # Skip already classified regions
            if region.region_type != DocumentRegionType.UNKNOWN:
                continue
                
            x, y, w, h = region.bbox
            
            # Position-based classification
            if y < image_height * 0.1:
                region.region_type = DocumentRegionType.HEADER
            elif y > image_height * 0.9:
                region.region_type = DocumentRegionType.FOOTER
            elif region.text_density < self.text_density_threshold:
                region.region_type = DocumentRegionType.FIGURE
            elif h < 50 and w > image_width * 0.8:
                region.region_type = DocumentRegionType.CAPTION
            elif h > w * 2:  # Tall narrow region
                region.region_type = DocumentRegionType.LIST
                region.is_vertical = True
            else:
                region.region_type = DocumentRegionType.PARAGRAPH
                
        return regions
    
    def visualize_segmentation(
        self,
        image: np.ndarray,
        regions: List[DocumentRegion]
    ) -> np.ndarray:
        """Visualize segmentation results.
        
        Args:
            image: Original document image
            regions: List of document regions
            
        Returns:
            Image with visualized regions
        """
        # Create color map for region types
        color_map = {
            DocumentRegionType.HEADER: (255, 0, 0),      # Red
            DocumentRegionType.PARAGRAPH: (0, 255, 0),   # Green
            DocumentRegionType.TABLE: (0, 0, 255),       # Blue
            DocumentRegionType.FIGURE: (255, 255, 0),    # Yellow
            DocumentRegionType.FOOTER: (255, 0, 255),    # Magenta
            DocumentRegionType.CAPTION: (0, 255, 255),   # Cyan
            DocumentRegionType.LIST: (128, 128, 0),      # Olive
            DocumentRegionType.UNKNOWN: (128, 128, 128)  # Gray
        }
        
        # Convert to color if needed
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = image.copy()
            
        # Draw regions
        for region in regions:
            x, y, w, h = region.bbox
            color = color_map.get(region.region_type, (0, 0, 0))
            
            # Draw rectangle
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
            
            # Add label
            label = f"{region.region_type.value} ({region.confidence:.2f})"
            cv2.putText(
                vis_image,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
            
        return vis_image