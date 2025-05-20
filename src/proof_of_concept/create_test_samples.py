#!/usr/bin/env python3
"""Create test sample images for the Hansman OCR proof of concept."""

import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path


def create_sample_document(text: str, filename: str, add_noise: bool = True):
    """Create a sample document image with text."""
    # Create image
    img_width, img_height = 800, 1000
    img = Image.new("RGB", (img_width, img_height), color="white")
    draw = ImageDraw.Draw(img)

    # Try to use a decent font, fallback to default if not available
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 20
        )
    except:
        font = ImageFont.load_default()

    # Add text with word wrapping
    margin = 50
    y_position = margin
    line_height = 30

    # Simple word wrapping
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        current_line.append(word)
        test_line = " ".join(current_line)
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] > img_width - 2 * margin:
            if len(current_line) > 1:
                current_line.pop()
                lines.append(" ".join(current_line))
                current_line = [word]
            else:
                lines.append(word)
                current_line = []
    if current_line:
        lines.append(" ".join(current_line))

    # Draw the text
    for line in lines:
        draw.text((margin, y_position), line, fill="black", font=font)
        y_position += line_height

    # Add some noise/aging effects if requested
    if add_noise:
        # Convert to numpy array
        img_array = np.array(img)

        # Add some gaussian noise
        noise = np.random.normal(0, 5, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

        # Add some spots/stains
        for _ in range(10):
            x = np.random.randint(0, img_width)
            y = np.random.randint(0, img_height)
            radius = np.random.randint(5, 15)
            y_indices, x_indices = np.ogrid[:img_height, :img_width]
            mask = (x_indices - x) ** 2 + (y_indices - y) ** 2 <= radius**2
            intensity = np.random.randint(200, 240)
            img_array[mask] = intensity

        img = Image.fromarray(img_array)

    # Save the image
    img.save(filename)
    print(f"Created: {filename}")


def create_test_samples():
    """Create several test sample documents."""
    samples_dir = Path("/workspaces/amy-project/data/hansman_samples")
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Sample texts representing different types of historical documents
    samples = [
        {
            "filename": "letter_1923.png",
            "text": """Syracuse, New York
June 15, 1923

Dear Mr. Thompson,

I am writing to inform you of the recent developments at the Syracuse Manufacturing Company.
The board has approved the expansion of our eastern facility, which will increase production
capacity by forty percent.

As discussed in our previous correspondence, the market conditions remain favorable for our
textile products. The new machinery from Cleveland has been installed and is operating at
full capacity.

Please send my regards to Mrs. Thompson and the children.

Sincerely yours,
John H. Hansman
President, Syracuse Manufacturing Co.""",
        },
        {
            "filename": "inventory_1925.png",
            "text": """Syracuse Manufacturing Company
Inventory Report - March 1925

Warehouse A:
- Cotton bolts: 2,450 units
- Wool fabric: 1,875 yards
- Silk material: 325 yards
- Thread spools: 15,000 units

Warehouse B:
- Finished shirts: 3,200 units
- Trousers: 2,100 units
- Overcoats: 850 units
- Ladies' dresses: 1,400 units

Total inventory value: $45,670.00

Prepared by: William R. Stevens
Inventory Manager""",
        },
        {
            "filename": "meeting_minutes_1924.png",
            "text": """Syracuse Chamber of Commerce
Meeting Minutes
September 12, 1924

Present: J. Hansman, R. Collins, M. Douglas, T. Wilson, S. Perry

The meeting was called to order at 2:00 PM by President Hansman.

Motion to approve August minutes: Approved unanimously.

New Business:
1. Downtown improvement project - Budget approved for $12,000
2. Annual charity gala - Scheduled for November 15th
3. Railroad expansion proposal - Tabled for further discussion

The meeting was adjourned at 4:30 PM.

Respectfully submitted,
Margaret Douglas, Secretary""",
        },
        {
            "filename": "financial_report_1926.png",
            "text": """Syracuse Manufacturing Company
Financial Statement - Year Ending December 31, 1926

Revenue:
Product sales: $287,450.00
Service contracts: $45,200.00
Total Revenue: $332,650.00

Expenses:
Raw materials: $124,300.00
Labor costs: $87,650.00
Overhead: $34,200.00
Total Expenses: $246,150.00

Net Profit: $86,500.00

This represents a 15% increase over the previous year.

Certified by: Charles M. Whitman, CPA""",
        },
        {
            "filename": "newspaper_clipping_1927.png",
            "text": """THE SYRACUSE HERALD
March 3, 1927

LOCAL MANUFACTURER EXPANDS OPERATIONS

The Syracuse Manufacturing Company, under the leadership of John H. Hansman,
announced yesterday the opening of a new production facility on the city's
west side. The expansion will create 200 new jobs for local workers.

"This investment demonstrates our commitment to the Syracuse community,"
said Mr. Hansman at the groundbreaking ceremony. "We believe in the future
of American manufacturing."

The new facility will focus on producing high-quality textiles for the
growing consumer market. Operations are expected to begin by summer.""",
        },
    ]

    for sample in samples:
        filepath = samples_dir / sample["filename"]
        create_sample_document(sample["text"], str(filepath), add_noise=True)

    print(f"\nCreated {len(samples)} test samples in {samples_dir}")


if __name__ == "__main__":
    create_test_samples()
