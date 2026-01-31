"""
Image drawing module
"""
from PIL import Image, ImageDraw
from typing import Tuple
import os

class ImageDrawer:
    """Image drawer class"""

    @staticmethod
    def draw_rectangle(
        image_path: str,
        top_left: Tuple[int, int],
        bottom_right: Tuple[int, int],
        output_path: str,
        color: str = "red",
        width: int = 3
    ) -> bool:
        """
        Draw rectangle on image

        Args:
            image_path: Input image path
            top_left: Top-left corner coordinates (x, y)
            bottom_right: Bottom-right corner coordinates (x, y)
            output_path: Output image path
            color: Color
            width: Line width

        Returns:
            Whether successful
        """
        try:
            # Open image
            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)

            # Draw rectangle
            draw.rectangle(
                [top_left, bottom_right],
                outline=color,
                width=width
            )

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save image
            image.save(output_path)
            return True

        except Exception as e:
            print(f"Failed to draw rectangle: {str(e)}")
            return False
