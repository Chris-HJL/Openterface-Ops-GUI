"""
Image encoding module
"""
import base64
from typing import Optional

class ImageEncoder:
    """Image encoder class"""

    @staticmethod
    def encode_to_base64(image_path: str) -> str:
        """
        Encode image file to Base64 string

        Args:
            image_path: Image file path

        Returns:
            Base64 encoded string
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            return f"Image encoding error: {str(e)}"

    @staticmethod
    def decode_from_base64(base64_str: str, output_path: str) -> bool:
        """
        Decode Base64 string and save as image file

        Args:
            base64_str: Base64 encoded string
            output_path: Output file path

        Returns:
            Whether successful
        """
        try:
            with open(output_path, "wb") as f:
                f.write(base64.b64decode(base64_str))
            return True
        except Exception as e:
            print(f"Failed to decode image: {str(e)}")
            return False
