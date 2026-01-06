"""
图像编码模块
"""
import base64
from typing import Optional

class ImageEncoder:
    """图像编码器类"""

    @staticmethod
    def encode_to_base64(image_path: str) -> str:
        """
        将图像文件编码为Base64字符串

        Args:
            image_path: 图像文件路径

        Returns:
            Base64编码的字符串
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            return f"Image encoding error: {str(e)}"

    @staticmethod
    def decode_from_base64(base64_str: str, output_path: str) -> bool:
        """
        将Base64字符串解码并保存为图像文件

        Args:
            base64_str: Base64编码的字符串
            output_path: 输出文件路径

        Returns:
            是否成功
        """
        try:
            with open(output_path, "wb") as f:
                f.write(base64.b64decode(base64_str))
            return True
        except Exception as e:
            print(f"Failed to decode image: {str(e)}")
            return False
