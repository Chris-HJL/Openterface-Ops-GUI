"""
图像绘制模块
"""
from PIL import Image, ImageDraw
from typing import Tuple
import os

class ImageDrawer:
    """图像绘制器类"""

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
        在图像上绘制矩形框

        Args:
            image_path: 输入图像路径
            top_left: 左上角坐标 (x, y)
            bottom_right: 右下角坐标 (x, y)
            output_path: 输出图像路径
            color: 颜色
            width: 线宽

        Returns:
            是否成功
        """
        try:
            # 打开图像
            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)

            # 绘制矩形
            draw.rectangle(
                [top_left, bottom_right],
                outline=color,
                width=width
            )

            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 保存图像
            image.save(output_path)
            return True

        except Exception as e:
            print(f"Failed to draw rectangle: {str(e)}")
            return False
