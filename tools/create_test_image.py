"""
创建测试图像
"""
from PIL import Image, ImageDraw
import os

# 创建测试图像目录
os.makedirs("./images", exist_ok=True)

# 创建测试图像
width, height = 1920, 1080
img = Image.new('RGB', (width, height), color='lightblue')
draw = ImageDraw.Draw(img)

# 绘制一些测试内容
draw.rectangle([100, 100, 300, 200], fill='red', outline='darkred')
draw.text((120, 130), "Test Button", fill='white')

draw.rectangle([400, 100, 600, 200], fill='green', outline='darkgreen')
draw.text((420, 130), "Settings", fill='white')

draw.ellipse([700, 100, 900, 300], fill='orange', outline='darkorange')
draw.text((730, 200), "Circle", fill='white')

# 保存图像
img.save("./images/test_screen.jpg", "JPEG", quality=90)
print(f"✅ 测试图像已创建：./images/test_screen.jpg ({width}x{height})")
