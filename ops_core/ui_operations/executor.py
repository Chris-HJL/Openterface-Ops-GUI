"""
命令执行模块
"""
from typing import Tuple
from PIL import Image
import os
from config import Config
from .parser import ResponseParser
from .ui_ins_client import UIInsClient
from ..image.drawer import ImageDrawer
from ..image_server.client import ImageServerClient

class CommandExecutor:
    """命令执行器类"""

    def __init__(self, ui_ins_api_url: str = None, ui_ins_model: str = None):
        """
        初始化命令执行器

        Args:
            ui_ins_api_url: UI-Ins API URL
            ui_ins_model: UI-Ins模型名称
        """
        self.ui_ins_client = UIInsClient(ui_ins_api_url, ui_ins_model)
        self.image_server_client = ImageServerClient()
        self.parser = ResponseParser()

    def process_ui_element_request(
        self,
        image_path: str,
        instruction: str,
        action: str,
        element: str = None,
        input_content: str = None,
        key_content: str = None
    ) -> Tuple[bool, str]:
        """
        处理UI元素请求

        Args:
            image_path: 图像文件路径
            instruction: 指令文本
            action: 动作类型
            element: 元素名称
            input_content: 输入内容
            key_content: 按键内容

        Returns:
            元组 (是否成功, 输出图像路径或错误消息)
        """
        try:
            if action in ["Click", "Double Click", "Right Click"] and element:
                # 调用UI-Model API获取坐标
                ui_model_response = self.ui_ins_client.call_api(image_path, element)

                # 解析坐标
                point_x, point_y = self.parser.parse_coordinates(ui_model_response)

                if point_x != -1:
                    # 生成输出图像路径
                    base_name = os.path.basename(image_path)
                    name, ext = os.path.splitext(base_name)
                    output_path = os.path.join(Config.OUTPUT_DIR, f"{name}_ui_model{ext}")

                    # 默认框大小
                    box_size = 50
                    left = point_x - box_size // 2
                    top = point_y - box_size // 2
                    right = point_x + box_size // 2
                    bottom = point_y + box_size // 2

                    # 确保框在图像边界内
                    width, height = Image.open(image_path).size
                    left = max(0, left)
                    top = max(0, top)
                    right = min(width - 1, right)
                    bottom = min(height - 1, bottom)

                    # 绘制矩形
                    ImageDrawer.draw_rectangle(
                        image_path,
                        (left, top),
                        (right, bottom),
                        output_path
                    )

                    # 构建并发送脚本命令到服务器
                    if action == "Click":
                        script_command = f'Send "{{Click {point_x}, {point_y}}}"'
                    elif action == "Double Click":
                        script_command = f'Send "{{Click {point_x}, {point_y}}}"\nSend "{{Click {point_x}, {point_y}}}"'
                    elif action == "Right Click":
                        script_command = f'Send "{{Click {point_x}, {point_y} Right}}"'
                    else:
                        script_command = None

                    if script_command:
                        self.image_server_client.send_script_command(script_command)

                    return (True, output_path)

            elif action == "Input" and input_content:
                # 处理Input动作 - 直接发送文本
                script_command = f'Send "{input_content}"'
                self.image_server_client.send_script_command(script_command)
                return (True, "Input executed successfully")

            elif action == "Keyboard" and key_content:
                # 处理Keyboard动作 - 发送按键命令
                script_command = f'Send "{{{key_content}}}"'
                self.image_server_client.send_script_command(script_command)
                return (True, "Keyboard executed successfully")

            return (False, "Invalid action or missing parameters")

        except Exception as e:
            return (False, f"Error: {str(e)}")
