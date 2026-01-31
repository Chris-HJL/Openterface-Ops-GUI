"""
Command execution module
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
    """Command executor class"""

    def __init__(self, ui_ins_api_url: str = None, ui_ins_model: str = None):
        """
        Initialize command executor

        Args:
            ui_ins_api_url: UI-Ins API URL
            ui_ins_model: UI-Ins model name
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
        Process UI element request

        Args:
            image_path: Image file path
            instruction: Instruction text
            action: Action type
            element: Element name
            input_content: Input content
            key_content: Key content

        Returns:
            Tuple (success, output image path or error message)
        """
        try:
            if action in ["Click", "Double Click", "Right Click"] and element:
                # Call UI-Model API to get coordinates
                ui_model_response = self.ui_ins_client.call_api(image_path, element)
                print(f"UI-Model Response: {ui_model_response}")

                # Parse coordinates
                point_x, point_y = self.parser.parse_coordinates(ui_model_response)

                if point_x != -1:
                    # Generate output image path
                    base_name = os.path.basename(image_path)
                    name, ext = os.path.splitext(base_name)
                    output_path = os.path.join(Config.OUTPUT_DIR, f"{name}_ui_model{ext}")

                    # Default box size
                    box_size = 50
                    left = point_x - box_size // 2
                    top = point_y - box_size // 2
                    right = point_x + box_size // 2
                    bottom = point_y + box_size // 2

                    # Ensure box within image boundaries
                    width, height = Image.open(image_path).size
                    left = max(0, left)
                    top = max(0, top)
                    right = min(width - 1, right)
                    bottom = min(height - 1, bottom)

                    # Draw rectangle
                    ImageDrawer.draw_rectangle(
                        image_path,
                        (left, top),
                        (right, bottom),
                        output_path
                    )

                    # Build and send script command to server
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
                # Process Input action - send text directly
                script_command = f'Send "{input_content}"'
                self.image_server_client.send_script_command(script_command)
                return (True, "Input executed successfully")

            elif action == "Keyboard" and key_content:
                # Process Keyboard action - send key command
                script_command = f'Send "{{{key_content}}}"'
                self.image_server_client.send_script_command(script_command)
                return (True, "Keyboard executed successfully")

            return (False, "Invalid action or missing parameters")

        except Exception as e:
            return (False, f"Error: {str(e)}")
