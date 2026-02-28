"""
Command execution module
"""
from typing import Tuple
from PIL import Image
import os
from config import Config
from .parser import ResponseParser
from .ui_ins_client import UIInsClient
from .checkbox_detector import CheckboxDetector
from ..image.drawer import ImageDrawer
from ..image_server.client import ImageServerClient
from ..coord_converter import CoordinateConverter

class CommandExecutor:
    """Command executor class"""

    def __init__(self, ui_ins_api_url: str = None, ui_ins_model: str = None, ui_ins_api_key: str = None):
        """
        Initialize command executor

        Args:
            ui_ins_api_url: UI-Ins API URL
            ui_ins_model: UI-Ins model name
            ui_ins_api_key: UI-Ins API key
        """
        self.ui_ins_client = UIInsClient(ui_ins_api_url, ui_ins_model, ui_ins_api_key)
        self.image_server_client = ImageServerClient()
        self.parser = ResponseParser()
        self.checkbox_detector = CheckboxDetector()
        # Coordinate converter - resolution will be loaded from images dynamically
        self.coord_converter = CoordinateConverter()

    def _is_checkbox_element(self, element: str) -> bool:
        """
        Check if the element description indicates a checkbox

        Args:
            element: Element description

        Returns:
            True if element is likely a checkbox
        """
        if not element:
            return False
        element_lower = element.lower()
        checkbox_keywords = [
            'checkbox', 'check box', 'check-box',
            '复选框', '勾选框', '选择框'
        ]
        return any(keyword in element_lower for keyword in checkbox_keywords)

    def _refine_checkbox_coordinates(
        self,
        image_path: str,
        point_x: int,
        point_y: int
    ) -> Tuple[int, int]:
        """
        Refine coordinates using checkbox detection

        Args:
            image_path: Image file path
            point_x: Initial X coordinate
            point_y: Initial Y coordinate

        Returns:
            Refined (x, y) coordinates
        """
        refined = self.checkbox_detector.find_checkbox_near_point(
            image_path, point_x, point_y
        )
        if refined:
            print(f"Checkbox detected: original ({point_x}, {point_y}) -> refined ({refined[0]}, {refined[1]})")
            return refined
        return (point_x, point_y)

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
                ui_model_response = self.ui_ins_client.call_api(image_path, element)
                print(f"UI-Model Response: {ui_model_response}")

                # Parse pixel coordinates from UI-Model
                pixel_x, pixel_y = self.parser.parse_coordinates(ui_model_response)

                if pixel_x != -1:
                    # ✨ Load resolution from image before coordinate conversion
                    self.coord_converter.load_resolution_from_image(image_path)

                    # Refine checkbox coordinates if needed (still in pixel coordinates)
                    if self._is_checkbox_element(element):
                        pixel_x, pixel_y = self._refine_checkbox_coordinates(
                            image_path, pixel_x, pixel_y
                        )

                    # Convert pixel coordinates to HID coordinates for TCP command
                    hid_x, hid_y = self.coord_converter.pixel_to_hid(pixel_x, pixel_y)

                    # Adjust HID coordinates to offset the impact of normalized coordinates
                    hid_y -= 10

                    print(f"[Executor] Coordinate conversion: pixel ({pixel_x}, {pixel_y}) -> HID ({hid_x}, {hid_y})")

                    base_name = os.path.basename(image_path)
                    name, ext = os.path.splitext(base_name)
                    output_path = os.path.join(Config.OUTPUT_DIR, f"{name}_ui_model{ext}")

                    # Drawing on image uses pixel coordinates
                    box_size = 50
                    left = pixel_x - box_size // 2
                    top = pixel_y - box_size // 2
                    right = pixel_x + box_size // 2
                    bottom = pixel_y + box_size // 2

                    width, height = Image.open(image_path).size
                    left = max(0, left)
                    top = max(0, top)
                    right = min(width - 1, right)
                    bottom = min(height - 1, bottom)

                    ImageDrawer.draw_rectangle(
                        image_path,
                        (left, top),
                        (right, bottom),
                        output_path
                    )

                    # TCP command uses HID coordinates
                    if action == "Click":
                        script_command = f'Send "{{Click {hid_x}, {hid_y}}}"'
                    elif action == "Double Click":
                        script_command = f'Send "{{Click {hid_x}, {hid_y}}}"\nSend "{{Click {hid_x}, {hid_y}}}"'
                    elif action == "Right Click":
                        script_command = f'Send "{{Click {hid_x}, {hid_y} Right}}"'
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
