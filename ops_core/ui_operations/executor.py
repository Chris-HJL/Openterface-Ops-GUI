"""
Command execution module
"""
from typing import Tuple, List, Dict
from PIL import Image
import os
import time
from config import Config
from .parser import ResponseParser
from .ui_ins_client import UIInsClient
from .checkbox_detector import CheckboxDetector
from ..image.drawer import ImageDrawer
from ..image_server.client import ImageServerClient
from ..coord_converter import CoordinateConverter
from ..utils.key_map import get_tcp_key_code
from ..utils.text_splitter import TextSplitter

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
                    # Click is an independent command, not embedded in Send
                    if action == "Click":
                        script_command = f'Click {hid_x},{hid_y}'
                    elif action == "Double Click":
                        script_command = f'Click {hid_x},{hid_y}\nClick {hid_x},{hid_y}'
                    elif action == "Right Click":
                        script_command = f'Click {hid_x},{hid_y} right'
                    else:
                        script_command = None

                    if script_command:
                        if action == "Double Click":
                            commands = script_command.split('\n')
                            print(f"[Executor] Double Click: sending {len(commands)} commands with delay={Config.DOUBLE_CLICK_INTERVAL}s")
                            print(f"[Executor] Commands: {commands}")
                            self.image_server_client.send_command_sequence(
                                commands, delay=Config.DOUBLE_CLICK_INTERVAL
                            )
                        else:
                            # For single click or right-click, send directly
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

    def execute_command_sequence(
        self,
        sequence: List[Dict],
        check_status: bool = True
    ) -> Tuple[bool, str]:
        """
        执行命令序列

        Args:
            sequence: 命令序列，格式见 CommandBuilder.build()
            check_status: 是否检查每个命令的状态

        Returns:
            (success, message)

        Example:
            sequence = [
                {"command": 'Send "{F5}"', "delay": 0.5},
                {"command": 'Send "{PrintScreen}"', "delay": 1.0}
            ]
            success, msg = executor.execute_command_sequence(sequence)
        """
        if not sequence:
            return (True, "Command sequence is empty")

        # 检查是否是双击操作（连续两个相同坐标的 Click 命令）
        commands = [item.get("command", "") for item in sequence]
        is_double_click = (
            len(commands) == 2 and
            commands[0].startswith("Click ") and
            commands[0] == commands[1]
        )

        if is_double_click:
            # 双击操作：使用持久连接发送命令序列
            print(f"[Executor] Detected double-click, using persistent connection")
            delays = [item.get("delay", 0.2) for item in sequence]
            success = self.image_server_client.send_command_sequence(
                commands, delay=delays[0]
            )
            if success:
                return (True, "Double-click executed successfully")
            else:
                return (False, "Failed to execute double-click")

        results = []

        for item in sequence:
            command = item.get("command", "")
            delay = item.get("delay", 0.5)

            # 跳过空命令（纯等待）
            if not command:
                time.sleep(delay)
                continue

            # 发送命令
            success = self.image_server_client.send_script_command(command)
            if not success:
                return (False, f"Failed to send command: {command}")

            # 可选的状态检查
            if check_status:
                status = self.image_server_client.check_command_status()
                if status == "failed":
                    return (False, f"Command execution failed: {command}")
                results.append({"command": command, "status": status})

            # 等待延迟
            time.sleep(delay)

        return (True, f"Successfully executed {len(results)} commands")

    def execute_with_retry(
        self,
        sequence: List[Dict],
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> Tuple[bool, str]:
        """
        带重试机制的执行

        Args:
            sequence: 命令序列
            max_retries: 最大重试次数
            retry_delay: 重试间隔（秒），会指数退避

        Returns:
            (success, message)

        Example:
            success, msg = executor.execute_with_retry(sequence, max_retries=3)
        """
        for attempt in range(max_retries):
            success, message = self.execute_command_sequence(sequence)

            if success:
                return (True, message)

            if attempt < max_retries - 1:
                # 指数退避
                delay = retry_delay * (2 ** attempt)
                print(f"Retry {attempt + 1}/{max_retries - 1} in {delay}s... (error: {message})")
                time.sleep(delay)

        return (False, f"All {max_retries} retry attempts failed: {message}")

    def execute_mixed_operation(
        self,
        image_path: str,
        operations: List[Dict]
    ) -> Tuple[bool, str]:
        """
        执行混合操作序列

        Args:
            image_path: 当前屏幕图像路径
            operations: 操作列表，格式:
                [
                    {"type": "click", "x": 960, "y": 540, "button": "left"},
                    {"type": "type", "text": "hello"},
                    {"type": "key", "key": "Enter"},
                    {"type": "wait", "duration": 1.0}
                ]

        Returns:
            (success, message)

        Example:
            operations = [
                {"type": "click", "x": 500, "y": 300},
                {"type": "type", "text": "admin"},
                {"type": "key", "key": "Tab"},
                {"type": "type", "text": "password123"},
                {"type": "click", "x": 800, "y": 500}
            ]
            success, msg = executor.execute_mixed_operation(image_path, operations)
        """
        # 从图像加载分辨率
        try:
            self.coord_converter.load_resolution_from_image(image_path)
        except Exception as e:
            return (False, f"Failed to load image resolution: {e}")

        # 构建命令序列
        builder = CommandBuilder(self)

        for op in operations:
            op_type = op.get("type", "").lower()

            if op_type == "click":
                x = op.get("x", 0)
                y = op.get("y", 0)
                button = op.get("button", "left")
                delay = op.get("delay", 0.3)
                builder.click(x, y, button, delay)

            elif op_type == "double_click":
                x = op.get("x", 0)
                y = op.get("y", 0)
                delay = op.get("delay", 0.3)
                builder.double_click(x, y, delay)

            elif op_type == "type":
                text = op.get("text", "")
                delay = op.get("delay", 0.5)
                builder.type_text(text, delay)

            elif op_type == "key":
                key = op.get("key", "")
                delay = op.get("delay", 0.3)
                builder.press_key(key, delay)

            elif op_type == "wait":
                duration = op.get("duration", 0.5)
                builder.wait(duration)

            else:
                print(f"Warning: Unknown operation type '{op_type}', skipping")

        # 执行序列
        sequence = builder.build()
        return self.execute_command_sequence(sequence)

    def execute_login_flow(
        self,
        image_path: str,
        username_coords: Tuple[int, int],
        password_coords: Tuple[int, int],
        username: str,
        password: str,
        submit_coords: Tuple[int, int],
        use_tab: bool = True
    ) -> Tuple[bool, str]:
        """
        执行标准登录流程

        Args:
            image_path: 当前屏幕图像
            username_coords: 用户名输入框像素坐标 (x, y)
            password_coords: 密码输入框像素坐标 (x, y)
            username: 用户名文本
            password: 密码文本
            submit_coords: 提交按钮像素坐标 (x, y)
            use_tab: 是否使用 Tab 切换而非点击密码框

        Returns:
            (success, message)

        Example:
            success, msg = executor.execute_login_flow(
                image_path=current_image,
                username_coords=(500, 300),
                password_coords=(500, 350),
                username="admin",
                password="password123",
                submit_coords=(800, 500)
            )
        """
        operations = [
            {"type": "click", "x": username_coords[0], "y": username_coords[1]},
            {"type": "type", "text": username}
        ]

        if use_tab:
            operations.append({"type": "key", "key": "Tab"})
        else:
            operations.append({"type": "click", "x": password_coords[0], "y": password_coords[1]})

        operations.extend([
            {"type": "type", "text": password},
            {"type": "click", "x": submit_coords[0], "y": submit_coords[1]}
        ])

        return self.execute_mixed_operation(image_path, operations)

    def execute_sequence_operations(
        self,
        sequence_ops: List[Dict],
        image_path: str,
        ui_ins_client=None
    ) -> Tuple[bool, str]:
        """
        执行序列操作（从 parser 解析的操作列表）

        Args:
            sequence_ops: 操作列表（来自 parser.parse_sequence_operations）
            image_path: 当前屏幕图像
            ui_ins_client: UI-Ins 客户端（用于元素定位）

        Returns:
            (success, message)

        Example:
            operations = [
                {"action": "Click", "element": "Username field"},
                {"action": "Type", "text": "admin"},
                {"action": "Press", "key": "Tab"}
            ]
            success, msg = executor.execute_sequence_operations(operations, image_path, ui_ins_client)
        """
        from ops_core import ResponseParser

        # 从图像加载分辨率
        try:
            self.coord_converter.load_resolution_from_image(image_path)
        except Exception as e:
            return (False, f"Failed to load image resolution: {e}")

        # 构建操作列表
        operations = []
        parser = ResponseParser()

        for op in sequence_ops:
            action = op.get("action", "")
            element = op.get("element")

            if action in ["Click", "Double Click", "Right Click"]:
                # 需要定位元素坐标
                if element and ui_ins_client:
                    try:
                        # 调用 UI-Model 获取坐标
                        ui_response = ui_ins_client.call_api(image_path, element)
                        x, y = parser.parse_coordinates(ui_response)

                        if x != -1:
                            op_type = "click" if action == "Click" else action.lower().replace(" ", "_")
                            operations.append({
                                "type": op_type,
                                "x": x,
                                "y": y
                            })
                            if action == "Right Click":
                                operations[-1]["button"] = "right"
                            continue
                    except Exception as e:
                        print(f"Warning: Failed to get coordinates for '{element}': {e}")

                # 无法获取坐标，跳过
                print(f"Warning: Skipping {action} without coordinates: {element}")

            elif action == "Type":
                text = op.get("text", "")
                operations.append({"type": "type", "text": text})

            elif action == "Press":
                key = op.get("key", "")
                operations.append({"type": "key", "key": key})

            elif action == "Wait":
                duration = op.get("text", "1")
                try:
                    duration = float(duration)
                except ValueError:
                    duration = 1.0
                operations.append({"type": "wait", "duration": duration})

            else:
                print(f"Warning: Unknown action type in sequence: {action}")

        if not operations:
            return (False, "No valid operations to execute")

        # 执行操作列表
        return self.execute_mixed_operation(image_path, operations)

    def execute_search_flow(
        self,
        image_path: str,
        search_box_coords: Tuple[int, int],
        search_query: str
    ) -> Tuple[bool, str]:
        """
        执行标准搜索流程

        Args:
            image_path: 当前屏幕图像
            search_box_coords: 搜索框像素坐标 (x, y)
            search_query: 搜索查询文本

        Returns:
            (success, message)

        Example:
            success, msg = executor.execute_search_flow(
                image_path=current_image,
                search_box_coords=(300, 100),
                search_query="python tutorial"
            )
        """
        operations = [
            {"type": "click", "x": search_box_coords[0], "y": search_box_coords[1]},
            {"type": "type", "text": search_query},
            {"type": "key", "key": "Enter"}
        ]

        return self.execute_mixed_operation(image_path, operations)


class CommandBuilder:
    """命令构建器 - 用于构建标准化的命令序列"""

    def __init__(self, executor: 'CommandExecutor'):
        """
        初始化构建器

        Args:
            executor: 命令执行器实例
        """
        self.executor = executor
        self.commands = []  # 命令文本
        self.delays = []    # 每命令后的延迟（秒）

    def click(
        self,
        x: int,
        y: int,
        button: str = "left",
        delay: float = 0.3
    ) -> 'CommandBuilder':
        """
        添加点击命令

        Args:
            x: 像素坐标 X（会自动转换为 HID）
            y: 像素坐标 Y（会自动转换为 HID）
            button: 鼠标按钮 ("left" | "right" | "middle")
            delay: 点击后延迟（秒）

        Returns:
            self（支持链式调用）

        Example:
            builder = CommandBuilder(executor)
            builder.click(960, 540).click(1000, 600, "right")

        Note:
            Click 命令是独立命令，格式为：Click x,y [button]
            不是嵌入式命令 Send "{Click x,y}"
            独立 Click 命令使用 50ms 延迟，嵌入式使用 5ms 延迟
        """
        # 坐标转换
        hid_x, hid_y = self.executor.coord_converter.pixel_to_hid(x, y)

        # 应用坐标偏移（与现有 executor.py 保持一致）
        hid_y -= 10

        # 标准化按钮名称
        button = button.lower() if button else "left"
        if button not in ["left", "right", "middle"]:
            print(f"Warning: Invalid button '{button}', using 'left'")
            button = "left"

        # 生成独立 Click 命令（不是嵌入式）
        if button == "left":
            cmd = f'Click {hid_x},{hid_y}'
        else:
            cmd = f'Click {hid_x},{hid_y} {button}'

        self.commands.append(cmd)
        self.delays.append(delay)

        return self

    def double_click(
        self,
        x: int,
        y: int,
        delay: float = 0.3
    ) -> 'CommandBuilder':
        """
        添加双击命令

        Args:
            x: 像素坐标 X
            y: 像素坐标 Y
            delay: 第二次点击后延迟

        Returns:
            self

        Example:
            builder.double_click(960, 540)

        Note:
            双击通过发送两个独立的 Click 命令实现
            两次点击之间的延迟由 Config.DOUBLE_CLICK_INTERVAL 控制（默认 50ms）
            使用独立 Click 命令而非嵌入式
        """
        hid_x, hid_y = self.executor.coord_converter.pixel_to_hid(x, y)

        # 应用坐标偏移
        hid_y -= 10

        # 第一次点击
        self.commands.append(f'Click {hid_x},{hid_y}')
        self.delays.append(Config.DOUBLE_CLICK_INTERVAL)  # 双击间隔

        # 第二次点击
        self.commands.append(f'Click {hid_x},{hid_y}')
        self.delays.append(delay)

        return self

    def type_text(
        self,
        text: str,
        delay: float = 0.5
    ) -> 'CommandBuilder':
        """
        添加文本输入命令

        Args:
            text: 要输入的文本（>25 字符会自动拆分）
            delay: 输入完成后延迟

        Returns:
            self

        Note:
            长文本会自动拆分为多个 Send 命令，每个≤25 字符
            块间延迟固定为 0.3 秒

        Example:
            builder.type_text("这是一段很长的文本内容，会自动拆分")
        """
        # 使用文本拆分器
        splitter = TextSplitter(max_length=25, delay_between_chunks=0.3)

        # 拆分文本
        chunks = splitter.split(text)

        # 添加每个块
        for i, chunk in enumerate(chunks):
            self.commands.append(f'Send "{chunk}"')
            # 最后一个使用指定的延迟，其他使用块间延迟
            self.delays.append(delay if i == len(chunks) - 1 else splitter.delay_between_chunks)

        return self

    def press_key(
        self,
        key: str,
        delay: float = 0.3
    ) -> 'CommandBuilder':
        """
        添加按键命令

        Args:
            key: 键名（支持多种格式）
            delay: 按键后延迟

        Returns:
            self

        Supported Key Names:
            - 标准：'Enter', 'Tab', 'Escape', 'Backspace', 'Delete'
            - 方向：'Up', 'Down', 'Left', 'Right'
            - 功能：'F1' - 'F12'
            - 控制：'Ctrl', 'Alt', 'Shift', 'Win'
            - 其他：'Space', 'PrintScreen', 'ScrollLock', 'Pause'

        Example:
            builder.press_key("Enter").press_key("Tab").press_key("F5")
        """
        try:
            # 获取 TCP 格式键码
            key_code = get_tcp_key_code(key)
            cmd = f'Send "{key_code}"'
            self.commands.append(cmd)
            self.delays.append(delay)
        except KeyError as e:
            print(f"Warning: {e} Skipping this key.")

        return self

    def wait(self, duration: float) -> 'CommandBuilder':
        """
        添加等待延迟

        Args:
            duration: 等待时长（秒）

        Returns:
            self

        Example:
            builder.wait(1.5)
        """
        # 使用空命令作为占位符，实际上通过延迟实现
        self.commands.append("")
        self.delays.append(duration)
        return self

    def build(self) -> List[Dict]:
        """
        构建命令序列

        Returns:
            [
                {"command": 'Send "hello"', "delay": 0.3},
                {"command": 'Send "{Enter}"', "delay": 0.5},
                ...
            ]

        Example:
            sequence = (
                CommandBuilder(executor)
                .click(960, 540)
                .type_text("admin")
                .press_key("Tab")
                .type_text("password123")
                .click(800, 500)
                .build()
            )
        """
        # 过滤掉空命令（wait 命令）
        result = []
        for cmd, delay in zip(self.commands, self.delays):
            if cmd:  # 非空命令
                result.append({"command": cmd, "delay": delay})
            else:  # 空命令（wait）
                # 如果是连续的等待，合并延迟
                if result and result[-1]["command"] == "":
                    result[-1]["delay"] += delay
                else:
                    result.append({"command": "", "delay": delay})

        return result