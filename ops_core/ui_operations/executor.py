"""
Command execution module
"""
from typing import Tuple, List, Dict
from PIL import Image
import os
import time
from config import Config
from .parser import ResponseParser
from .checkbox_detector import CheckboxDetector
from ..image.drawer import ImageDrawer
from ..image_server.client import ImageServerClient
from ..coord_converter import CoordinateConverter
from ..utils.key_map import get_tcp_key_code, is_combo_key
from ..utils.text_splitter import TextSplitter

class CommandExecutor:
    """Command executor class"""

    def __init__(self):
        """Initialize command executor"""
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

    def execute_click_at_point(
        self,
        image_path: str,
        action: str,
        point_coords: Tuple[int, int]
    ) -> Tuple[bool, str]:
        """
        直接在指定坐标执行点击操作（使用归一化坐标 0-1000）

        Args:
            image_path: 图像文件路径
            action: 操作类型 (Click, Double Click, Right Click)
            point_coords: 归一化坐标 (norm_x, norm_y) 范围 0-1000

        Returns:
            Tuple (success, output image path or error message)
        """
        try:
            norm_x, norm_y = point_coords
            
            # Convert normalized coordinates to actual pixel coordinates
            pixel_x, pixel_y = self.coord_converter.denormalize_coordinates(
                norm_x, norm_y, image_path
            )
            
            # Load resolution from image before coordinate conversion
            self.coord_converter.load_resolution_from_image(image_path)
            
            # Convert pixel coordinates to HID coordinates for TCP command
            hid_x, hid_y = self.coord_converter.pixel_to_hid(pixel_x, pixel_y)

            # Y offset is now applied inside coord_converter.pixel_to_hid()

            print(f"[Executor] Coordinate conversion: normalized ({norm_x}, {norm_y}) -> pixel ({pixel_x}, {pixel_y}) -> HID ({hid_x}, {hid_y})")
            
            # Draw rectangle on image for visualization
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            output_path = os.path.join(Config.OUTPUT_DIR, f"{name}_clicked{ext}")
            
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
            
            # Send TCP command
            if action == "Click":
                script_command = f'Click {hid_x},{hid_y}'
                self.image_server_client.send_script_command(script_command)
            elif action == "Double Click":
                commands = [f'Click {hid_x},{hid_y}', f'Click {hid_x},{hid_y}']
                print(f"[Executor] Double Click: sending 2 commands with delay={Config.DOUBLE_CLICK_INTERVAL}s")
                self.image_server_client.send_command_sequence(
                    commands, delay=Config.DOUBLE_CLICK_INTERVAL
                )
            elif action == "Right Click":
                script_command = f'Click {hid_x},{hid_y} right'
                self.image_server_client.send_script_command(script_command)
            else:
                return (False, f"Unsupported action type: {action}")
            
            return (True, output_path)
        
        except Exception as e:
            return (False, f"Error: {str(e)}")

    def execute_input_at_point(
        self,
        image_path: str,
        point_coords: Tuple[int, int],
        text: str
    ) -> Tuple[bool, str]:
        """
        在指定坐标点击然后输入文本

        Args:
            image_path: 图像文件路径
            point_coords: 归一化坐标 (norm_x, norm_y) 范围 0-1000
            text: 要输入的文本

        Returns:
            Tuple (success, message)
        """
        try:
            norm_x, norm_y = point_coords
            
            # Convert normalized coordinates to pixel coordinates
            pixel_x, pixel_y = self.coord_converter.denormalize_coordinates(
                norm_x, norm_y, image_path
            )
            
            # Load resolution from image
            self.coord_converter.load_resolution_from_image(image_path)
            
            # Convert to HID coordinates (Y offset applied inside pixel_to_hid)
            hid_x, hid_y = self.coord_converter.pixel_to_hid(pixel_x, pixel_y)

            print(f"[Executor] Input at point: normalized ({norm_x}, {norm_y}) -> HID ({hid_x}, {hid_y})")
            
            # First, click to focus the input field
            click_command = f'Click {hid_x},{hid_y}'
            self.image_server_client.send_script_command(click_command)
            
            # Small delay to ensure focus
            time.sleep(0.2)
            
            # Then send the text
            # Use text splitter for long text
            splitter = TextSplitter(max_length=25, delay_between_chunks=0.3)
            chunks = splitter.split(text)
            
            for i, chunk in enumerate(chunks):
                send_command = f'Send "{chunk}"'
                self.image_server_client.send_script_command(send_command)
                if i < len(chunks) - 1:
                    time.sleep(splitter.delay_between_chunks)
            
            return (True, f"Input executed successfully at ({norm_x}, {norm_y})")
        
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

            elif op_type == "move_mouse":
                x = op.get("x", 0)
                y = op.get("y", 0)
                delay = op.get("delay", Config.MOUSE_MOVE_DELAY)
                builder.move_mouse(x, y, delay)

            elif op_type == "triple_click":
                x = op.get("x", 0)
                y = op.get("y", 0)
                delay = op.get("delay", 0.3)
                builder.triple_click(x, y, delay)

            elif op_type == "lock_state":
                lock_type = op.get("lock_type", "CapsLock")
                state = op.get("state", "On")
                delay = op.get("delay", 0.3)
                builder.set_lock_state(lock_type, state, delay)

            elif op_type == "screenshot":
                path = op.get("path", "/tmp/screenshot.png")
                delay = op.get("delay", 0.5)
                builder.full_screen_capture(path, delay)

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
        image_path: str
    ) -> Tuple[bool, str]:
        """
        执行序列操作（从 parser 解析的操作列表）

        Args:
            sequence_ops: 操作列表（来自 parser.parse_sequence_operations）
            image_path: 当前屏幕图像

        Returns:
            (success, message)
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
            point = op.get("point")  # Get point coordinates from LLM

            if action in ["Click", "Double Click", "Right Click"]:
                # Use point coordinates from LLM directly
                if point:
                    op_type = "click" if action == "Click" else action.lower().replace(" ", "_")
                    operations.append({
                        "type": op_type,
                        "norm_x": point[0],
                        "norm_y": point[1]
                    })
                    if action == "Right Click":
                        operations[-1]["button"] = "right"
                    continue
                
                # Fallback: warn if no point provided
                print(f"Warning: Skipping {action} without point coordinates: {element}")

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

        # Execute operations with normalized coordinates
        return self.execute_sequence_with_norm_coords(image_path, operations)

    def execute_sequence_with_norm_coords(
        self,
        image_path: str,
        operations: List[Dict]
    ) -> Tuple[bool, str]:
        """
        执行带有归一化坐标的操作序列

        Args:
            image_path: 屏幕图像路径
            operations: 操作列表，包含归一化坐标

        Returns:
            (success, message)
        """
        try:
            # Load resolution from image
            self.coord_converter.load_resolution_from_image(image_path)
            
            results = []
            
            for op in operations:
                op_type = op.get("type", "").lower()
                
                if op_type == "click":
                    # Convert normalized coordinates to HID
                    norm_x = op.get("norm_x", 0)
                    norm_y = op.get("norm_y", 0)
                    button = op.get("button", "left")
                    
                    # Normalize to pixel
                    pixel_x, pixel_y = self.coord_converter.denormalize_coordinates(norm_x, norm_y, image_path)
                    
                    # Pixel to HID (Y offset applied inside pixel_to_hid)
                    hid_x, hid_y = self.coord_converter.pixel_to_hid(pixel_x, pixel_y)

                    if button == "left":
                        cmd = f'Click {hid_x},{hid_y}'
                    else:
                        cmd = f'Click {hid_x},{hid_y} {button}'
                    
                    self.image_server_client.send_script_command(cmd)
                    results.append(cmd)
                    
                elif op_type == "double_click":
                    norm_x = op.get("norm_x", 0)
                    norm_y = op.get("norm_y", 0)
                    
                    pixel_x, pixel_y = self.coord_converter.denormalize_coordinates(norm_x, norm_y, image_path)
                    hid_x, hid_y = self.coord_converter.pixel_to_hid(pixel_x, pixel_y)

                    commands = [f'Click {hid_x},{hid_y}', f'Click {hid_x},{hid_y}']
                    self.image_server_client.send_command_sequence(
                        commands, delay=Config.DOUBLE_CLICK_INTERVAL
                    )
                    results.extend(commands)
                    
                elif op_type == "type":
                    text = op.get("text", "")
                    splitter = TextSplitter(max_length=25, delay_between_chunks=0.3)
                    chunks = splitter.split(text)
                    
                    for i, chunk in enumerate(chunks):
                        cmd = f'Send "{chunk}"'
                        self.image_server_client.send_script_command(cmd)
                        results.append(cmd)
                        if i < len(chunks) - 1:
                            time.sleep(splitter.delay_between_chunks)
                    
                elif op_type == "key":
                    key = op.get("key", "")
                    key_code = get_tcp_key_code(key)
                    # Combo keys don't need quotes, regular keys do
                    if is_combo_key(key):
                        cmd = f'Send {key_code}'
                    else:
                        cmd = f'Send "{key_code}"'
                    self.image_server_client.send_script_command(cmd)
                    results.append(cmd)
                    
                elif op_type == "wait":
                    duration = op.get("duration", 0.5)
                    time.sleep(duration)

                elif op_type == "move_mouse":
                    norm_x = op.get("norm_x", 0)
                    norm_y = op.get("norm_y", 0)
                    pixel_x, pixel_y = self.coord_converter.denormalize_coordinates(norm_x, norm_y, image_path)
                    hid_x, hid_y = self.coord_converter.pixel_to_hid(pixel_x, pixel_y)
                    cmd = f'MouseMove {hid_x},{hid_y}'
                    self.image_server_client.send_script_command(cmd)
                    results.append(cmd)

                elif op_type == "triple_click":
                    norm_x = op.get("norm_x", 0)
                    norm_y = op.get("norm_y", 0)
                    pixel_x, pixel_y = self.coord_converter.denormalize_coordinates(norm_x, norm_y, image_path)
                    hid_x, hid_y = self.coord_converter.pixel_to_hid(pixel_x, pixel_y)
                    for i in range(3):
                        self.image_server_client.send_script_command(f'Click {hid_x},{hid_y}')
                        results.append(f'Click {hid_x},{hid_y}')
                        if i < 2:
                            time.sleep(Config.TRIPLE_CLICK_INTERVAL)

                elif op_type == "lock_state":
                    lock_type = op.get("lock_type", "CapsLock")
                    state = op.get("state", "On")
                    cmd = f'Set{lock_type}State {state}'
                    self.image_server_client.send_script_command(cmd)
                    results.append(cmd)

                elif op_type == "screenshot":
                    path = op.get("path", "/tmp/screenshot.png")
                    cmd = f'FullScreenCapture "{path}"'
                    self.image_server_client.send_script_command(cmd)
                    results.append(cmd)
                
                # Delay between operations
                time.sleep(0.3)
            
            return (True, f"Successfully executed {len(results)} operations")
        
        except Exception as e:
            return (False, f"Error executing sequence: {str(e)}")

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
        # 坐标转换（Y offset 已内置在 pixel_to_hid 中）
        hid_x, hid_y = self.executor.coord_converter.pixel_to_hid(x, y)

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

        # Y offset 已内置在 pixel_to_hid 中

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
        添加按键命令（支持普通键和组合键）

        Args:
            key: 键名或组合键字符串
                普通键: 'Enter', 'Tab', 'Escape', 'Backspace', 'Delete', 'Up', 'Down', 'F1'..'F12'
                组合键: '^c' (Ctrl+C), '^v' (Ctrl+V), '^a' (Ctrl+A), '!F4' (Alt+F4),
                       '#e' (Win+E), '^+esc' (Ctrl+Shift+Esc)
            delay: 按键后延迟

        Returns:
            self

        Prefix Notation:
            ^ = Ctrl, + = Shift, ! = Alt, # = Win

        Example:
            builder.press_key("Enter").press_key("^c").press_key("!F4")
        """
        try:
            # 获取 TCP 格式键码
            key_code = get_tcp_key_code(key)
            # 组合键不需要引号，普通键需要引号
            if is_combo_key(key):
                cmd = f'Send {key_code}'
            else:
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

    def move_mouse(
        self,
        x: int,
        y: int,
        delay: float = 0.2
    ) -> 'CommandBuilder':
        """
        添加鼠标移动命令（不点击）

        Args:
            x: 像素坐标 X
            y: 像素坐标 Y
            delay: 移动后延迟

        Returns:
            self

        Example:
            builder.move_mouse(960, 540)  # 移动到屏幕中心
        """
        hid_x, hid_y = self.executor.coord_converter.pixel_to_hid(x, y)
        cmd = f'MouseMove {hid_x},{hid_y}'
        self.commands.append(cmd)
        self.delays.append(delay)
        return self

    def triple_click(
        self,
        x: int,
        y: int,
        delay: float = 0.3
    ) -> 'CommandBuilder':
        """
        添加三击命令（选中整行/整段文本）

        Args:
            x: 像素坐标 X
            y: 像素坐标 Y
            delay: 第三次点击后延迟

        Returns:
            self

        Example:
            builder.triple_click(960, 540)  # 三击选中整段
        """
        hid_x, hid_y = self.executor.coord_converter.pixel_to_hid(x, y)

        # 三次点击
        for i in range(3):
            self.commands.append(f'Click {hid_x},{hid_y}')
            if i < 2:
                self.delays.append(Config.TRIPLE_CLICK_INTERVAL)
            else:
                self.delays.append(delay)

        return self

    def set_lock_state(
        self,
        lock_type: str,
        state: str,
        delay: float = 0.3
    ) -> 'CommandBuilder':
        """
        设置锁定键状态（CapsLock/NumLock/ScrollLock）

        Args:
            lock_type: 锁定键类型 ("CapsLock", "NumLock", "ScrollLock")
            state: 目标状态 ("On", "Off", "Toggle")
            delay: 操作后延迟

        Returns:
            self

        Example:
            builder.set_lock_state("CapsLock", "On")
            builder.set_lock_state("NumLock", "Off")
        """
        valid_types = ["CapsLock", "NumLock", "ScrollLock"]
        valid_states = ["On", "Off", "Toggle"]
        if lock_type not in valid_types:
            print(f"Warning: Invalid lock_type '{lock_type}', must be one of {valid_types}")
            return self
        if state not in valid_states:
            print(f"Warning: Invalid state '{state}', must be one of {valid_states}")
            return self
        cmd = f'Set{lock_type}State {state}'
        self.commands.append(cmd)
        self.delays.append(delay)
        return self

    def full_screen_capture(
        self,
        path: str,
        delay: float = 0.5
    ) -> 'CommandBuilder':
        """
        添加全屏截图命令

        Args:
            path: 截图保存路径
            delay: 截图后延迟

        Returns:
            self

        Example:
            builder.full_screen_capture("/tmp/screenshot.png")
        """
        cmd = f'FullScreenCapture "{path}"'
        self.commands.append(cmd)
        self.delays.append(delay)
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