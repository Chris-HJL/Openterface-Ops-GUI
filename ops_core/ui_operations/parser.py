"""
Response parsing module
"""
import re
from typing import Optional, Tuple, List, Dict

class ResponseParser:
    """Response parser class"""

    @staticmethod
    def extract_action_and_element(text: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """
        Extract action and element from response

        Args:
            text: Response text containing <action> and <element> tags

        Returns:
            Tuple (action, element, input_content, key_content), each element is None if not found
        """
        # Extract action
        action_pattern = r'<action>(.*?)</action>'
        action_matches = re.findall(action_pattern, text, re.DOTALL)
        action = action_matches[0].strip() if action_matches else None

        # Extract element
        element_pattern = r'<element>(.*?)</element>'
        element_matches = re.findall(element_pattern, text, re.DOTALL)
        element = element_matches[0].strip() if element_matches else None

        # Extract input content
        input_pattern = r'<input>(.*?)</input>'
        input_matches = re.findall(input_pattern, text, re.DOTALL)
        input_content = input_matches[0].strip() if input_matches else None

        # Extract key content
        key_pattern = r'<key>(.*?)</key>'
        key_matches = re.findall(key_pattern, text, re.DOTALL)
        key_content = key_matches[0].strip() if key_matches else None

        return (action, element, input_content, key_content)

    @staticmethod
    def parse_coordinates(raw_string: str) -> Tuple[int, int]:
        """
        Parse coordinate string

        Args:
            raw_string: Coordinate string

        Returns:
            Tuple (x, y), returns (-1, -1) on failure
        """
        try:
            # Try to extract coordinates from string, supports [x, y], (x, y) and [x1,y1,x2,y2] formats
            match = re.search(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', raw_string)
            if match:
                x1 = int(match.group(1))
                y1 = int(match.group(2))
                return (x1, y1)

            match = re.search(r'\[(\d+),\s*(\d+)\]', raw_string)
            if not match:
                match = re.search(r'\((\d+),\s*(\d+)\)', raw_string)

            if match:
                x = int(match.group(1))
                y = int(match.group(2))
                return (x, y)
            return (-1, -1)
        except Exception:
            return (-1, -1)

    def parse_sequence_operations(self, response: str) -> List[Dict]:
        """
        解析序列操作

        Args:
            response: LLM 响应文本，包含 <steps> 标签

        Returns:
            [
                {
                    "action": "Click",
                    "element": "Username field",
                    "reasoning": "First, focus on the username input"
                },
                {
                    "action": "Type",
                    "text": "admin"
                },
                {
                    "action": "Press",
                    "key": "Tab",
                    "reasoning": "Move to password field"
                }
            ]

        Raises:
            ValueError: 解析失败或格式不正确

        Note:
            如果响应不包含 <steps> 标签，返回空列表
        """
        # 查找 <steps> 标签
        steps_pattern = r'<steps>(.*?)</steps>'
        steps_match = re.search(steps_pattern, response, re.DOTALL)

        if not steps_match:
            return []

        steps_content = steps_match.group(1)

        # 查找所有 <step> 标签
        step_pattern = r'<step>(.*?)</step>'
        step_matches = re.findall(step_pattern, steps_content, re.DOTALL)

        if not step_matches:
            return []

        # 解析每个 step
        operations = []
        for step_xml in step_matches:
            try:
                op = self.parse_operation_step(step_xml)
                if op:
                    operations.append(op)
            except Exception as e:
                print(f"Warning: Failed to parse step: {e}")
                continue

        return operations

    def parse_operation_step(self, step_xml: str) -> Optional[Dict]:
        """
        解析单个操作步骤

        Args:
            step_xml: <step>...</step> XML 片段

        Returns:
            {
                "action": str,
                "element": str | None,
                "text": str | None,
                "key": str | None,
                "reasoning": str | None,
                "task_status": str | None
            }

        Supports:
            - <action>Click</action> → {"action": "Click"}
            - <element>Button</element> → {"element": "Button"}
            - <input>text</input> → {"text": "text"}
            - <key>Enter</key> → {"key": "Enter"}
            - <reasoning>...</reasoning> → {"reasoning": "..."}
        """
        result = {}

        # 解析 action
        action_pattern = r'<action>(.*?)</action>'
        action_match = re.search(action_pattern, step_xml, re.DOTALL)
        if action_match:
            result["action"] = action_match.group(1).strip()

        # 解析 element
        element_pattern = r'<element>(.*?)</element>'
        element_match = re.search(element_pattern, step_xml, re.DOTALL)
        if element_match:
            result["element"] = element_match.group(1).strip()

        # 解析 text (用于 Type 操作)
        text_pattern = r'<text>(.*?)</text>'
        text_match = re.search(text_pattern, step_xml, re.DOTALL)
        if text_match:
            result["text"] = text_match.group(1).strip()

        # 解析 key (用于 Press 操作)
        key_pattern = r'<key>(.*?)</key>'
        key_match = re.search(key_pattern, step_xml, re.DOTALL)
        if key_match:
            result["key"] = key_match.group(1).strip()

        # 解析 reasoning
        reasoning_pattern = r'<reasoning>(.*?)</reasoning>'
        reasoning_match = re.search(reasoning_pattern, step_xml, re.DOTALL)
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip()

        # 解析 task_status
        status_pattern = r'<task_status>(.*?)</task_status>'
        status_match = re.search(status_pattern, step_xml, re.DOTALL)
        if status_match:
            result["task_status"] = status_match.group(1).strip()

        # 必须有 action 才返回
        if "action" not in result:
            return None

        return result

    def parse_operation_list_from_sequence(
        self,
        response: str,
        image_path: str,
        ui_ins_client=None
    ) -> List[Dict]:
        """
        从序列响应中构建操作列表（带坐标解析）

        Args:
            response: LLM 响应（包含 <steps>）
            image_path: 屏幕图像（用于元素定位）
            ui_ins_client: UI-Ins 客户端（可选，用于元素定位）

        Returns:
            [
                {"type": "click", "x": 960, "y": 540},  # 带坐标
                {"type": "type", "text": "hello"},
                {"type": "key", "key": "Enter"}
            ]

        Process:
            1. 解析 <steps> 和每个 <step>
            2. 对每个包含 <element> 的 Click 操作，调用 UI-Model 获取坐标
            3. 将坐标转换为操作列表格式

        Note:
            需要多次调用 UI-Model（每个元素一次）
            可通过缓存优化性能
        """
        operations = self.parse_sequence_operations(response)
        result = []

        for op in operations:
            action = op.get("action", "")
            element = op.get("element")

            # 转换操作类型
            if action in ["Click", "Double Click", "Right Click"]:
                op_type = "click" if action == "Click" else action.lower().replace(" ", "_")

                # 如果有 element，需要调用 UI-Model 获取坐标
                if element and ui_ins_client:
                    try:
                        ui_response = ui_ins_client.call_api(image_path, element)
                        x, y = self.parse_coordinates(ui_response)

                        if x != -1:
                            click_op = {
                                "type": op_type,
                                "x": x,
                                "y": y
                            }
                            if action == "Right Click":
                                click_op["button"] = "right"
                            result.append(click_op)
                            continue
                    except Exception as e:
                        print(f"Warning: Failed to get coordinates for '{element}': {e}")

                # 无法获取坐标，跳过或添加不带坐标的操作
                print(f"Warning: Skipping operation without coordinates: {op}")
                continue

            elif action == "Type":
                result.append({
                    "type": "type",
                    "text": op.get("text", "")
                })

            elif action == "Press":
                result.append({
                    "type": "key",
                    "key": op.get("key", "")
                })

            elif action == "Wait":
                # Wait 操作可能需要特殊处理
                duration = op.get("text", "1")  # 默认 1 秒
                try:
                    duration = float(duration)
                except ValueError:
                    duration = 1.0
                result.append({
                    "type": "wait",
                    "duration": duration
                })

            else:
                print(f"Warning: Unknown action type in sequence: {action}")

        return result
