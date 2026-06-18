"""
Response parsing module
"""
import re
from typing import Optional, Tuple, List, Dict

class ResponseParser:
    """Response parser class"""

    @staticmethod
    def parse_plan(text: str) -> Optional[Dict]:
        """
        Parse task plan from LLM response.

        Expected format:
        <plan>
        <overview>High level summary</overview>
        <subtask id="1">Description</subtask>
        <subtask id="2">Description</subtask>
        ...
        </plan>

        Returns:
            Dict with 'overview' and 'subtasks' list, or None if no plan found.
        """
        plan_pattern = r'<plan>(.*?)</plan>'
        plan_match = re.search(plan_pattern, text, re.DOTALL)
        if not plan_match:
            return None

        plan_content = plan_match.group(1)

        # Extract overview
        overview_pattern = r'<overview>(.*?)</overview>'
        overview_match = re.search(overview_pattern, plan_content, re.DOTALL)
        overview = overview_match.group(1).strip() if overview_match else ""

        # Extract subtasks
        subtask_pattern = r'<subtask\s+id="(\d+)">(.*?)</subtask>'
        subtask_matches = re.findall(subtask_pattern, plan_content, re.DOTALL)
        subtasks = [
            {"id": sid, "description": desc.strip()}
            for sid, desc in subtask_matches
        ]

        if not subtasks:
            return None

        return {"overview": overview, "subtasks": subtasks}

    @staticmethod
    def parse_subtask_status_update(text: str) -> Optional[Dict]:
        """
        Parse subtask status update from LLM response.

        Expected format:
        <subtask_status id="1" status="completed">optional reason</subtask_status>

        Returns:
            Dict with 'id', 'status', and 'notes', or None if not found.
        """
        pattern = r'<subtask_status\s+id="(\d+)"\s+status="(\w+)">(.*?)</subtask_status>'
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            return None
        return {"id": match.group(1), "status": match.group(2), "notes": match.group(3).strip()}

    @staticmethod
    def extract_action_and_element(text: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[Tuple[int, int]]]:
        """
        Extract action and element from response

        Args:
            text: Response text containing <action> and <element> tags

        Returns:
            Tuple (action, element, input_content, key_content, point_coordinates), each element is None if not found
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

        # Extract point coordinates
        point_pattern = r'<point>(.*?)</point>'
        point_matches = re.findall(point_pattern, text, re.DOTALL)
        point_coords = None
        if point_matches:
            point_str = point_matches[0].strip()
            try:
                coords = [int(x.strip()) for x in point_str.split(',')]
                if len(coords) == 2:
                    point_coords = (coords[0], coords[1])
            except (ValueError, IndexError):
                pass

        return (action, element, input_content, key_content, point_coords)

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
                "point": Tuple[int, int] | None,
                "reasoning": str | None,
                "task_status": str | None
            }

        Supports:
            - <action>Click</action> → {"action": "Click"}
            - <element>Button</element> → {"element": "Button"}
            - <input>text</input> → {"text": "text"}
            - <key>Enter</key> → {"key": "Enter"}
            - <point>x,y</point> → {"point": (x, y)}
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

        # 解析 point (坐标)
        point_pattern = r'<point>(.*?)</point>'
        point_match = re.search(point_pattern, step_xml, re.DOTALL)
        if point_match:
            point_str = point_match.group(1).strip()
            try:
                coords = [int(x.strip()) for x in point_str.split(',')]
                if len(coords) == 2:
                    result["point"] = (coords[0], coords[1])
            except (ValueError, IndexError):
                pass

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
