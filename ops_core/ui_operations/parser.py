"""
响应解析模块
"""
import re
from typing import Optional, Tuple

class ResponseParser:
    """响应解析器类"""

    @staticmethod
    def extract_action_and_element(text: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """
        从响应中提取action和element

        Args:
            text: 包含 <action> 和 <element> 标签的响应文本

        Returns:
            元组 (action, element, input_content, key_content)，每个元素如果未找到则为None
        """
        # 提取action
        action_pattern = r'<action>(.*?)</action>'
        action_matches = re.findall(action_pattern, text, re.DOTALL)
        action = action_matches[0].strip() if action_matches else None

        # 提取element
        element_pattern = r'<element>(.*?)</element>'
        element_matches = re.findall(element_pattern, text, re.DOTALL)
        element = element_matches[0].strip() if element_matches else None

        # 提取input内容
        input_pattern = r'<input>(.*?)</input>'
        input_matches = re.findall(input_pattern, text, re.DOTALL)
        input_content = input_matches[0].strip() if input_matches else None

        # 提取key内容
        key_pattern = r'<key>(.*?)</key>'
        key_matches = re.findall(key_pattern, text, re.DOTALL)
        key_content = key_matches[0].strip() if key_matches else None

        return (action, element, input_content, key_content)

    @staticmethod
    def parse_coordinates(raw_string: str) -> Tuple[int, int]:
        """
        解析坐标字符串

        Args:
            raw_string: 坐标字符串

        Returns:
            元组 (x, y)，失败返回 (-1, -1)
        """
        try:
            # 尝试从字符串中提取坐标，支持 [x, y] 和 (x, y) 两种格式
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
