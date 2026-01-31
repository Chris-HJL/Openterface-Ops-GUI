"""
Response parsing module
"""
import re
from typing import Optional, Tuple

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
