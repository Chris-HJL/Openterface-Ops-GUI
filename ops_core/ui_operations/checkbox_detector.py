"""
Checkbox detection module using contour analysis
"""
import cv2
import numpy as np
from typing import Optional, Tuple, List
from PIL import Image


class CheckboxDetector:
    """Detect checkbox elements using contour analysis"""

    DEFAULT_SEARCH_RADIUS = 50
    MIN_CHECKBOX_SIZE = 8
    MAX_CHECKBOX_SIZE = 40
    ASPECT_RATIO_THRESHOLD = 0.3

    @classmethod
    def find_checkbox_near_point(
        cls,
        image_path: str,
        center_x: int,
        center_y: int,
        search_radius: int = None
    ) -> Optional[Tuple[int, int]]:
        """
        Find checkbox near the given point using contour detection

        Args:
            image_path: Path to the screenshot image
            center_x: X coordinate from UI model
            center_y: Y coordinate from UI model
            search_radius: Search radius around the point

        Returns:
            Tuple (x, y) of checkbox center, or None if not found
        """
        if search_radius is None:
            search_radius = cls.DEFAULT_SEARCH_RADIUS

        try:
            image = cv2.imread(image_path)
            if image is None:
                return None

            height, width = image.shape[:2]

            left = max(0, center_x - search_radius)
            top = max(0, center_y - search_radius)
            right = min(width, center_x + search_radius)
            bottom = min(height, center_y + search_radius)

            roi = image[top:bottom, left:right]
            if roi.size == 0:
                return None

            checkboxes = cls._detect_checkboxes_in_roi(roi)
            
            if not checkboxes:
                return None

            best_checkbox = cls._find_closest_checkbox(checkboxes, 
                                                        center_x - left, 
                                                        center_y - top)
            
            if best_checkbox:
                abs_x = best_checkbox[0] + left
                abs_y = best_checkbox[1] + top
                return (abs_x, abs_y)

            return None

        except Exception as e:
            print(f"Checkbox detection error: {str(e)}")
            return None

    @classmethod
    def _detect_checkboxes_in_roi(cls, roi: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect all checkbox-like shapes in the region of interest

        Args:
            roi: Region of interest image

        Returns:
            List of (center_x, center_y, width, height) tuples
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        edges = cv2.Canny(blurred, 50, 150)
        
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        checkboxes = []
        for contour in contours:
            checkbox = cls._analyze_contour(contour)
            if checkbox:
                checkboxes.append(checkbox)

        return checkboxes

    @classmethod
    def _analyze_contour(cls, contour: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Analyze if a contour is a checkbox

        Args:
            contour: OpenCV contour

        Returns:
            (center_x, center_y, width, height) if checkbox, None otherwise
        """
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 20:
            return None

        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        
        if len(approx) != 4:
            return None

        x, y, w, h = cv2.boundingRect(approx)
        
        if w < cls.MIN_CHECKBOX_SIZE or w > cls.MAX_CHECKBOX_SIZE:
            return None
        if h < cls.MIN_CHECKBOX_SIZE or h > cls.MAX_CHECKBOX_SIZE:
            return None

        aspect_ratio = float(w) / h if h > 0 else 0
        if aspect_ratio < (1 - cls.ASPECT_RATIO_THRESHOLD) or aspect_ratio > (1 + cls.ASPECT_RATIO_THRESHOLD):
            return None

        center_x = x + w // 2
        center_y = y + h // 2

        return (center_x, center_y, w, h)

    @classmethod
    def _find_closest_checkbox(
        cls,
        checkboxes: List[Tuple[int, int, int, int]],
        target_x: int,
        target_y: int
    ) -> Optional[Tuple[int, int]]:
        """
        Find the checkbox closest to the target point

        Args:
            checkboxes: List of checkbox info tuples
            target_x: Target X coordinate
            target_y: Target Y coordinate

        Returns:
            (center_x, center_y) of closest checkbox
        """
        if not checkboxes:
            return None

        min_distance = float('inf')
        best_match = None

        for cx, cy, w, h in checkboxes:
            distance = np.sqrt((cx - target_x) ** 2 + (cy - target_y) ** 2)
            if distance < min_distance:
                min_distance = distance
                best_match = (cx, cy)

        return best_match

    @classmethod
    def draw_detection_result(
        cls,
        image_path: str,
        checkbox_center: Tuple[int, int],
        output_path: str,
        box_size: int = 20
    ) -> bool:
        """
        Draw detection result on image for debugging

        Args:
            image_path: Input image path
            checkbox_center: Detected checkbox center
            output_path: Output image path
            box_size: Size of marker box

        Returns:
            Whether successful
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return False

            x, y = checkbox_center
            cv2.rectangle(
                image,
                (x - box_size // 2, y - box_size // 2),
                (x + box_size // 2, y + box_size // 2),
                (0, 255, 0),
                2
            )
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)

            cv2.imwrite(output_path, image)
            return True

        except Exception as e:
            print(f"Failed to draw detection result: {str(e)}")
            return False
