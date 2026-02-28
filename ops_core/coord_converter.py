"""
Coordinate conversion module
Provides bidirectional conversion between pixel coordinates and HID normalized coordinates

This module solves the coordinate system mismatch issue:
- UI-Model returns pixel coordinates (based on screen resolution)
- Openterface KVM expects HID coordinates (0-4096 normalized range)
"""
from typing import Tuple, Optional
from PIL import Image
from config import Config


class CoordinateConverter:
    """
    Coordinate converter for pixel ↔ HID coordinate transformation
    
    HID (Human Interface Device) coordinates use a normalized 12-bit range (0-4096)
    that works across different screen resolutions. This allows the same coordinate
    values to represent the same relative position regardless of actual resolution.
    
    Example:
        - Screen center on 1920×1080: pixel (960, 540) → HID (2048, 2048)
        - Screen center on 3840×2160: pixel (1920, 1080) → HID (2048, 2048)
    
    Usage:
        converter = CoordinateConverter()
        
        # Method 1: Load resolution from image
        converter.load_resolution_from_image(image_path)
        hid_x, hid_y = converter.pixel_to_hid(960, 540)  # Returns (2048, 2048)
        
        # Method 2: Use explicit resolution
        converter.update_resolution(1920, 1080)
        pixel_x, pixel_y = converter.hid_to_pixel(2048, 2048)  # Returns (960, 540)
    """

    # HID coordinate constants (12-bit precision per USB HID specification)
    HID_MAX_X: int = 4096
    HID_MAX_Y: int = 4096

    def __init__(self, screen_width: int = None, screen_height: int = None):
        """
        Initialize coordinate converter

        Args:
            screen_width: Screen width in pixels (default: will be loaded from image)
            screen_height: Screen height in pixels (default: will be loaded from image)
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self._resolution_loaded_from_image = False

    def load_resolution_from_image(self, image_path: str) -> Tuple[int, int]:
        """
        Load screen resolution from image file
        
        This is the recommended method - automatically detects the resolution
        from the screenshot image, ensuring accurate coordinate conversion.

        Args:
            image_path: Path to the screenshot image

        Returns:
            Tuple (width, height) of the image resolution

        Raises:
            FileNotFoundError: If image file doesn't exist
            IOError: If image cannot be opened
        """
        try:
            with Image.open(image_path) as img:
                self.screen_width, self.screen_height = img.size
                self._resolution_loaded_from_image = True
                print(f"[CoordinateConverter] Resolution loaded from image: {self.screen_width}x{self.screen_height}")
                return (self.screen_width, self.screen_height)
        except Exception as e:
            print(f"[CoordinateConverter] Error loading resolution from image {image_path}: {e}")
            # Fallback to defaults if available
            if self.screen_width is None or self.screen_height is None:
                self.screen_width = Config.SCREEN_WIDTH
                self.screen_height = Config.SCREEN_HEIGHT
                print(f"[CoordinateConverter] Using fallback resolution: {self.screen_width}x{self.screen_height}")
            raise

    def update_resolution(self, width: int, height: int):
        """
        Update screen resolution manually

        Args:
            width: New screen width in pixels
            height: New screen height in pixels
        """
        old_width, old_height = self.screen_width, self.screen_height
        self.screen_width = width
        self.screen_height = height
        self._resolution_loaded_from_image = False
        print(f"[CoordinateConverter] Resolution updated: {old_width}x{old_height} → {width}x{height}")

    def pixel_to_hid(self, x: int, y: int) -> Tuple[int, int]:
        """
        Convert pixel coordinates to HID normalized coordinates

        Formula:
            HID_X = (pixel_x × 4096) / screen_width
            HID_Y = (pixel_y × 4096) / screen_height

        Args:
            x: X coordinate in pixels (0 to screen_width)
            y: Y coordinate in pixels (0 to screen_height)

        Returns:
            Tuple (hid_x, hid_y) in range 0-4096, clamped to valid range

        Note:
            If resolution is not set, will use Config defaults.
            Best practice: Call load_resolution_from_image() before using this method.

        Example:
            >>> converter = CoordinateConverter()
            >>> converter.load_resolution_from_image("screenshot.jpg")
            >>> converter.pixel_to_hid(960, 540)
            (2048, 2048)  # Screen center
        """
        # Use config defaults if resolution not set
        width = self.screen_width if self.screen_width is not None else Config.SCREEN_WIDTH
        height = self.screen_height if self.screen_height is not None else Config.SCREEN_HEIGHT

        if width <= 0 or height <= 0:
            print(f"[CoordinateConverter] Warning: Invalid resolution {width}x{height}, using defaults")
            width = Config.SCREEN_WIDTH
            height = Config.SCREEN_HEIGHT

        # Apply linear transformation with normalization
        hid_x = int(round(x * self.HID_MAX_X / width))
        hid_y = int(round(y * self.HID_MAX_Y / height))

        # Clamp to valid HID range (0-4096)
        hid_x = max(0, min(self.HID_MAX_X, hid_x))
        hid_y = max(0, min(self.HID_MAX_Y, hid_y))

        return (hid_x, hid_y)

    def hid_to_pixel(self, hid_x: int, hid_y: int) -> Tuple[int, int]:
        """
        Convert HID normalized coordinates to pixel coordinates

        Formula:
            pixel_x = (hid_x × screen_width) / 4096
            pixel_y = (hid_y × screen_height) / 4096

        Args:
            hid_x: X coordinate in HID range (0-4096)
            hid_y: Y coordinate in HID range (0-4096)

        Returns:
            Tuple (pixel_x, pixel_y) in screen pixel coordinates

        Note:
            If resolution is not set, will use Config defaults.
            Best practice: Call load_resolution_from_image() before using this method.

        Example:
            >>> converter = CoordinateConverter()
            >>> converter.load_resolution_from_image("screenshot.jpg")
            >>> converter.hid_to_pixel(2048, 2048)
            (960, 540)  # Screen center
        """
        # Use config defaults if resolution not set
        width = self.screen_width if self.screen_width is not None else Config.SCREEN_WIDTH
        height = self.screen_height if self.screen_height is not None else Config.SCREEN_HEIGHT

        if width <= 0 or height <= 0:
            print(f"[CoordinateConverter] Warning: Invalid resolution {width}x{height}, using defaults")
            width = Config.SCREEN_WIDTH
            height = Config.SCREEN_HEIGHT

        # Apply inverse transformation
        pixel_x = int(round(hid_x * width / self.HID_MAX_X))
        pixel_y = int(round(hid_y * height / self.HID_MAX_Y))

        # Clamp to valid pixel range
        pixel_x = max(0, min(width - 1, pixel_x))
        pixel_y = max(0, min(height - 1, pixel_y))

        return (pixel_x, pixel_y)


def convert_coordinates(
    x: int,
    y: int,
    from_system: str = "pixel",
    to_system: str = "hid",
    screen_width: int = None,
    screen_height: int = None
) -> Tuple[int, int]:
    """
    Convenience function for coordinate conversion without creating converter instance

    Args:
        x: X coordinate
        y: Y coordinate
        from_system: Source coordinate system ("pixel" or "hid")
        to_system: Target coordinate system ("pixel" or "hid")
        screen_width: Screen width (default from config)
        screen_height: Screen height (default from config)

    Returns:
        Converted (x, y) coordinates

    Example:
        >>> convert_coordinates(960, 540, "pixel", "hid", 1920, 1080)
        (2048, 2048)
    """
    converter = CoordinateConverter(screen_width, screen_height)

    if from_system == to_system:
        return (x, y)

    if from_system == "pixel" and to_system == "hid":
        return converter.pixel_to_hid(x, y)
    elif from_system == "hid" and to_system == "pixel":
        return converter.hid_to_pixel(x, y)
    else:
        raise ValueError(f"Unsupported coordinate system conversion: {from_system} -> {to_system}")


# Global converter instance for backward compatibility
_default_converter = CoordinateConverter()


def get_global_converter() -> CoordinateConverter:
    """Get the default global converter instance"""
    return _default_converter


# Backward compatibility functions (maintain existing API)
def pixel_to_hid(x: int, y: int) -> Tuple[int, int]:
    """Convert pixel coordinates to HID coordinates using global converter"""
    return _default_converter.pixel_to_hid(x, y)


def hid_to_pixel(x: int, y: int) -> Tuple[int, int]:
    """Convert HID coordinates to pixel coordinates using global converter"""
    return _default_converter.hid_to_pixel(x, y)
