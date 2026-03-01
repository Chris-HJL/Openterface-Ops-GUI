"""
Test script for coordinate conversion functionality

This script tests the coordinate converter with various screen resolutions
and validates that the conversion between pixel and HID coordinates works correctly.
"""
import sys
sys.path.insert(0, '.')

from ops_core.coord_converter import (
    CoordinateConverter,
    convert_coordinates,
    pixel_to_hid,
    hid_to_pixel
)
from config import Config


def test_basic_conversion():
    """Test basic pixel to HID conversion"""
    print("=" * 60)
    print("Test 1: Basic Conversion (1920x1080)")
    print("=" * 60)
    
    converter = CoordinateConverter(1920, 1080)
    
    # Test screen center
    pixel_x, pixel_y = 960, 540
    hid_x, hid_y = converter.pixel_to_hid(pixel_x, pixel_y)
    print(f"Screen center: pixel ({pixel_x}, {pixel_y}) -> HID ({hid_x}, {hid_y})")
    assert hid_x == 2048 and hid_y == 2048, f"Expected (2048, 2048), got ({hid_x}, {hid_y})"
    
    # Test corners (allow ±2 pixels for rounding)
    tests = [
        ((0, 0), "Top-left corner"),
        ((1919, 1079), "Bottom-right corner"),
        ((959, 0), "Top center"),
        ((1919, 539), "Middle right"),
    ]
    
    for (px, py), desc in tests:
        hx, hy = converter.pixel_to_hid(px, py)
        print(f"{desc}: pixel ({px}, {py}) -> HID ({hx}, {hy})")
        
        # Verify they are within expected range (±2 pixels for rounding)
        if px == 0:
            assert hx == 0, f"{desc}: Expected hx=0, got {hx}"
        elif px == 1919:
            assert 4092 <= hx <= 4096, f"{desc}: Expected hx≈4096, got {hx}"
        
        if py == 0:
            assert hy == 0, f"{desc}: Expected hy=0, got {hy}"
        elif py == 1079:
            assert 4092 <= hy <= 4096, f"{desc}: Expected hy≈4096, got {hy}"
    
    print("✓ Basic conversion test passed\n")


def test_bidirectional_conversion():
    """Test bidirectional conversion (round-trip)"""
    print("=" * 60)
    print("Test 2: Bidirectional Conversion")
    print("=" * 60)
    
    converter = CoordinateConverter(1920, 1080)
    
    test_points = [
        (0, 0),
        (960, 540),
        (1919, 1079),
        (480, 270),
        (1440, 810),
    ]
    
    for px, py in test_points:
        # Convert pixel -> HID -> pixel
        hx, hy = converter.pixel_to_hid(px, py)
        back_px, back_py = converter.hid_to_pixel(hx, hy)
        
        # Allow small rounding errors (±1 pixel)
        assert abs(back_px - px) <= 1 and abs(back_py - py) <= 1, \
            f"Round-trip error: ({px}, {py}) -> HID ({hx}, {hy}) -> ({back_px}, {back_py})"
        
        print(f"Pixel ({px}, {py}) -> HID ({hx}, {hy}) -> Pixel ({back_px}, {back_py}) ✓")
    
    print("✓ Bidirectional conversion test passed\n")


def test_different_resolutions():
    """Test conversion with different screen resolutions"""
    print("=" * 60)
    print("Test 3: Different Screen Resolutions")
    print("=" * 60)
    
    # Screen center should always map to HID (2048, 2048)
    resolutions = [
        (1280, 720, 640, 360),     # 720p
        (1920, 1080, 960, 540),    # 1080p
        (2560, 1440, 1280, 720),   # 1440p
        (3840, 2160, 1920, 1080),  # 4K
    ]
    
    for width, height, center_x, center_y in resolutions:
        converter = CoordinateConverter(width, height)
        hx, hy = converter.pixel_to_hid(center_x, center_y)
        print(f"{width}x{height}: center pixel ({center_x}, {center_y}) -> HID ({hx}, {hy})")
        assert hx == 2048 and hy == 2048, f"Center should map to (2048, 2048), got ({hx}, {hy})"
    
    print("✓ Different resolutions test passed\n")


def test_function_interface():
    """Test standalone conversion functions"""
    print("=" * 60)
    print("Test 4: Function Interface")
    print("=" * 60)
    
    # Test convert_coordinates function
    hx, hy = convert_coordinates(960, 540, "pixel", "hid", 1920, 1080)
    print(f"convert_coordinates(960, 540, 'pixel', 'hid', 1920, 1080) = ({hx}, {hy})")
    assert hx == 2048 and hy == 2048
    
    # Test global functions
    px, py = hid_to_pixel(2048, 2048)
    print(f"hid_to_pixel(2048, 2048) with config resolution ({Config.SCREEN_WIDTH}x{Config.SCREEN_HEIGHT}) = ({px}, {py})")
    
    print("✓ Function interface test passed\n")


def test_config_integration():
    """Test integration with Config"""
    print("=" * 60)
    print("Test 5: Config Integration")
    print("=" * 60)

    print(f"Config.SCREEN_WIDTH = {Config.SCREEN_WIDTH}")
    print(f"Config.SCREEN_HEIGHT = {Config.SCREEN_HEIGHT}")
    print(f"Config.COORD_SYSTEM = {Config.COORD_SYSTEM}")

    # Test that converter can be initialized without parameters
    # and will use Config defaults when needed
    converter = CoordinateConverter()  # No parameters
    assert converter.screen_width is None, "Default converter should have None resolution"
    assert converter.screen_height is None, "Default converter should have None resolution"
    
    # Test that conversion still works (uses config fallback)
    hx, hy = converter.pixel_to_hid(960, 540)
    print(f"Conversion with config fallback: pixel (960, 540) -> HID ({hx}, {hy})")
    # Should use 1920x1080 fallback, so center maps to (2048, 2048)
    assert hx == 2048 and hy == 2048, f"Expected (2048, 2048) with fallback, got ({hx}, {hy})"
    
    print("✓ Config integration test passed (fallback mechanism works)\n")


def test_edge_cases():
    """Test edge cases and boundary conditions"""
    print("=" * 60)
    print("Test 6: Edge Cases")
    print("=" * 60)
    
    converter = CoordinateConverter(1920, 1080)
    
    # Test out-of-bounds coordinates (should be clamped)
    hx, hy = converter.pixel_to_hid(-100, -100)
    print(f"Negative pixel (-100, -100) -> clamped to HID ({hx}, {hy})")
    assert hx == 0 and hy == 0
    
    hx, hy = converter.pixel_to_hid(5000, 5000)
    print(f"Large pixel (5000, 5000) -> clamped to HID ({hx}, {hy})")
    assert hx == 4096 and hy == 4096
    
    # Test exact boundaries
    hx, hy = converter.pixel_to_hid(0, 0)
    print(f"Boundary pixel (0, 0) -> HID ({hx}, {hy})")
    assert hx == 0 and hy == 0
    
    hx, hy = converter.pixel_to_hid(1920, 1080)
    print(f"Boundary pixel (1920, 1080) -> HID ({hx}, {hy})")
    assert hx == 4096 and hy == 4096
    
    print("✓ Edge cases test passed\n")


def test_load_resolution_from_image():
    """Test loading resolution from image file"""
    print("=" * 60)
    print("Test 8: Load Resolution from Image")
    print("=" * 60)
    
    # Create a test image
    from PIL import Image
    
    test_image_path = "./test_screen_temp.jpg"
    test_resolution = (1920, 1080)
    
    # Create a simple test image
    test_img = Image.new('RGB', test_resolution, color='blue')
    test_img.save(test_image_path)
    print(f"Created test image: {test_image_path} with resolution {test_resolution}")
    
    try:
        # Test loading resolution from image
        converter = CoordinateConverter()  # Initialize without parameters
        loaded_resolution = converter.load_resolution_from_image(test_image_path)
        
        print(f"Loaded resolution: {loaded_resolution}")
        assert loaded_resolution == test_resolution, f"Expected {test_resolution}, got {loaded_resolution}"
        
        # Test conversion with loaded resolution
        hid_x, hid_y = converter.pixel_to_hid(960, 540)
        print(f"Conversion test: pixel (960, 540) -> HID ({hid_x}, {hid_y})")
        assert hid_x == 2048 and hid_y == 2048, f"Expected (2048, 2048), got ({hid_x}, {hid_y})"
        
        print("✓ Load resolution from image test passed\n")
        
    finally:
        # Cleanup
        import os
        if os.path.exists(test_image_path):
            os.remove(test_image_path)


def test_accuracy():
    """Test conversion accuracy"""
    print("=" * 60)
    print("Test 9: Conversion Accuracy")
    print("=" * 60)
    
    converter = CoordinateConverter(1920, 1080)
    
    # Test multiple points for accuracy
    total_error = 0
    test_count = 0
    
    for x in range(0, 1920, 192):  # Every 10%
        for y in range(0, 1080, 108):
            hx, hy = converter.pixel_to_hid(x, y)
            back_x, back_y = converter.hid_to_pixel(hx, hy)
            
            error = abs(back_x - x) + abs(back_y - y)
            total_error += error
            test_count += 1
            
            if error > 2:
                print(f"Warning: Large error at ({x}, {y}): {error} pixels")
    
    avg_error = total_error / test_count
    print(f"Average round-trip error: {avg_error:.3f} pixels ({test_count} test points)")
    assert avg_error < 1.5, f"Average error too high: {avg_error}"
    
    print("✓ Accuracy test passed\n")


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "Coordinate Conversion Tests" + " " * 16 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    try:
        test_basic_conversion()
        test_bidirectional_conversion()
        test_different_resolutions()
        test_function_interface()
        test_config_integration()
        test_edge_cases()
        test_load_resolution_from_image()
        test_accuracy()
        
        print("╔" + "═" * 58 + "╗")
        print("║" + " " * 20 + "ALL TESTS PASSED ✓" + " " * 21 + "║")
        print("╚" + "═" * 58 + "╝")
        print()
        
        # Print usage example
        print("=" * 60)
        print("Usage Example:")
        print("=" * 60)
        print("""
# In your code (recommended):
from ops_core.coord_converter import CoordinateConverter

# Create converter (resolution will be loaded from image)
converter = CoordinateConverter()

# Load resolution from image automatically
image_path = "screenshot.jpg"
converter.load_resolution_from_image(image_path)

# Convert pixel coordinates from UI-Model to HID coordinates
pixel_x, pixel_y = 960, 540  # From UI-Model
hid_x, hid_y = converter.pixel_to_hid(pixel_x, pixel_y)

# Use HID coordinates in TCP command
script_command = f'Send "{{Click {hid_x}, {hid_y}}}"'

# No need to configure SCREEN_WIDTH/SCREEN_HEIGHT!
        """)
        print()
        
        return 0
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}\n")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
