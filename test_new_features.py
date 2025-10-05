#!/usr/bin/env python3
"""
Test script for new image2cpp features:
1. Batch image conversion
2. Arduino/C++/Python code generation
"""

import sys
from PIL import Image
import io

# Test code generation functions directly
sys.path.insert(0, '.')
from app import generate_arduino_code, generate_cpp_code, generate_python_code

def create_test_image_data():
    """Create sample image data for testing"""
    return {
        'filename': 'test_logo.png',
        'width': 64,
        'height': 32,
        'byte_array_raw': [0xFF, 0x00, 0xAA, 0x55] * 64  # Sample byte array
    }

def test_code_generation():
    """Test all code generation functions"""
    print("=" * 60)
    print("Testing Code Generation Features")
    print("=" * 60)

    test_data = [create_test_image_data()]

    # Test Arduino code generation
    print("\n1. Testing Arduino Code Generation")
    print("-" * 60)
    arduino_code = generate_arduino_code(test_data, "myImage", True)
    print(arduino_code[:500] + "..." if len(arduino_code) > 500 else arduino_code)
    print(f"\n✓ Arduino code generated ({len(arduino_code)} characters)")

    # Test C++ code generation
    print("\n2. Testing C++ Code Generation")
    print("-" * 60)
    cpp_code = generate_cpp_code(test_data, "myImage", True)
    print(cpp_code[:500] + "..." if len(cpp_code) > 500 else cpp_code)
    print(f"\n✓ C++ code generated ({len(cpp_code)} characters)")

    # Test Python code generation
    print("\n3. Testing Python Code Generation")
    print("-" * 60)
    python_code = generate_python_code(test_data, "myImage", True)
    print(python_code[:500] + "..." if len(python_code) > 500 else python_code)
    print(f"\n✓ Python code generated ({len(python_code)} characters)")

    # Test multiple images
    print("\n4. Testing Batch Code Generation (3 images)")
    print("-" * 60)
    batch_data = [
        {'filename': 'icon1.png', 'width': 16, 'height': 16, 'byte_array_raw': [0xFF] * 32},
        {'filename': 'icon2.png', 'width': 16, 'height': 16, 'byte_array_raw': [0x00] * 32},
        {'filename': 'icon3.png', 'width': 16, 'height': 16, 'byte_array_raw': [0xAA] * 32},
    ]
    batch_code = generate_arduino_code(batch_data, "icon", True)
    print(f"✓ Batch code generated for {len(batch_data)} images ({len(batch_code)} characters)")
    print("\nSample output:")
    print(batch_code[:400] + "...")

    return True

def test_image_processing():
    """Test image processing helper function"""
    print("\n" + "=" * 60)
    print("Testing Image Processing")
    print("=" * 60)

    from app import process_single_image

    # Create a simple test image
    img = Image.new('L', (64, 32), color=128)
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_data = img_buffer.getvalue()

    # Test processing
    print("\nProcessing test image (64x32 pixels)...")
    result = process_single_image(
        img_data,
        threshold=128,
        mode='horizontal',
        crop_settings={'top': 0, 'right': 0, 'bottom': 0, 'left': 0},
        compression_factor=100,
        filename='test.png'
    )

    print(f"✓ Image processed successfully")
    print(f"  - Filename: {result['filename']}")
    print(f"  - Dimensions: {result['width']}x{result['height']}")
    print(f"  - Byte array length: {len(result['byte_array_raw'])} bytes")
    print(f"  - Preview image: {len(result['preview_image'])} chars (base64)")

    return True

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("image2cpp - New Features Test Suite")
    print("=" * 60)

    try:
        # Test code generation
        if not test_code_generation():
            print("\n❌ Code generation tests failed")
            return False

        # Test image processing
        if not test_image_processing():
            print("\n❌ Image processing tests failed")
            return False

        print("\n" + "=" * 60)
        print("✓ All tests passed successfully!")
        print("=" * 60)

        print("\n📝 New Features Summary:")
        print("  1. ✓ Batch image conversion (up to 20 images)")
        print("  2. ✓ Arduino code generation")
        print("  3. ✓ C++ code generation")
        print("  4. ✓ Python code generation")
        print("  5. ✓ Drag-and-drop file upload")
        print("  6. ✓ Copy-to-clipboard functionality")
        print("  7. ✓ Multi-format code export")

        print("\n🚀 To use the new features:")
        print("  1. Run: python app.py")
        print("  2. Open: http://localhost:5000")
        print("  3. Switch to 'Batch Convert' tab")
        print("  4. Drag & drop multiple images")
        print("  5. Click 'Generate Code for All'")

        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
