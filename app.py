from dotenv import load_dotenv, set_key
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError
import io
import base64
import numpy as np
import requests
import os
import logging
import traceback
from functools import wraps
from typing import Dict, Any, Tuple, Optional

app = Flask(__name__)
CORS(app)

load_dotenv()

# Configure comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('image2cpp.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Error handling constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_IMAGE_DIMENSION = 4096
ALLOWED_IMAGE_FORMATS = {'PNG', 'JPEG', 'JPG', 'BMP', 'GIF', 'WEBP'}
REQUEST_TIMEOUT = 10  # seconds

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class ImageProcessingError(Exception):
    """Custom exception for image processing errors"""
    pass

def error_handler(f):
    """Decorator for comprehensive error handling"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValidationError as e:
            logger.warning(f"Validation error in {f.__name__}: {str(e)}")
            return jsonify({
                "error": "Validation Error",
                "message": str(e),
                "type": "validation"
            }), 400
        except ImageProcessingError as e:
            logger.error(f"Image processing error in {f.__name__}: {str(e)}")
            return jsonify({
                "error": "Image Processing Error",
                "message": str(e),
                "type": "processing"
            }), 422
        except FileNotFoundError as e:
            logger.error(f"File not found in {f.__name__}: {str(e)}")
            return jsonify({
                "error": "File Not Found",
                "message": "The requested file could not be found",
                "type": "file_error"
            }), 404
        except PermissionError as e:
            logger.error(f"Permission error in {f.__name__}: {str(e)}")
            return jsonify({
                "error": "Permission Denied",
                "message": "Insufficient permissions to access the resource",
                "type": "permission_error"
            }), 403
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error in {f.__name__}: {str(e)}")
            return jsonify({
                "error": "Network Error",
                "message": "Failed to fetch image from URL",
                "details": str(e),
                "type": "network"
            }), 502
        except Exception as e:
            logger.error(f"Unexpected error in {f.__name__}: {str(e)}\n{traceback.format_exc()}")
            return jsonify({
                "error": "Internal Server Error",
                "message": "An unexpected error occurred",
                "type": "internal",
                "details": str(e) if app.debug else None
            }), 500
    return decorated_function

def validate_crop_settings(settings: Dict[str, Any]) -> None:
    """Validate crop settings"""
    required_keys = ['top', 'right', 'bottom', 'left']

    for key in required_keys:
        if key not in settings:
            raise ValidationError(f"Missing required crop setting: {key}")

        try:
            value = int(settings[key])
            if value < 0:
                raise ValidationError(f"Crop setting '{key}' must be non-negative, got {value}")
            if value > 10000:
                raise ValidationError(f"Crop setting '{key}' is too large, got {value}")
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Invalid value for crop setting '{key}': {settings[key]}")

def validate_threshold(threshold: int) -> None:
    """Validate threshold value"""
    if not isinstance(threshold, int) or threshold < 0 or threshold > 255:
        raise ValidationError(f"Threshold must be an integer between 0 and 255, got {threshold}")

def validate_compression_factor(factor: int) -> None:
    """Validate compression factor"""
    if not isinstance(factor, int) or factor < 1 or factor > 100:
        raise ValidationError(f"Compression factor must be an integer between 1 and 100, got {factor}")

def validate_image_dimensions(width: int, height: int) -> None:
    """Validate image dimensions"""
    if width <= 0 or height <= 0:
        raise ValidationError(f"Image dimensions must be positive, got {width}x{height}")

    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise ValidationError(f"Image dimensions exceed maximum allowed ({MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION})")

def validate_image_url(url: str) -> None:
    """Validate image URL"""
    if not url or not isinstance(url, str):
        raise ValidationError("Invalid URL provided")

    if not url.startswith(('http://', 'https://')):
        raise ValidationError("URL must start with http:// or https://")

    if len(url) > 2048:
        raise ValidationError("URL is too long")

def load_crop_settings() -> Dict[str, int]:
    """Load crop settings from environment with error handling"""
    try:
        load_dotenv()
        settings = {
            'top': int(os.getenv('CROP_TOP', 0)),
            'right': int(os.getenv('CROP_RIGHT', 0)),
            'bottom': int(os.getenv('CROP_BOTTOM', 0)),
            'left': int(os.getenv('CROP_LEFT', 0))
        }
        validate_crop_settings(settings)
        logger.debug(f"Loaded crop settings: {settings}")
        return settings
    except ValueError as e:
        logger.error(f"Error parsing crop settings: {e}")
        return {'top': 0, 'right': 0, 'bottom': 0, 'left': 0}
    except Exception as e:
        logger.error(f"Unexpected error loading crop settings: {e}")
        return {'top': 0, 'right': 0, 'bottom': 0, 'left': 0}

def load_compression_factor() -> int:
    """Load compression factor from environment with error handling"""
    try:
        load_dotenv()
        compression_factor = int(os.getenv('COMPRESSION_FACTOR', 100))
        validate_compression_factor(compression_factor)
        logger.debug(f"Loaded compression factor: {compression_factor}%")
        return compression_factor
    except ValueError as e:
        logger.error(f"Error parsing compression factor: {e}")
        return 100
    except Exception as e:
        logger.error(f"Unexpected error loading compression factor: {e}")
        return 100

def save_crop_settings(settings: Dict[str, Any]) -> None:
    """Save crop settings with validation"""
    try:
        validate_crop_settings(settings)
        logger.debug(f"Saving crop settings: {settings}")

        # Create .env file if it doesn't exist
        if not os.path.exists('.env'):
            with open('.env', 'w') as f:
                f.write('')

        for key, value in settings.items():
            set_key('.env', f'CROP_{key.upper()}', str(value))
        load_dotenv()  # Reload environment variables
    except Exception as e:
        logger.error(f"Error saving crop settings: {e}")
        raise ImageProcessingError(f"Failed to save crop settings: {str(e)}")

def apply_crop(image: Image.Image, crop_settings: Dict[str, int]) -> Image.Image:
    """Apply crop to image with validation"""
    try:
        width, height = image.size
        left = crop_settings['left']
        top = crop_settings['top']
        right = width - crop_settings['right']
        bottom = height - crop_settings['bottom']

        # Validate crop boundaries
        if left >= right or top >= bottom:
            raise ValidationError(f"Invalid crop settings: result would be empty or negative")

        if left < 0 or top < 0 or right > width or bottom > height:
            raise ValidationError(f"Crop settings exceed image boundaries")

        logger.debug(f"Applying crop: left={left}, top={top}, right={right}, bottom={bottom}")
        return image.crop((left, top, right, bottom))
    except Exception as e:
        logger.error(f"Error applying crop: {e}")
        raise ImageProcessingError(f"Failed to crop image: {str(e)}")

def compress_image(image: Image.Image, compression_factor: int) -> Image.Image:
    """Compress image with validation"""
    try:
        validate_compression_factor(compression_factor)

        if compression_factor < 100:
            width, height = image.size
            new_width = max(1, int(width * (compression_factor / 100)))
            new_height = max(1, int(height * (compression_factor / 100)))

            validate_image_dimensions(new_width, new_height)

            logger.debug(f"Compressing image: new dimensions=({new_width}, {new_height})")
            return image.resize((new_width, new_height), Image.LANCZOS)
        return image
    except Exception as e:
        logger.error(f"Error compressing image: {e}")
        raise ImageProcessingError(f"Failed to compress image: {str(e)}")

def image_to_byte_array(image, threshold=128, mode='horizontal'):
    width, height = image.size
    pixels = list(image.getdata())
    byte_array = []
    byte = 0
    bit_count = 0

    for y in range(height):
        for x in range(width):
            if mode == 'horizontal':
                pixel = pixels[y * width + x]
            else:  # vertical
                pixel = pixels[x * height + y]

            if isinstance(pixel, tuple):
                pixel = sum(pixel[:3]) // 3  # Convert RGB to grayscale

            if pixel > threshold:
                byte |= (1 << (7 - bit_count))

            bit_count += 1
            if bit_count == 8:
                byte_array.append(byte)
                byte = 0
                bit_count = 0

    # Add any remaining bits
    if bit_count > 0:
        byte_array.append(byte)

    return byte_array

def byte_array_to_image(byte_array, width, height, mode='horizontal'):
    img = Image.new('1', (width, height))
    pixels = img.load()

    byte_index = 0
    bit_index = 0

    for y in range(height):
        for x in range(width):
            if mode == 'horizontal':
                if byte_array[byte_index] & (1 << (7 - bit_index)):
                    pixels[x, y] = 1
                else:
                    pixels[x, y] = 0
            else:  # vertical
                if byte_array[byte_index] & (1 << (7 - bit_index)):
                    pixels[y, x] = 1
                else:
                    pixels[y, x] = 0

            bit_index += 1
            if bit_index == 8:
                byte_index += 1
                bit_index = 0

    return img


@app.route('/')
def index():
    crop_settings = load_crop_settings()
    return render_template('index.html', initial_crop_settings=crop_settings)

@app.route('/save_crop', methods=['POST'])
def save_crop():
    crop_settings = request.json
    logging.debug(f"Received crop settings: {crop_settings}")
    save_crop_settings(crop_settings)
    return jsonify({'status': 'success', 'crop_settings': crop_settings})

def process_single_image(image_data, threshold, mode, crop_settings, compression_factor, filename="image"):
    """
    Process a single image and return conversion results.
    Helper function used by both single and batch conversion endpoints.
    """
    image = Image.open(io.BytesIO(image_data)).convert('L')

    logger.debug(f"Original image size: {image.size}")
    image = apply_crop(image, crop_settings)
    logger.debug(f"Cropped image size: {image.size}")

    image = compress_image(image, compression_factor)
    logger.debug(f"Compressed image size: {image.size}")

    byte_array = image_to_byte_array(image, threshold, mode)
    formatted_array = ', '.join(f'0x{byte:02X}' for byte in byte_array)

    preview_buffer = io.BytesIO()
    image.save(preview_buffer, format='PNG')
    preview_base64 = base64.b64encode(preview_buffer.getvalue()).decode('utf-8')

    return {
        "filename": filename,
        "byte_array": formatted_array,
        "byte_array_raw": byte_array,
        "width": image.width,
        "height": image.height,
        "preview_image": preview_base64,
        "crop_settings": crop_settings
    }

@app.route('/convert', methods=['POST'])
def convert_image():
    threshold = int(request.form.get('threshold', 128))
    mode = request.form.get('mode', 'horizontal')
    crop_settings = load_crop_settings()
    compression_factor = load_compression_factor()

    try:
        if 'image' in request.files:
            image_file = request.files['image']
            image_data = image_file.read()
            filename = image_file.filename
        elif 'image_url' in request.form:
            image_url = request.form['image_url']
            response = requests.get(image_url)
            image_data = response.content
            filename = "url_image"
        else:
            return jsonify({"error": "No image file or URL provided"}), 400

        result = process_single_image(image_data, threshold, mode, crop_settings, compression_factor, filename)
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error in convert_image: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/batch_convert', methods=['POST'])
@error_handler
def batch_convert():
    """
    Convert multiple images at once.
    Accepts multiple image files and returns conversion results for all.
    """
    threshold = int(request.form.get('threshold', 128))
    mode = request.form.get('mode', 'horizontal')
    crop_settings = load_crop_settings()
    compression_factor = load_compression_factor()

    if 'images' not in request.files:
        raise ValidationError("No images provided for batch conversion")

    images = request.files.getlist('images')

    if len(images) == 0:
        raise ValidationError("No images provided for batch conversion")

    if len(images) > 20:
        raise ValidationError("Maximum 20 images allowed per batch")

    results = []
    errors = []

    for idx, image_file in enumerate(images):
        try:
            if not image_file.filename:
                continue

            logger.info(f"Processing image {idx+1}/{len(images)}: {image_file.filename}")

            image_data = image_file.read()
            result = process_single_image(
                image_data,
                threshold,
                mode,
                crop_settings,
                compression_factor,
                image_file.filename
            )
            results.append(result)

        except Exception as e:
            logger.error(f"Error processing {image_file.filename}: {str(e)}")
            errors.append({
                "filename": image_file.filename,
                "error": str(e)
            })

    return jsonify({
        "results": results,
        "errors": errors,
        "total_processed": len(results),
        "total_errors": len(errors)
    })

@app.route('/generate_code', methods=['POST'])
@error_handler
def generate_code():
    """
    Generate Arduino/C++ code from conversion results.
    Accepts byte array data and generates ready-to-use code.
    """
    data = request.json

    if not data:
        raise ValidationError("No data provided")

    # Extract parameters
    images_data = data.get('images', [])
    code_style = data.get('code_style', 'arduino')  # arduino, cpp, or python
    variable_prefix = data.get('variable_prefix', 'image')
    include_dimensions = data.get('include_dimensions', True)

    if not images_data:
        raise ValidationError("No image data provided")

    # Generate code based on style
    if code_style == 'arduino':
        code = generate_arduino_code(images_data, variable_prefix, include_dimensions)
    elif code_style == 'cpp':
        code = generate_cpp_code(images_data, variable_prefix, include_dimensions)
    elif code_style == 'python':
        code = generate_python_code(images_data, variable_prefix, include_dimensions)
    else:
        raise ValidationError(f"Unsupported code style: {code_style}")

    return jsonify({
        "code": code,
        "code_style": code_style,
        "image_count": len(images_data)
    })

def generate_arduino_code(images_data, variable_prefix, include_dimensions):
    """Generate Arduino-compatible C++ code"""
    code_lines = [
        "// Generated by image2cpp",
        "// https://github.com/javl/image2cpp",
        "",
    ]

    for idx, img_data in enumerate(images_data):
        filename = img_data.get('filename', f'image_{idx}')
        # Clean filename for variable name
        var_name = filename.split('.')[0].replace('-', '_').replace(' ', '_')
        var_name = f"{variable_prefix}_{var_name}" if variable_prefix else var_name

        width = img_data.get('width', 0)
        height = img_data.get('height', 0)
        byte_array = img_data.get('byte_array_raw', [])

        if include_dimensions:
            code_lines.append(f"// '{filename}', {width}x{height}px")

        code_lines.append(f"const unsigned char {var_name}[] PROGMEM = {{")

        # Format byte array with 16 bytes per line
        for i in range(0, len(byte_array), 16):
            chunk = byte_array[i:i+16]
            formatted_chunk = ', '.join(f'0x{byte:02X}' for byte in chunk)
            code_lines.append(f"  {formatted_chunk}" + ("," if i + 16 < len(byte_array) else ""))

        code_lines.append("};")

        if include_dimensions:
            code_lines.append(f"const int {var_name}_width = {width};")
            code_lines.append(f"const int {var_name}_height = {height};")

        code_lines.append("")

    # Add example usage
    code_lines.extend([
        "// Example usage with Adafruit GFX library:",
        "// display.drawBitmap(x, y, " + f"{variable_prefix}_yourimage" + ", width, height, color);",
        ""
    ])

    return '\n'.join(code_lines)

def generate_cpp_code(images_data, variable_prefix, include_dimensions):
    """Generate C++ code"""
    code_lines = [
        "// Generated by image2cpp",
        "// https://github.com/javl/image2cpp",
        "",
        "#include <cstdint>",
        "",
    ]

    for idx, img_data in enumerate(images_data):
        filename = img_data.get('filename', f'image_{idx}')
        var_name = filename.split('.')[0].replace('-', '_').replace(' ', '_')
        var_name = f"{variable_prefix}_{var_name}" if variable_prefix else var_name

        width = img_data.get('width', 0)
        height = img_data.get('height', 0)
        byte_array = img_data.get('byte_array_raw', [])

        if include_dimensions:
            code_lines.append(f"// '{filename}', {width}x{height}px")

        code_lines.append(f"const uint8_t {var_name}[] = {{")

        for i in range(0, len(byte_array), 16):
            chunk = byte_array[i:i+16]
            formatted_chunk = ', '.join(f'0x{byte:02X}' for byte in chunk)
            code_lines.append(f"  {formatted_chunk}" + ("," if i + 16 < len(byte_array) else ""))

        code_lines.append("};")

        if include_dimensions:
            code_lines.append(f"const int {var_name}_width = {width};")
            code_lines.append(f"const int {var_name}_height = {height};")

        code_lines.append("")

    return '\n'.join(code_lines)

def generate_python_code(images_data, variable_prefix, include_dimensions):
    """Generate Python code"""
    code_lines = [
        "# Generated by image2cpp",
        "# https://github.com/javl/image2cpp",
        "",
    ]

    for idx, img_data in enumerate(images_data):
        filename = img_data.get('filename', f'image_{idx}')
        var_name = filename.split('.')[0].replace('-', '_').replace(' ', '_')
        var_name = f"{variable_prefix}_{var_name}" if variable_prefix else var_name

        width = img_data.get('width', 0)
        height = img_data.get('height', 0)
        byte_array = img_data.get('byte_array_raw', [])

        if include_dimensions:
            code_lines.append(f"# '{filename}', {width}x{height}px")

        code_lines.append(f"{var_name} = [")

        for i in range(0, len(byte_array), 16):
            chunk = byte_array[i:i+16]
            formatted_chunk = ', '.join(f'0x{byte:02X}' for byte in chunk)
            code_lines.append(f"    {formatted_chunk}" + ("," if i + 16 < len(byte_array) else ""))

        code_lines.append("]")

        if include_dimensions:
            code_lines.append(f"{var_name}_width = {width}")
            code_lines.append(f"{var_name}_height = {height}")

        code_lines.append("")

    return '\n'.join(code_lines)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
