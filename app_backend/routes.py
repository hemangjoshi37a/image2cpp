"""
Flask Routes Module
Defines all API endpoints and request handlers
"""

import logging
import io
import requests
from functools import wraps
from flask import Blueprint, request, render_template, jsonify
from PIL import Image, UnidentifiedImageError

from .config import config, REQUEST_TIMEOUT
from .exceptions import ValidationError, ImageProcessingError
from .validators import (validate_threshold, validate_conversion_mode,
                        validate_code_style, validate_batch_size)
from .image_processor import ImageProcessor
from .code_generator import CodeGenerator

logger = logging.getLogger(__name__)

# Create Blueprint for routes
api = Blueprint('api', __name__)


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
        except UnidentifiedImageError as e:
            logger.error(f"Invalid image format in {f.__name__}: {str(e)}")
            return jsonify({
                "error": "Invalid Image",
                "message": "The provided file is not a valid image",
                "type": "image_error"
            }), 400
        except Exception as e:
            logger.error(f"Unexpected error in {f.__name__}: {str(e)}")
            return jsonify({
                "error": "Internal Server Error",
                "message": "An unexpected error occurred",
                "type": "internal",
                "details": str(e)
            }), 500
    return decorated_function


@api.route('/')
def index():
    """Render main application page"""
    crop_settings = config.get_crop_settings()
    return render_template('index.html', initial_crop_settings=crop_settings)


@api.route('/save_crop', methods=['POST'])
@error_handler
def save_crop():
    """
    Save crop settings to configuration

    Request JSON:
        {
            "top": int,
            "right": int,
            "bottom": int,
            "left": int
        }

    Returns:
        JSON response with saved settings
    """
    crop_settings = request.json
    logger.debug(f"Received crop settings: {crop_settings}")

    # Save settings
    config.save_crop_settings(crop_settings)

    return jsonify({
        'status': 'success',
        'crop_settings': crop_settings
    })


@api.route('/convert', methods=['POST'])
@error_handler
def convert_image():
    """
    Convert single image to byte array

    Request Form:
        - image: Image file (optional if image_url provided)
        - image_url: Image URL (optional if image file provided)
        - threshold: int (0-255, default 128)
        - mode: str ('horizontal' or 'vertical', default 'horizontal')

    Returns:
        JSON with conversion results including byte array and preview
    """
    # Parse parameters
    threshold = int(request.form.get('threshold', 128))
    mode = request.form.get('mode', 'horizontal')

    # Validate parameters
    validate_threshold(threshold)
    validate_conversion_mode(mode)

    # Load settings
    crop_settings = config.get_crop_settings()
    compression_factor = config.get_compression_factor()

    # Get image data
    if 'image' in request.files:
        image_file = request.files['image']
        image_data = image_file.read()
        filename = image_file.filename
    elif 'image_url' in request.form:
        image_url = request.form['image_url']
        response = requests.get(image_url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        image_data = response.content
        filename = "url_image"
    else:
        raise ValidationError("No image file or URL provided")

    # Process image
    result = ImageProcessor.process_image(
        image_data, threshold, mode, crop_settings, compression_factor, filename
    )

    return jsonify(result)


@api.route('/batch_convert', methods=['POST'])
@error_handler
def batch_convert():
    """
    Convert multiple images at once

    Request Form:
        - images: Multiple image files
        - threshold: int (0-255, default 128)
        - mode: str ('horizontal' or 'vertical', default 'horizontal')

    Returns:
        JSON with array of conversion results and any errors
    """
    # Parse parameters
    threshold = int(request.form.get('threshold', 128))
    mode = request.form.get('mode', 'horizontal')

    # Validate parameters
    validate_threshold(threshold)
    validate_conversion_mode(mode)

    # Get images
    if 'images' not in request.files:
        raise ValidationError("No images provided for batch conversion")

    images = request.files.getlist('images')
    validate_batch_size(len(images))

    # Load settings
    crop_settings = config.get_crop_settings()
    compression_factor = config.get_compression_factor()

    results = []
    errors = []

    # Process each image
    for idx, image_file in enumerate(images):
        try:
            if not image_file.filename:
                continue

            logger.info(f"Processing image {idx+1}/{len(images)}: {image_file.filename}")

            image_data = image_file.read()
            result = ImageProcessor.process_image(
                image_data, threshold, mode, crop_settings,
                compression_factor, image_file.filename
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


@api.route('/generate_code', methods=['POST'])
@error_handler
def generate_code():
    """
    Generate Arduino/C++/Python code from conversion results

    Request JSON:
        {
            "images": [array of image conversion results],
            "code_style": str ('arduino', 'cpp', or 'python'),
            "variable_prefix": str (default 'image'),
            "include_dimensions": bool (default true)
        }

    Returns:
        JSON with generated code
    """
    data = request.json

    if not data:
        raise ValidationError("No data provided")

    # Extract parameters
    images_data = data.get('images', [])
    code_style = data.get('code_style', 'arduino')
    variable_prefix = data.get('variable_prefix', 'image')
    include_dimensions = data.get('include_dimensions', True)

    if not images_data:
        raise ValidationError("No image data provided")

    # Validate code style
    validate_code_style(code_style)

    # Generate code
    code = CodeGenerator.generate_code(
        images_data, code_style, variable_prefix, include_dimensions
    )

    return jsonify({
        "code": code,
        "code_style": code_style,
        "image_count": len(images_data)
    })
