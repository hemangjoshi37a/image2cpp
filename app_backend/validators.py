"""
Input Validation Module
Validates user inputs, settings, and image parameters
"""

import logging
from typing import Dict, Any
from .exceptions import ValidationError
from .config import MAX_IMAGE_DIMENSION, MAX_BATCH_SIZE

logger = logging.getLogger(__name__)


def validate_crop_settings(settings: Dict[str, Any]) -> None:
    """
    Validate crop settings

    Args:
        settings: Dictionary containing crop settings

    Raises:
        ValidationError: If settings are invalid
    """
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
    """
    Validate threshold value

    Args:
        threshold: Threshold value for image binarization

    Raises:
        ValidationError: If threshold is invalid
    """
    if not isinstance(threshold, int) or threshold < 0 or threshold > 255:
        raise ValidationError(f"Threshold must be an integer between 0 and 255, got {threshold}")


def validate_compression_factor(factor: int) -> None:
    """
    Validate compression factor

    Args:
        factor: Compression factor as percentage (1-100)

    Raises:
        ValidationError: If compression factor is invalid
    """
    if not isinstance(factor, int) or factor < 1 or factor > 100:
        raise ValidationError(f"Compression factor must be an integer between 1 and 100, got {factor}")


def validate_image_dimensions(width: int, height: int) -> None:
    """
    Validate image dimensions

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Raises:
        ValidationError: If dimensions are invalid
    """
    if width <= 0 or height <= 0:
        raise ValidationError(f"Image dimensions must be positive, got {width}x{height}")

    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise ValidationError(
            f"Image dimensions exceed maximum allowed ({MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION})"
        )


def validate_image_url(url: str) -> None:
    """
    Validate image URL

    Args:
        url: URL string to validate

    Raises:
        ValidationError: If URL is invalid
    """
    if not url or not isinstance(url, str):
        raise ValidationError("Invalid URL provided")

    if not url.startswith(('http://', 'https://')):
        raise ValidationError("URL must start with http:// or https://")

    if len(url) > 2048:
        raise ValidationError("URL is too long")


def validate_batch_size(count: int) -> None:
    """
    Validate batch conversion size

    Args:
        count: Number of images in batch

    Raises:
        ValidationError: If batch size is invalid
    """
    if count <= 0:
        raise ValidationError("Batch must contain at least one image")

    if count > MAX_BATCH_SIZE:
        raise ValidationError(f"Maximum {MAX_BATCH_SIZE} images allowed per batch, got {count}")


def validate_conversion_mode(mode: str) -> None:
    """
    Validate conversion mode

    Args:
        mode: Conversion mode ('horizontal' or 'vertical')

    Raises:
        ValidationError: If mode is invalid
    """
    valid_modes = ['horizontal', 'vertical']
    if mode not in valid_modes:
        raise ValidationError(f"Invalid mode '{mode}'. Must be one of: {', '.join(valid_modes)}")


def validate_code_style(style: str) -> None:
    """
    Validate code generation style

    Args:
        style: Code style ('arduino', 'cpp', or 'python')

    Raises:
        ValidationError: If style is invalid
    """
    valid_styles = ['arduino', 'cpp', 'python']
    if style not in valid_styles:
        raise ValidationError(f"Invalid code style '{style}'. Must be one of: {', '.join(valid_styles)}")
