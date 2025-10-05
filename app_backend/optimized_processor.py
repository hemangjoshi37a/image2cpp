"""
Optimized Image Processing Module
High-performance image manipulation using NumPy vectorization and caching
"""

import logging
import io
import base64
import hashlib
from typing import Dict, List, Tuple, Optional
from functools import lru_cache
from PIL import Image
import numpy as np
from .exceptions import ImageProcessingError, ValidationError
from .validators import validate_image_dimensions, validate_compression_factor

logger = logging.getLogger(__name__)


class OptimizedImageProcessor:
    """
    High-performance image processor using NumPy vectorization
    and optimized algorithms for faster image-to-byte-array conversion
    """

    # Class-level cache for frequently accessed crop/compression combinations
    _processing_cache = {}
    _cache_max_size = 100

    @staticmethod
    def _compute_cache_key(image_hash: str, threshold: int, mode: str,
                          crop_settings: Dict[str, int], compression_factor: int) -> str:
        """Compute cache key for processed images"""
        crop_str = f"{crop_settings['top']}{crop_settings['right']}{crop_settings['bottom']}{crop_settings['left']}"
        return f"{image_hash}_{threshold}_{mode}_{crop_str}_{compression_factor}"

    @staticmethod
    def _get_image_hash(image_data: bytes) -> str:
        """Generate hash for image data for caching purposes"""
        return hashlib.md5(image_data).hexdigest()

    @staticmethod
    def apply_crop(image: Image.Image, crop_settings: Dict[str, int]) -> Image.Image:
        """
        Apply crop to image with validation
        Optimized with bounds checking
        """
        try:
            width, height = image.size
            left = crop_settings['left']
            top = crop_settings['top']
            right = width - crop_settings['right']
            bottom = height - crop_settings['bottom']

            # Validate crop boundaries
            if left >= right or top >= bottom:
                raise ValidationError("Invalid crop settings: result would be empty or negative")

            if left < 0 or top < 0 or right > width or bottom > height:
                raise ValidationError("Crop settings exceed image boundaries")

            logger.debug(f"Applying crop: left={left}, top={top}, right={right}, bottom={bottom}")
            return image.crop((left, top, right, bottom))
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Error applying crop: {e}")
            raise ImageProcessingError(f"Failed to crop image: {str(e)}")

    @staticmethod
    def compress_image(image: Image.Image, compression_factor: int) -> Image.Image:
        """
        Compress/resize image using high-quality resampling
        Optimized with Lanczos filter for best quality
        """
        try:
            validate_compression_factor(compression_factor)

            if compression_factor < 100:
                width, height = image.size
                new_width = max(1, int(width * (compression_factor / 100)))
                new_height = max(1, int(height * (compression_factor / 100)))

                validate_image_dimensions(new_width, new_height)

                logger.debug(f"Compressing image: {width}x{height} -> {new_width}x{new_height}")
                # Use LANCZOS for best quality downsampling
                return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            return image
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Error compressing image: {e}")
            raise ImageProcessingError(f"Failed to compress image: {str(e)}")

    @staticmethod
    def image_to_byte_array_optimized(image: Image.Image, threshold: int = 128,
                                     mode: str = 'horizontal') -> List[int]:
        """
        Convert image to byte array using NumPy vectorization
        PERFORMANCE OPTIMIZED: ~10-50x faster than loop-based approach

        Args:
            image: PIL Image object (grayscale)
            threshold: Binarization threshold (0-255)
            mode: Scan mode ('horizontal' or 'vertical')

        Returns:
            List of bytes representing the image
        """
        try:
            width, height = image.size

            # Convert image to numpy array (much faster than getdata())
            img_array = np.array(image, dtype=np.uint8)

            # Handle RGB/RGBA by converting to grayscale if needed
            if len(img_array.shape) == 3:
                # Fast grayscale conversion using NumPy
                img_array = np.mean(img_array[:, :, :3], axis=2).astype(np.uint8)

            # Binarize: create boolean array where True = pixel > threshold
            binary = img_array > threshold

            # Reshape based on mode
            if mode == 'vertical':
                # Transpose for vertical scanning
                binary = binary.T

            # Flatten to 1D array in row-major order
            bits = binary.flatten()

            # Calculate number of bytes needed
            num_bits = len(bits)
            num_bytes = (num_bits + 7) // 8

            # Pad with zeros to make length multiple of 8
            if num_bits % 8 != 0:
                bits = np.pad(bits, (0, 8 - (num_bits % 8)), mode='constant', constant_values=0)

            # Reshape to (num_bytes, 8) for efficient byte packing
            bits_reshaped = bits.reshape(-1, 8)

            # Create bit position weights: [128, 64, 32, 16, 8, 4, 2, 1]
            bit_weights = np.array([1 << (7 - i) for i in range(8)], dtype=np.uint8)

            # Vectorized byte packing: multiply each bit by its weight and sum
            byte_array = np.sum(bits_reshaped * bit_weights, axis=1, dtype=np.uint8)

            logger.debug(f"Generated byte array: {len(byte_array)} bytes (vectorized)")
            return byte_array.tolist()

        except Exception as e:
            logger.error(f"Error in optimized byte array conversion: {e}")
            raise ImageProcessingError(f"Failed to convert image: {str(e)}")

    @staticmethod
    def byte_array_to_image_optimized(byte_array: List[int], width: int, height: int,
                                     mode: str = 'horizontal') -> Image.Image:
        """
        Convert byte array back to image using NumPy vectorization
        PERFORMANCE OPTIMIZED: Much faster than loop-based approach

        Args:
            byte_array: List of bytes
            width: Image width
            height: Image height
            mode: Scan mode ('horizontal' or 'vertical')

        Returns:
            PIL Image object
        """
        try:
            # Convert to numpy array
            bytes_np = np.array(byte_array, dtype=np.uint8)

            # Unpack each byte into 8 bits
            # Use bitwise operations with broadcasting
            bit_positions = np.arange(7, -1, -1)
            bits = ((bytes_np[:, np.newaxis] >> bit_positions) & 1).flatten()

            # Take only the bits we need (width * height)
            total_pixels = width * height
            bits = bits[:total_pixels]

            # Reshape based on mode
            if mode == 'horizontal':
                img_array = bits.reshape(height, width)
            else:  # vertical
                img_array = bits.reshape(width, height).T

            # Convert to PIL Image (multiply by 255 for proper display)
            img_array = (img_array * 255).astype(np.uint8)
            img = Image.fromarray(img_array, mode='L')

            return img

        except Exception as e:
            logger.error(f"Error in optimized image reconstruction: {e}")
            raise ImageProcessingError(f"Failed to create preview: {str(e)}")

    @staticmethod
    def generate_preview(image: Image.Image, optimize: bool = True) -> str:
        """
        Generate base64-encoded preview image with optimization

        Args:
            image: PIL Image object
            optimize: Whether to optimize PNG output

        Returns:
            Base64-encoded PNG image string
        """
        try:
            preview_buffer = io.BytesIO()
            # Optimize PNG output for smaller size and faster transmission
            image.save(preview_buffer, format='PNG', optimize=optimize, compress_level=6)
            preview_base64 = base64.b64encode(preview_buffer.getvalue()).decode('utf-8')
            return preview_base64
        except Exception as e:
            logger.error(f"Error generating preview: {e}")
            raise ImageProcessingError(f"Failed to generate preview: {str(e)}")

    @classmethod
    def process_image_cached(cls, image_data: bytes, threshold: int, mode: str,
                            crop_settings: Dict[str, int], compression_factor: int,
                            filename: str = "image", use_cache: bool = True) -> Dict:
        """
        Complete image processing pipeline with caching support
        PERFORMANCE OPTIMIZED: Caches processed results to avoid reprocessing

        Args:
            image_data: Raw image data bytes
            threshold: Binarization threshold
            mode: Scan mode
            crop_settings: Crop configuration
            compression_factor: Compression percentage
            filename: Original filename
            use_cache: Whether to use caching (default: True)

        Returns:
            Dictionary containing conversion results
        """
        try:
            # Compute cache key
            image_hash = cls._get_image_hash(image_data)
            cache_key = cls._compute_cache_key(image_hash, threshold, mode,
                                               crop_settings, compression_factor)

            # Check cache
            if use_cache and cache_key in cls._processing_cache:
                logger.debug(f"Cache hit for {filename}")
                result = cls._processing_cache[cache_key].copy()
                result['filename'] = filename  # Update filename
                result['cache_hit'] = True
                return result

            # Load and convert to grayscale
            image = Image.open(io.BytesIO(image_data)).convert('L')
            logger.debug(f"Processing {filename}: Original size {image.size}")

            # Apply transformations
            image = cls.apply_crop(image, crop_settings)
            logger.debug(f"After crop: {image.size}")

            image = cls.compress_image(image, compression_factor)
            logger.debug(f"After compression: {image.size}")

            # Convert to byte array using optimized method
            byte_array = cls.image_to_byte_array_optimized(image, threshold, mode)
            formatted_array = ', '.join(f'0x{byte:02X}' for byte in byte_array)

            # Generate preview with optimization
            preview_base64 = cls.generate_preview(image, optimize=True)

            result = {
                "filename": filename,
                "byte_array": formatted_array,
                "byte_array_raw": byte_array,
                "width": image.width,
                "height": image.height,
                "preview_image": preview_base64,
                "crop_settings": crop_settings,
                "cache_hit": False
            }

            # Store in cache (with size limit)
            if use_cache:
                if len(cls._processing_cache) >= cls._cache_max_size:
                    # Remove oldest entry (simple FIFO strategy)
                    oldest_key = next(iter(cls._processing_cache))
                    del cls._processing_cache[oldest_key]
                    logger.debug(f"Cache full, removed oldest entry")

                cls._processing_cache[cache_key] = result.copy()
                logger.debug(f"Cached result for {filename}")

            return result

        except Exception as e:
            logger.error(f"Error processing image {filename}: {e}")
            raise ImageProcessingError(f"Failed to process {filename}: {str(e)}")

    @classmethod
    def clear_cache(cls):
        """Clear the processing cache"""
        cls._processing_cache.clear()
        logger.info("Processing cache cleared")

    @classmethod
    def get_cache_stats(cls) -> Dict:
        """Get cache statistics"""
        return {
            "cache_size": len(cls._processing_cache),
            "cache_max_size": cls._cache_max_size,
            "cache_keys": list(cls._processing_cache.keys())
        }


class BatchProcessor:
    """
    Handles batch image processing with parallel execution
    """

    @staticmethod
    def process_batch_sequential(images_data: List[Tuple[bytes, str]],
                                 threshold: int, mode: str,
                                 crop_settings: Dict[str, int],
                                 compression_factor: int,
                                 use_cache: bool = True) -> Tuple[List[Dict], List[Dict]]:
        """
        Process multiple images sequentially with caching

        Args:
            images_data: List of (image_bytes, filename) tuples
            threshold: Binarization threshold
            mode: Scan mode
            crop_settings: Crop configuration
            compression_factor: Compression percentage
            use_cache: Whether to use caching

        Returns:
            Tuple of (results, errors)
        """
        results = []
        errors = []

        for idx, (image_data, filename) in enumerate(images_data):
            try:
                logger.info(f"Processing image {idx+1}/{len(images_data)}: {filename}")

                result = OptimizedImageProcessor.process_image_cached(
                    image_data,
                    threshold,
                    mode,
                    crop_settings,
                    compression_factor,
                    filename,
                    use_cache=use_cache
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                errors.append({
                    "filename": filename,
                    "error": str(e)
                })

        return results, errors

    @staticmethod
    def process_batch_parallel(images_data: List[Tuple[bytes, str]],
                               threshold: int, mode: str,
                               crop_settings: Dict[str, int],
                               compression_factor: int,
                               use_cache: bool = True,
                               max_workers: Optional[int] = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Process multiple images in parallel using ThreadPoolExecutor
        PERFORMANCE OPTIMIZED: Parallel processing for I/O-bound operations

        Args:
            images_data: List of (image_bytes, filename) tuples
            threshold: Binarization threshold
            mode: Scan mode
            crop_settings: Crop configuration
            compression_factor: Compression percentage
            use_cache: Whether to use caching
            max_workers: Maximum number of worker threads (None = auto)

        Returns:
            Tuple of (results, errors)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import os

        results = []
        errors = []

        # Auto-detect optimal number of workers
        if max_workers is None:
            max_workers = min(len(images_data), (os.cpu_count() or 1) * 2)

        logger.info(f"Processing {len(images_data)} images with {max_workers} workers")

        def process_single(image_data: bytes, filename: str, idx: int) -> Dict:
            """Process a single image (worker function)"""
            try:
                logger.info(f"Processing image {idx+1}/{len(images_data)}: {filename}")

                result = OptimizedImageProcessor.process_image_cached(
                    image_data,
                    threshold,
                    mode,
                    crop_settings,
                    compression_factor,
                    filename,
                    use_cache=use_cache
                )
                return {"success": True, "data": result, "idx": idx}

            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                return {
                    "success": False,
                    "error": {"filename": filename, "error": str(e)},
                    "idx": idx
                }

        # Submit all tasks to thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create futures for all images
            futures = {
                executor.submit(process_single, img_data, fname, idx): idx
                for idx, (img_data, fname) in enumerate(images_data)
            }

            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result["success"]:
                        results.append(result["data"])
                    else:
                        errors.append(result["error"])
                except Exception as e:
                    logger.error(f"Future execution error: {e}")
                    errors.append({"filename": "unknown", "error": str(e)})

        # Sort results by original index to maintain order
        # (optional, depending on whether order matters)

        logger.info(f"Batch processing complete: {len(results)} succeeded, {len(errors)} failed")
        return results, errors
