"""
Configuration Management Module
Handles environment variables, settings, and application constants
"""

import os
import logging
from dotenv import load_dotenv, set_key
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Application Constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_IMAGE_DIMENSION = 4096
ALLOWED_IMAGE_FORMATS = {'PNG', 'JPEG', 'JPG', 'BMP', 'GIF', 'WEBP'}
REQUEST_TIMEOUT = 10  # seconds
MAX_BATCH_SIZE = 20  # Maximum images per batch

# Default values
DEFAULT_CROP_SETTINGS = {'top': 0, 'right': 0, 'bottom': 0, 'left': 0}
DEFAULT_COMPRESSION_FACTOR = 100
DEFAULT_THRESHOLD = 128


class Config:
    """Application configuration manager"""

    def __init__(self):
        """Initialize configuration"""
        load_dotenv()
        self._setup_logging()

    def _setup_logging(self):
        """Configure application logging"""
        log_level = os.getenv('LOG_LEVEL', 'DEBUG')
        log_file = os.getenv('LOG_FILE', 'image2cpp.log')

        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def get_crop_settings(self) -> Dict[str, int]:
        """
        Load crop settings from environment

        Returns:
            Dictionary containing crop settings (top, right, bottom, left)
        """
        try:
            load_dotenv()
            settings = {
                'top': int(os.getenv('CROP_TOP', 0)),
                'right': int(os.getenv('CROP_RIGHT', 0)),
                'bottom': int(os.getenv('CROP_BOTTOM', 0)),
                'left': int(os.getenv('CROP_LEFT', 0))
            }
            logger.debug(f"Loaded crop settings: {settings}")
            return settings
        except ValueError as e:
            logger.error(f"Error parsing crop settings: {e}")
            return DEFAULT_CROP_SETTINGS.copy()
        except Exception as e:
            logger.error(f"Unexpected error loading crop settings: {e}")
            return DEFAULT_CROP_SETTINGS.copy()

    def get_compression_factor(self) -> int:
        """
        Load compression factor from environment

        Returns:
            Compression factor as percentage (1-100)
        """
        try:
            load_dotenv()
            compression_factor = int(os.getenv('COMPRESSION_FACTOR', DEFAULT_COMPRESSION_FACTOR))
            logger.debug(f"Loaded compression factor: {compression_factor}%")
            return compression_factor
        except ValueError as e:
            logger.error(f"Error parsing compression factor: {e}")
            return DEFAULT_COMPRESSION_FACTOR
        except Exception as e:
            logger.error(f"Unexpected error loading compression factor: {e}")
            return DEFAULT_COMPRESSION_FACTOR

    def save_crop_settings(self, settings: Dict[str, Any]) -> None:
        """
        Save crop settings to environment file

        Args:
            settings: Dictionary containing crop settings

        Raises:
            Exception: If settings cannot be saved
        """
        try:
            logger.debug(f"Saving crop settings: {settings}")

            # Create .env file if it doesn't exist
            if not os.path.exists('.env'):
                with open('.env', 'w') as f:
                    f.write('')

            for key, value in settings.items():
                set_key('.env', f'CROP_{key.upper()}', str(value))
            load_dotenv()  # Reload environment variables
            logger.info("Crop settings saved successfully")
        except Exception as e:
            logger.error(f"Error saving crop settings: {e}")
            raise


# Global configuration instance
config = Config()
