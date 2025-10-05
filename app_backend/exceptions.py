"""
Custom Exception Classes
Defines application-specific exceptions for better error handling
"""


class Image2CppException(Exception):
    """Base exception for all image2cpp errors"""
    pass


class ValidationError(Image2CppException):
    """Raised when input validation fails"""
    pass


class ImageProcessingError(Image2CppException):
    """Raised when image processing operations fail"""
    pass


class ConfigurationError(Image2CppException):
    """Raised when configuration is invalid or missing"""
    pass


class NetworkError(Image2CppException):
    """Raised when network operations fail"""
    pass
