"""
image2cpp - Image to C++ Byte Array Converter
Main Flask application entry point

This application converts images to byte arrays for use in embedded systems,
Arduino projects, and other C++ applications.
"""

from flask import Flask
from flask_cors import CORS
from app_backend.config import config
from app_backend.routes import api
import logging

# Initialize Flask application
app = Flask(__name__)
CORS(app)

# Register blueprints
app.register_blueprint(api)

# Get logger
logger = logging.getLogger(__name__)


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return {"error": "Resource not found", "status": 404}, 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return {"error": "Internal server error", "status": 500}, 500


@app.route('/health')
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "image2cpp"}, 200


if __name__ == '__main__':
    logger.info("Starting image2cpp server...")
    logger.info("Access the application at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
