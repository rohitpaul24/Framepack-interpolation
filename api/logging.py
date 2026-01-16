"""
API Request Logging

This module provides middleware for logging API requests and responses.
"""

import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all API requests"""
    
    async def dispatch(self, request: Request, call_next: Callable):
        # Start timer
        start_time = time.time()
        
        # Log request
        logger.info(f"→ {request.method} {request.url.path}")
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log response
        logger.info(
            f"← {request.method} {request.url.path} "
            f"- Status: {response.status_code} "
            f"- Duration: {duration:.2f}s"
        )
        
        return response


def log_video_processing(video_name: str, duration: float, loops: int):
    """Log video processing completion"""
    logger.info(
        f"✓ Processed video: {video_name} "
        f"({duration:.2f}s, {loops} loops)"
    )


def log_config_update(updated_fields: dict):
    """Log configuration updates"""
    logger.info(f"Configuration updated: {updated_fields}")


def log_error(error_code: str, message: str):
    """Log errors"""
    logger.error(f"Error [{error_code}]: {message}")
