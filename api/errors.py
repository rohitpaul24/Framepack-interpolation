"""
API Error Handling

This module defines custom exceptions and error handlers for the API.
"""

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import traceback


class APIError(Exception):
    """Base API error"""
    def __init__(
        self, 
        message: str, 
        error_code: str = "API_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(APIError):
    """Validation error"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            details=details
        )


class ProcessingError(APIError):
    """Video processing error"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="PROCESSING_ERROR",
            status_code=500,
            details=details
        )


class ModelNotLoadedError(APIError):
    """Models not loaded error"""
    def __init__(self):
        super().__init__(
            message="Models are not loaded yet. Please wait for server initialization.",
            error_code="MODELS_NOT_LOADED",
            status_code=503
        )


class InvalidDurationError(ValidationError):
    """Invalid duration error"""
    def __init__(self, duration: int):
        super().__init__(
            message=f"Invalid duration: {duration}. Must be between 1 and 60 seconds.",
            details={"duration": duration, "min": 1, "max": 60}
        )


class VideoTooShortError(ValidationError):
    """Video too short error"""
    def __init__(self, duration: float):
        super().__init__(
            message=f"Video is too short ({duration:.2f}s). Minimum length is 1 second.",
            details={"duration": duration, "minimum": 1.0}
        )


async def api_error_handler(request: Request, exc: APIError):
    """Handle API errors"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error_code": exc.error_code,
            "message": exc.message,
            "details": exc.details
        }
    )


async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    # Log the full traceback
    error_trace = traceback.format_exc()
    print(f"Unexpected error: {error_trace}")
    
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error_code": "INTERNAL_ERROR",
            "message": "An unexpected error occurred",
            "details": {"error": str(exc)}
        }
    )
