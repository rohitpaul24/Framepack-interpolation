"""
API Package

This package contains modules for API validation, error handling, and logging.
"""

from .validation import (
    ProcessVideoRequest,
    ProcessVideoResponse,
    ConfigResponse,
    ConfigUpdateRequest,
    HealthResponse,
    ErrorResponse,
    VideoOutputs
)

from .errors import (
    APIError,
    ValidationError,
    ProcessingError,
    ModelNotLoadedError,
    InvalidDurationError,
    VideoTooShortError,
    api_error_handler,
    general_exception_handler
)

from .logging import (
    RequestLoggingMiddleware,
    log_video_processing,
    log_config_update,
    log_error,
    logger
)

# Job management
from .jobs import Job, JobStatus, JobData

# Worker
from .worker import start_worker, queue_job, get_queue_size

__all__ = [
    # Validation
    'ProcessVideoRequest', 'ProcessVideoResponse', 'ConfigResponse',
    'ConfigUpdateRequest', 'HealthResponse', 'ErrorResponse', 'VideoOutputs',
    # Errors
    'APIError', 'ValidationError', 'ProcessingError', 'ModelNotLoadedError',
    'InvalidDurationError', 'VideoTooShortError',
    'api_error_handler', 'general_exception_handler',
    # Logging
    'RequestLoggingMiddleware', 'log_video_processing', 'log_config_update', 'log_error', 'logger',
    # Jobs
    'Job', 'JobStatus', 'JobData',
    # Worker
    'start_worker', 'queue_job', 'get_queue_size'
]
