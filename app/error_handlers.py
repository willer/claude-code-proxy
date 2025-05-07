"""
Exception handlers for Claude Code Proxy.
"""
import logging
import traceback
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException

logger = logging.getLogger(__name__)


async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handle HTTP exceptions with proper logging and formatting.
    
    Args:
        request: The incoming request
        exc: The HTTP exception
        
    Returns:
        JSONResponse with error details
    """
    # Log the exception with appropriate level based on status code
    if exc.status_code >= 500:
        logger.error(f"HTTP {exc.status_code} error: {exc.detail}")
    elif exc.status_code >= 400:
        logger.warning(f"HTTP {exc.status_code} error: {exc.detail}")
    
    # Return formatted error response
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "type": "error",
            "error": {
                "status": exc.status_code,
                "message": str(exc.detail)
            }
        }
    )


async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle unexpected exceptions with detailed logging.
    
    Args:
        request: The incoming request
        exc: The exception
        
    Returns:
        JSONResponse with error details
    """
    # Log the exception with full traceback
    error_traceback = traceback.format_exc()
    method = request.method
    path = request.url.path
    
    logger.error(f"Unhandled exception in {method} {path}: {str(exc)}")
    logger.error(error_traceback)
    
    # Return formatted error response
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "type": "error",
            "error": {
                "status": 500,
                "message": f"Internal server error: {str(exc)}"
            }
        }
    )