"""
Middleware for Claude Code Proxy.
"""
import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging request duration and basic info.
    """
    
    async def dispatch(self, request: Request, call_next):
        """
        Process a request and log metrics.
        
        Args:
            request: The incoming request
            call_next: The next middleware or endpoint handler
            
        Returns:
            The response from the next handler
        """
        # Start timer for request duration
        start_time = time.time()
        
        # Get the request method and path
        method = request.method
        path = request.url.path
        
        # Skip logging for certain paths
        if path == "/" or path.startswith("/docs") or path.startswith("/openapi"):
            return await call_next(request)
        
        # Log the incoming request
        logger.debug(f"Request started: {method} {path}")
        
        # Process the request
        try:
            response = await call_next(request)
            
            # Calculate request duration
            duration = time.time() - start_time
            
            # Log the completed request with status code and duration
            status_code = response.status_code
            logger.debug(f"Request completed: {method} {path} - {status_code} - {duration:.2f}s")
            
            return response
            
        except Exception as e:
            # Log any exceptions
            logger.error(f"Request error: {method} {path} - {str(e)}")
            duration = time.time() - start_time
            logger.debug(f"Request failed after {duration:.2f}s")
            raise