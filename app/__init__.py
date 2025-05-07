"""
Claude Code Proxy application.
"""
import os
import logging
from fastapi import FastAPI, Request, status
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.responses import JSONResponse

# Setup logging
from app.utils.logging import setup_logging
logger = setup_logging()

# Log environment configuration
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
BIG_MODEL = os.environ.get("BIG_MODEL", "")
SMALL_MODEL = os.environ.get("SMALL_MODEL", "")

logger.info(f"API Keys Configured: Anthropic={'✓ YES' if ANTHROPIC_API_KEY else '✗ NO'}, " +
           f"OpenAI={'✓ YES' if OPENAI_API_KEY else '✗ NO'}, " +
           f"Gemini={'✓ YES' if GEMINI_API_KEY else '✗ NO'}")
logger.info(f"Model Configuration: BIG={BIG_MODEL}, SMALL={SMALL_MODEL}")

# Import middleware and exception handlers
from app.middleware import LoggingMiddleware
from app.error_handlers import http_exception_handler, general_exception_handler

# Create FastAPI app
app = FastAPI(
    title="Claude Code Proxy",
    description="A FastAPI service for proxying Claude Code requests",
    version="1.0.0",
)

# Add middleware
app.add_middleware(LoggingMiddleware)

# Register exception handlers
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# Add validation error handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with proper formatting."""
    # Get error details from the exception
    errors = exc.errors()
    error_messages = []
    
    # Format error messages
    for error in errors:
        loc = " -> ".join(str(l) for l in error["loc"])
        msg = error["msg"]
        error_messages.append(f"{loc}: {msg}")
    
    # Combine all error messages
    detail = "\n".join(error_messages)
    logger.warning(f"Validation error: {detail}")
    
    # Return formatted error response
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "type": "error",
            "error": {
                "status": 422,
                "message": "Validation error",
                "details": error_messages
            }
        }
    )

# Import and include routes
from app.routes.token_count import router as token_count_router
from app.routes.messages import router as messages_router

# Register routers
app.include_router(token_count_router)
app.include_router(messages_router)

# Root endpoint
@app.get("/")
def read_root():
    """Root endpoint that displays API information."""
    return {
        "name": "Claude Code Proxy",
        "version": "1.0.0",
        "status": "running",
        "models": {
            "big": BIG_MODEL,
            "small": SMALL_MODEL
        }
    }