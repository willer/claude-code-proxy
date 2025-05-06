"""
Claude Code Proxy application.
"""
import os
import logging
from fastapi import FastAPI

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

# Create FastAPI app
app = FastAPI(
    title="Claude Code Proxy",
    description="A FastAPI service for proxying Claude Code requests",
    version="1.0.0",
)

# Import and include routes
from app.routes.token_count import router as token_count_router

# Register routers
app.include_router(token_count_router)

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