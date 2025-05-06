"""
Authentication utilities for Claude Code Proxy.
"""
import os
import logging
import json
from typing import Dict, Optional, Tuple
from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)

# Environment variables
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


def get_anthropic_auth_headers(request: Request) -> Dict[str, str]:
    """
    Extract and prepare Anthropic authentication headers.
    
    Args:
        request: The FastAPI request object
        
    Returns:
        Dictionary with appropriate authentication headers
        
    Raises:
        HTTPException: If no API key is available
    """
    # Extract headers from the request
    client_headers = dict(request.headers.items())
    
    # Normalize header keys to lowercase for consistent lookup
    client_headers_normalized = {k.lower(): v for k, v in client_headers.items()}
    client_headers = {**client_headers, **client_headers_normalized}
    
    # Debug logging with sensitive data redacted
    if logger.level <= logging.DEBUG:
        safe_headers = {k: ("**REDACTED**" if k.lower() in ["x-api-key", "authorization"] else v) 
                        for k, v in client_headers.items()}
        logger.debug(f"Received headers: {json.dumps(safe_headers)}")
    
    # Initialize headers with required content type
    headers = {
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",  # Basic Claude-3 version
    }
    
    # Check for custom version header in client request
    if "anthropic-version" in client_headers:
        headers["anthropic-version"] = client_headers["anthropic-version"]
        logger.debug(f"Using client-provided Anthropic API version: {client_headers['anthropic-version']}")
    
    # Extract client's API key or Authorization token
    client_api_key = None
    
    # Check for x-api-key header
    if "x-api-key" in client_headers:
        client_api_key = client_headers["x-api-key"]
        logger.debug("Found client's API key in x-api-key header")
    
    # Check for Authorization Bearer token (Claude Max)
    elif "authorization" in client_headers:
        auth_header = client_headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            client_api_key = auth_header.replace("Bearer ", "")
            logger.debug("Found client's API key in Authorization Bearer header")
    
    # Check for anthropic-api-key header
    elif "anthropic-api-key" in client_headers:
        client_api_key = client_headers["anthropic-api-key"]
        logger.debug("Found client's API key in anthropic-api-key header")
    
    # Use client's key if available, fall back to server's key
    api_key = client_api_key or ANTHROPIC_API_KEY
    
    # Handle authentication method (Bearer token vs API key)
    if "authorization" in client_headers and client_headers["authorization"].startswith("Bearer "):
        # For Claude Max, use Authorization header directly
        auth_value = client_headers["authorization"]
        logger.debug("Using Authorization header for Claude Max subscription")
        headers["authorization"] = auth_value
    elif api_key:
        # For regular Anthropic API, use x-api-key
        logger.debug("Using x-api-key header for regular Anthropic API")
        headers["x-api-key"] = api_key
    else:
        # No authentication available
        raise HTTPException(
            status_code=401,
            detail="No API key found for Anthropic. Please provide an API key in the request headers or set ANTHROPIC_API_KEY in environment variables."
        )
    
    # Forward Claude Max subscription-related headers if present
    subscription_headers = [
        "anthropic-beta",
        "anthropic-organization",
        "anthropic-account-id",
        "anthropic-client-id",
        "anthropic-client-secret",
        "anthropic-workspace-id"
    ]
    
    for header in subscription_headers:
        if header in client_headers:
            headers[header] = client_headers[header]
            logger.debug(f"Forwarding header: {header}")
    
    return headers


def get_anthropic_token_count_auth() -> Dict[str, str]:
    """
    Get authentication headers for Anthropic token counting.
    Always uses the environment variable for simplicity.
    
    Returns:
        Dictionary with authentication headers
        
    Raises:
        HTTPException: If no environment API key is available
    """
    # For token counting, always use the environment API key for simplicity
    api_key = ANTHROPIC_API_KEY
    
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="No API key found for Anthropic token counting. Please set ANTHROPIC_API_KEY in environment variables."
        )
    
    # Create headers with the API key from environment
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    
    logger.debug("Using environment ANTHROPIC_API_KEY for token counting")
    return headers