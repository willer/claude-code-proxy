"""
Token counting endpoint for Claude Code Proxy.
"""
import os
import json
import logging
import httpx
from typing import Dict, Any, List, Union
from fastapi import APIRouter, HTTPException, Request

from app.models import TokenCountRequest, TokenCountResponse
from app.utils.auth import get_anthropic_token_count_auth

# Environment variables
BIG_MODEL = os.environ.get("BIG_MODEL", "")

# Create router
router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/v1/messages/count_tokens")
async def count_tokens(
    request: TokenCountRequest,
    raw_request: Request
):
    """
    Count tokens for a messages request.
    
    Args:
        request: The token count request
        raw_request: The raw FastAPI request
        
    Returns:
        TokenCountResponse with input token count
        
    Raises:
        HTTPException: If an error occurs during token counting
    """
    try:
        # Log the incoming token count request
        original_model = request.original_model or request.model
        
        # Get the display name for logging, just the model name without provider prefix
        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]
        
        # Clean model name for capability check
        clean_model = request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/"):]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/"):]
        
        # Check if we need direct passthrough to Anthropic for token counting
        is_direct_anthropic = (BIG_MODEL == "passthrough" and 
                             (request.model.startswith("anthropic/") or "claude" in request.model.lower()))
        
        # Handle direct passthrough to Anthropic API for token counting
        if is_direct_anthropic:
            logger.info(f"📌 TRUE PASSTHROUGH: token counting directly with Anthropic API for {request.model}")
            
            # Ensure model has anthropic/ prefix if needed
            model_name = request.model
            if not model_name.startswith("anthropic/") and "claude" in model_name.lower():
                model_name = f"anthropic/{model_name}"
                logger.debug(f"Added anthropic/ prefix to model: {request.model} -> {model_name}")
            
            try:
                # Create direct Anthropic request
                api_url = "https://api.anthropic.com/v1/messages/count_tokens"
                
                # Dump the original request to JSON (preserving Anthropic format)
                request_dict = request.dict(exclude_none=True, exclude_unset=True)
                
                # Always remove the 'anthropic/' prefix for the Anthropic API
                request_dict["model"] = model_name.replace("anthropic/", "")
                logger.debug(f"Using model for token counting: {request_dict['model']}")
                
                # Remove empty or null fields that might cause API errors
                # Specifically, don't send tool_choice if tools is not present
                if "tool_choice" in request_dict and (not request_dict.get("tools") or len(request_dict.get("tools", [])) == 0):
                    logger.debug("Removing tool_choice from request as tools array is empty")
                    request_dict.pop("tool_choice", None)
                    
                # Remove any other null or empty fields
                for key in list(request_dict.keys()):
                    if request_dict[key] is None or (isinstance(request_dict[key], (list, dict)) and len(request_dict[key]) == 0):
                        logger.debug(f"Removing empty field from request: {key}")
                        request_dict.pop(key)
                
                # Get authentication headers (simplified to always use environment variable)
                headers = get_anthropic_token_count_auth()
                
                # Use httpx for async request with timeout
                async with httpx.AsyncClient(timeout=30.0) as client:
                    # Token counting rarely has too many tools, but still log for debugging
                    logger.debug(f"Token count request: {len(request_dict.get('tools', []))} tools")
                    
                    # Post the request with timeout
                    response = await client.post(api_url, json=request_dict, headers=headers, timeout=30.0)
                    
                    # Check for errors
                    if response.status_code != 200:
                        error_text = response.text
                        logger.error(f"Anthropic token counting error ({response.status_code}): {error_text}")
                        
                        # Log model being used for token counting
                        logger.error(f"Model used for token counting: {request_dict['model']}")
                        
                        # Return a clear error
                        raise HTTPException(
                            status_code=response.status_code,
                            detail=f"Anthropic token counting error: {error_text}"
                        )
                    
                    # Return the token count directly
                    token_count_data = response.json()
                    return TokenCountResponse(input_tokens=token_count_data.get("input_tokens", 0))
            
            except Exception as e:
                # Get full traceback for detailed debugging
                import traceback
                error_traceback = traceback.format_exc()
                
                # Log more helpful info about the request
                tool_count = len(request_dict.get('tools', []))
                message_count = len(request_dict.get('messages', []))
                
                # Create a detailed error message
                error_message = (
                    f"Error in direct Anthropic token counting for {model_name} with "
                    f"{tool_count} tools and {message_count} messages: {str(e)}"
                )
                
                # Log the full details
                logger.error(f"{error_message}\n{error_traceback}")
                
                # Create a more user-friendly error
                raise HTTPException(status_code=500, detail=error_message)
        
        # Standard non-passthrough flow using LiteLLM
        # This would be implemented based on your existing code
        # This is a placeholder
        raise HTTPException(
            status_code=501,
            detail="Non-passthrough token counting not implemented in this module"
        )
        
    except Exception as e:
        # Handle any unexpected errors
        logger.error(f"Unexpected error in token counting: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")