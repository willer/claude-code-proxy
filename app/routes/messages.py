"""
Main message creation endpoint for Claude Code Proxy.
"""
import os
import json
import logging
import httpx
import time
import asyncio
from typing import Dict, Any, List, Union, Optional
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
import litellm

from app.models import MessagesRequest, MessagesResponse
from app.utils.auth import get_anthropic_auth_headers
from app.utils.model_conversion import (
    convert_anthropic_to_litellm, 
    convert_litellm_to_anthropic,
    parse_tool_result_content
)

# Environment variables
BIG_MODEL = os.environ.get("BIG_MODEL", "")
SMALL_MODEL = os.environ.get("SMALL_MODEL", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Create router
router = APIRouter()
logger = logging.getLogger(__name__)

# Get debug level for verbose logging
log_level = logger.getEffectiveLevel()


@router.post("/v1/messages")
async def create_message(
    request: MessagesRequest,
    raw_request: Request,
    response: Response
):
    """
    Create a new message with the Anthropic API or proxy to other models.
    
    Args:
        request: The message request
        raw_request: The raw FastAPI request
        response: The FastAPI response
        
    Returns:
        MessagesResponse or StreamingResponse for streaming
        
    Raises:
        HTTPException: If an error occurs during message creation
    """
    try:
        # Log the incoming message request
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
        
        # Get the thinking budget if set
        thinking_budget = None
        if request.thinking and request.thinking.enabled and request.thinking.budget_tokens:
            thinking_budget = request.thinking.budget_tokens
        
        # Number of messages and tools for logging
        num_messages = len(request.messages)
        num_tools = len(request.tools) if request.tools else 0
        
        # Log basic request info
        reasoning_level = "high" if thinking_budget and thinking_budget > 4096 else ("medium" if thinking_budget else None)
        from app.utils.logging import log_request_beautifully
        log_request_beautifully("POST", "/v1/messages", display_model, None, num_messages, num_tools, 200, reasoning_level)
        
        # Check if we need direct passthrough to Anthropic
        is_direct_anthropic = (BIG_MODEL == "passthrough" and 
                             (request.model.startswith("anthropic/") or "claude" in request.model.lower()))
        
        # Handle direct passthrough to Anthropic API
        if is_direct_anthropic:
            logger.info(f"📌 TRUE PASSTHROUGH: connecting directly to Anthropic API for {request.model}")
            return await handle_anthropic_direct(request, raw_request)
        
        # If not in passthrough mode, use LiteLLM to handle the request with the configured model
        if BIG_MODEL != "passthrough":
            logger.info(f"📌 USING NON-PASSTHROUGH MODEL: {BIG_MODEL} for request with model {request.model}")
            return await handle_litellm(request, raw_request, BIG_MODEL)
        
        # If we reach here, we don't have a handler for this case
        raise HTTPException(
            status_code=501,
            detail="Unsupported model or configuration"
        )
    
    except Exception as e:
        # Handle any unexpected errors
        logger.error(f"Unexpected error in messages endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating message: {str(e)}")


async def handle_anthropic_direct(request: MessagesRequest, raw_request: Request):
    """
    Handle direct passthrough to Anthropic API.
    
    Args:
        request: The message request
        raw_request: The raw FastAPI request
        
    Returns:
        MessagesResponse or StreamingResponse
        
    Raises:
        HTTPException: If an error occurs with the Anthropic API
    """
    # Ensure model has anthropic/ prefix if needed
    model_name = request.model
    if not model_name.startswith("anthropic/") and "claude" in model_name.lower():
        model_name = f"anthropic/{model_name}"
        logger.debug(f"Added anthropic/ prefix to model: {request.model} -> {model_name}")
    
    # Check if streaming is requested
    is_streaming = request.stream if request.stream is not None else False
    
    if is_streaming:
        return await handle_anthropic_streaming(request, raw_request, model_name)
    else:
        return await handle_anthropic_non_streaming(request, raw_request, model_name)


async def handle_anthropic_streaming(request: MessagesRequest, raw_request: Request, model_name: str):
    """
    Handle streaming requests to Anthropic API.
    
    Args:
        request: The message request
        raw_request: The raw FastAPI request
        model_name: The model name with proper prefix
        
    Returns:
        StreamingResponse
        
    Raises:
        HTTPException: If an error occurs with the Anthropic API
    """
    try:
        # Create direct Anthropic request
        api_url = "https://api.anthropic.com/v1/messages"
        
        # Dump the original request to JSON (preserving Anthropic format)
        # Use dictionary to ensure we don't modify the original request's model
        request_dict = request.dict(exclude_none=True, exclude_unset=True)
        
        # Extract client headers first to avoid reference before assignment
        client_headers = dict(raw_request.headers.items())
        
        # Normalize header keys to lowercase for consistent lookup
        client_headers_normalized = {k.lower(): v for k, v in client_headers.items()}
        client_headers = {**client_headers, **client_headers_normalized}
        
        # Handle model name format based on API used:
        # - Claude Max (auth token present) may use full model name without removing prefixes
        # - Regular Anthropic API needs the "anthropic/" prefix removed
        if "authorization" in client_headers and client_headers["authorization"].startswith("Bearer "):
            # For Claude Max, use the original model requested
            original_model = request.original_model if hasattr(request, "original_model") and request.original_model else request.model
            model_no_prefix = original_model.replace("anthropic/", "")
            request_dict["model"] = model_no_prefix
            logger.debug(f"Using original model for Claude Max streaming: {model_no_prefix}")
        else:
            # Regular Anthropic API needs the "anthropic/" prefix removed
            request_dict["model"] = model_name.replace("anthropic/", "")
            logger.debug(f"Using model for regular API streaming: {request_dict['model']}")
        
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
        
        # Get authentication headers
        headers = get_anthropic_auth_headers(raw_request)
        
        # Log the request details (with sensitive info redacted)
        num_tools = len(request.tools) if request.tools else 0
        reasoning_level = "high" if request.thinking and request.thinking.budget_tokens and request.thinking.budget_tokens > 4096 else ("medium" if request.thinking and request.thinking.budget_tokens else None)
        reasoning_str = f" ({reasoning_level})" if reasoning_level else ""
        logger.info(f"⏳ Streaming with {model_name}{reasoning_str}, {len(request.messages)} msgs, {num_tools} tools")
        
        # Create an async function for streaming responses
        async def stream_response():
            try:
                async with httpx.AsyncClient(timeout=600.0) as client:
                    # Set up an asynchronous request
                    async with client.stream("POST", api_url, json=request_dict, headers=headers, timeout=600.0) as response:
                        # Forward status code if there's an error
                        if response.status_code != 200:
                            error_body = await response.aread()
                            error_text = error_body.decode()
                            logger.error(f"Anthropic streaming API error ({response.status_code}): {error_text}")
                            
                            # For auth errors, provide more helpful debugging info
                            if response.status_code == 401:
                                # Show which auth headers were sent (without revealing values)
                                auth_headers_used = []
                                if "authorization" in headers:
                                    auth_headers_used.append("authorization: Bearer ****")
                                if "x-api-key" in headers:
                                    auth_headers_used.append("x-api-key: ****")
                                
                                # Log additional debug info
                                logger.error(f"Streaming authentication error details - Headers used: {auth_headers_used}")
                                logger.error(f"Anthropic model requested for streaming: {model_name}")
                            
                            raise HTTPException(
                                status_code=response.status_code,
                                detail=f"Anthropic API error: {error_text}"
                            )
                        
                        # Stream the response directly
                        async for chunk in response.aiter_bytes():
                            yield chunk
            except Exception as e:
                logger.error(f"Error in streaming response: {str(e)}")
                # Yield error as SSE event
                error_json = json.dumps({"error": {"message": str(e)}})
                yield f"data: {error_json}\n\n".encode('utf-8')
                
        # Return the streaming response
        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        logger.error(f"Error setting up streaming: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error streaming from Anthropic: {str(e)}")


async def handle_litellm(request: MessagesRequest, raw_request: Request, model_name: str):
    """
    Handle requests using LiteLLM (supports OpenAI, Anthropic, and other models).
    
    Args:
        request: The message request
        raw_request: The raw FastAPI request
        model_name: The model name to use with LiteLLM
        
    Returns:
        MessagesResponse or StreamingResponse
        
    Raises:
        HTTPException: If an error occurs with the LiteLLM call
    """
    try:
        # Convert Anthropic format to LiteLLM format
        litellm_request = convert_anthropic_to_litellm(request.dict(exclude_none=True, exclude_unset=True))
        
        # Override the model with the configured model - strip provider prefix for LiteLLM
        clean_model_name = model_name
        if model_name.startswith("openai/"):
            clean_model_name = model_name.replace("openai/", "")
        elif model_name.startswith("anthropic/"):
            clean_model_name = model_name.replace("anthropic/", "")
            
        litellm_request["model"] = clean_model_name
        
        # Extract client headers for API keys
        client_headers = dict(raw_request.headers.items())
        client_headers_normalized = {k.lower(): v for k, v in client_headers.items()}
        client_headers = {**client_headers, **client_headers_normalized}
        
        # Get streaming preference
        is_streaming = request.stream if request.stream is not None else False
        
        # For streaming requests
        if is_streaming:
            return await handle_litellm_streaming(litellm_request, request, model_name)
        
        # For non-streaming requests
        # Log the request details (with sensitive info redacted)
        num_tools = len(request.tools) if request.tools else 0
        reasoning_level = "high" if request.thinking and request.thinking.budget_tokens and request.thinking.budget_tokens > 4096 else ("medium" if request.thinking and request.thinking.budget_tokens else None)
        reasoning_str = f" ({reasoning_level})" if reasoning_level else ""
        logger.info(f"⏳ LiteLLM request with {model_name}{reasoning_str}, {len(request.messages)} msgs, {num_tools} tools")
        
        # Make the LiteLLM call
        start_time = time.time()
        
        # Try to get API keys from headers or environment
        api_key = None
        if model_name.startswith("openai/"):
            api_key = client_headers.get("openai-api-key", os.environ.get("OPENAI_API_KEY"))
        elif model_name.startswith("anthropic/"):
            api_key = client_headers.get("anthropic-api-key", os.environ.get("ANTHROPIC_API_KEY"))
        
        # Make the API call with completion
        try:
            litellm_response = await litellm.acompletion(**litellm_request, api_key=api_key)
        except Exception as e:
            logger.error(f"LiteLLM error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error calling {model_name}: {str(e)}")
        
        # Convert LiteLLM response to Anthropic format
        anthropic_response = convert_litellm_to_anthropic(litellm_response, original_model=request.model)
        
        # Calculate and log timing information
        end_time = time.time()
        elapsed_seconds = end_time - start_time
        tokens_out = anthropic_response.get("usage", {}).get("output_tokens", 0)
        tokens_in = anthropic_response.get("usage", {}).get("input_tokens", 0)
        
        tokens_per_second = tokens_out / elapsed_seconds if elapsed_seconds > 0 else 0
        logger.info(f"✅ Response: {tokens_out} tokens in {elapsed_seconds:.2f}s ({tokens_per_second:.1f} t/s), input: {tokens_in} tokens")
        
        return anthropic_response
        
    except Exception as e:
        logger.error(f"Error in LiteLLM handler: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error with LiteLLM: {str(e)}")


async def handle_litellm_streaming(litellm_request: Dict[str, Any], original_request: MessagesRequest, model_name: str):
    """
    Handle streaming requests via LiteLLM.
    
    Args:
        litellm_request: The request in LiteLLM format
        original_request: The original message request
        model_name: The model name with proper prefix
        
    Returns:
        StreamingResponse
        
    Raises:
        HTTPException: If an error occurs with the LiteLLM streaming
    """
    try:
        # Log the request details
        num_tools = len(original_request.tools) if original_request.tools else 0
        reasoning_level = "high" if original_request.thinking and original_request.thinking.budget_tokens and original_request.thinking.budget_tokens > 4096 else ("medium" if original_request.thinking and original_request.thinking.budget_tokens else None)
        reasoning_str = f" ({reasoning_level})" if reasoning_level else ""
        logger.info(f"⏳ LiteLLM streaming with {model_name}{reasoning_str}, {len(original_request.messages)} msgs, {num_tools} tools")
        
        # Create an async function for streaming
        async def stream_response():
            try:
                # Set the correct model and get an API key if needed
                api_key = None
                clean_model_name = model_name
                
                # Get the correct API key and clean model name
                if model_name.startswith("openai/"):
                    api_key = os.environ.get("OPENAI_API_KEY")
                    clean_model_name = model_name.replace("openai/", "")
                elif model_name.startswith("anthropic/"):
                    api_key = os.environ.get("ANTHROPIC_API_KEY")
                    clean_model_name = model_name.replace("anthropic/", "")
                
                # Make the streaming request
                # Override model in the request to use clean name
                litellm_request["model"] = clean_model_name
                
                stream = await litellm.acompletion(
                    **litellm_request,
                    stream=True,
                    api_key=api_key
                )
                
                async for chunk in stream:
                    # Convert chunk to JSON string and yield as SSE
                    chunk_json = json.dumps(chunk)
                    yield f"data: {chunk_json}\n\n".encode('utf-8')
                
            except Exception as e:
                logger.error(f"Error in LiteLLM streaming: {str(e)}")
                error_json = json.dumps({"error": {"message": str(e)}})
                yield f"data: {error_json}\n\n".encode('utf-8')
        
        # Return streaming response
        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        logger.error(f"Error setting up LiteLLM streaming: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error streaming from LiteLLM: {str(e)}")


async def handle_anthropic_non_streaming(request: MessagesRequest, raw_request: Request, model_name: str):
    """
    Handle non-streaming requests to Anthropic API.
    
    Args:
        request: The message request
        raw_request: The raw FastAPI request
        model_name: The model name with proper prefix
        
    Returns:
        MessagesResponse
        
    Raises:
        HTTPException: If an error occurs with the Anthropic API
    """
    try:
        # Create direct Anthropic request
        api_url = "https://api.anthropic.com/v1/messages"
        
        # Dump the original request to JSON (preserving Anthropic format)
        request_dict = request.dict(exclude_none=True, exclude_unset=True)
        
        # Extract client headers first to avoid reference before assignment
        client_headers = dict(raw_request.headers.items())
        
        # Normalize header keys to lowercase for consistent lookup
        client_headers_normalized = {k.lower(): v for k, v in client_headers.items()}
        client_headers = {**client_headers, **client_headers_normalized}
        
        # Handle model name format based on API used:
        # - Claude Max (auth token present) may use full model name without removing prefixes
        # - Regular Anthropic API needs the "anthropic/" prefix removed
        if "authorization" in client_headers and client_headers["authorization"].startswith("Bearer "):
            # For Claude Max, use the original model requested
            original_model = request.original_model if hasattr(request, "original_model") and request.original_model else request.model
            model_no_prefix = original_model.replace("anthropic/", "")
            request_dict["model"] = model_no_prefix
            logger.debug(f"Using original model for Claude Max: {model_no_prefix}")
        else:
            # Regular Anthropic API needs the "anthropic/" prefix removed
            request_dict["model"] = model_name.replace("anthropic/", "")
            logger.debug(f"Using model for regular API: {request_dict['model']}")
        
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
        
        # Get authentication headers
        headers = get_anthropic_auth_headers(raw_request)
        
        # Log the request details (with sensitive info redacted)
        num_tools = len(request.tools) if request.tools else 0
        reasoning_level = "high" if request.thinking and request.thinking.budget_tokens and request.thinking.budget_tokens > 4096 else ("medium" if request.thinking and request.thinking.budget_tokens else None)
        reasoning_str = f" ({reasoning_level})" if reasoning_level else ""
        logger.info(f"⏳ Request with {model_name}{reasoning_str}, {len(request.messages)} msgs, {num_tools} tools")
        
        # Send the request with timeout
        async with httpx.AsyncClient() as client:
            start_time = time.time()
            response = await client.post(api_url, json=request_dict, headers=headers, timeout=600.0)
            
            # Check for errors
            if response.status_code != 200:
                error_text = response.text
                logger.error(f"Anthropic API error ({response.status_code}): {error_text}")
                
                # For auth errors, provide more helpful debugging info
                if response.status_code == 401:
                    # Show which auth headers were sent (without revealing values)
                    auth_headers_used = []
                    if "authorization" in headers:
                        auth_headers_used.append("authorization: Bearer ****")
                    if "x-api-key" in headers:
                        auth_headers_used.append("x-api-key: ****")
                    
                    # Log additional debug info
                    logger.error(f"Authentication error details - Headers used: {auth_headers_used}")
                    logger.error(f"Anthropic model requested: {model_name}")
                
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Anthropic API error: {error_text}"
                )
            
            # Process response
            anthropic_response = response.json()
            
            # Parse all tool result contents
            if "content" in anthropic_response:
                for idx, content_block in enumerate(anthropic_response["content"]):
                    if content_block.get("type") == "tool_result":
                        if "content" in content_block:
                            anthropic_response["content"][idx]["content"] = parse_tool_result_content(content_block["content"])
            
            # Calculate and log timing information
            end_time = time.time()
            elapsed_seconds = end_time - start_time
            tokens_out = anthropic_response.get("usage", {}).get("output_tokens", 0)
            tokens_in = anthropic_response.get("usage", {}).get("input_tokens", 0)
            
            tokens_per_second = tokens_out / elapsed_seconds if elapsed_seconds > 0 else 0
            logger.info(f"✅ Response: {tokens_out} tokens in {elapsed_seconds:.2f}s ({tokens_per_second:.1f} t/s), input: {tokens_in} tokens")
            
            # Return the formatted response
            return anthropic_response
        
    except Exception as e:
        logger.error(f"Error in non-streaming response: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error getting response from Anthropic: {str(e)}")