"""
Utility functions for model format conversion between Anthropic, OpenAI, and others.
"""
import json
import logging
from typing import Any, Dict, List, Union, Optional

logger = logging.getLogger(__name__)


def convert_anthropic_to_litellm(anthropic_request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert an Anthropic request to LiteLLM format.
    
    Args:
        anthropic_request: Request in Anthropic format
        
    Returns:
        Request in LiteLLM format
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Create a new dictionary for the LiteLLM request
    litellm_request = {
        "model": anthropic_request.get("model"),
        "max_tokens": anthropic_request.get("max_tokens"),
        "temperature": anthropic_request.get("temperature"),
        "stream": anthropic_request.get("stream", False),
    }
    
    # Handle messages and system prompt format based on model
    model = anthropic_request.get("model", "")
    messages = anthropic_request.get("messages", [])
    
    # Special handling for system message with OpenAI models
    has_system = "system" in anthropic_request and anthropic_request["system"]
    
    if model.startswith("openai/"):
        # For OpenAI, convert system prompt to a system message
        if has_system:
            # Add system message to the beginning of messages list
            system_message = {"role": "system", "content": anthropic_request["system"]}
            litellm_request["messages"] = [system_message] + messages
            # Do not include separate system parameter for OpenAI
        else:
            litellm_request["messages"] = messages
        
        logger.debug(f"Converted for OpenAI: System added to messages array")
    else:
        # For other providers like Anthropic, keep both system and messages separate
        litellm_request["messages"] = messages
        
        if has_system:
            litellm_request["system"] = anthropic_request["system"]
    
    # Handle common optional parameters
    if "stop_sequences" in anthropic_request and anthropic_request["stop_sequences"]:
        litellm_request["stop"] = anthropic_request["stop_sequences"]
        
    if "top_p" in anthropic_request and anthropic_request["top_p"] is not None:
        litellm_request["top_p"] = anthropic_request["top_p"]
        
    if "top_k" in anthropic_request and anthropic_request["top_k"] is not None:
        litellm_request["top_k"] = anthropic_request["top_k"]
        
    # Handle metadata (if supported by provider)
    if "metadata" in anthropic_request and anthropic_request["metadata"]:
        if not model.startswith("openai/"):  # OpenAI doesn't support metadata directly
            litellm_request["metadata"] = anthropic_request["metadata"]
    
    # Handle tools
    if "tools" in anthropic_request and anthropic_request["tools"]:
        litellm_request["tools"] = anthropic_request["tools"]
        
    if "tool_choice" in anthropic_request and anthropic_request["tool_choice"]:
        litellm_request["tool_choice"] = anthropic_request["tool_choice"]
    
    # Handle thinking budget if present (provider-specific)
    if "thinking" in anthropic_request and anthropic_request["thinking"]:
        if not model.startswith("openai/"):  # OpenAI doesn't support thinking budget
            thinking_config = anthropic_request["thinking"]
            if thinking_config.get("enabled", True) and thinking_config.get("budget_tokens"):
                litellm_request["thinking_budget"] = thinking_config.get("budget_tokens")
    
    logger.debug(f"Converted request for model {model}: {litellm_request.keys()}")
    return litellm_request


def convert_litellm_to_anthropic(litellm_response: Dict[str, Any], 
                                original_model: str = None) -> Dict[str, Any]:
    """
    Convert a LiteLLM response to Anthropic format.
    
    Args:
        litellm_response: Response from LiteLLM
        original_model: The original model name requested
        
    Returns:
        Response in Anthropic format
    """
    # Extract basics from the LiteLLM response
    anthropic_response = {
        "id": litellm_response.get("id", ""),
        "type": "message",
        "role": "assistant",
        "model": original_model or litellm_response.get("model", ""),
        "stop_reason": litellm_response.get("stop_reason", None),
        "stop_sequence": litellm_response.get("stop_sequence", None),
    }
    
    # Extract content
    content = litellm_response.get("content", "")
    if isinstance(content, str):
        anthropic_response["content"] = [{"type": "text", "text": content}]
    else:
        anthropic_response["content"] = content
    
    # Extract usage information
    usage = litellm_response.get("usage", {})
    anthropic_response["usage"] = {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0)
    }
    
    return anthropic_response


def clean_gemini_schema(schema: Any) -> Any:
    """
    Clean and adapt JSON schema for Gemini compatibility.
    
    Args:
        schema: The original schema
        
    Returns:
        Cleaned schema suitable for Gemini
    """
    if isinstance(schema, dict):
        # Create a new dictionary to avoid modifying the original
        clean_schema = {}
        
        # Copy over compatible properties
        for key, value in schema.items():
            if key in ["additionalProperties", "type", "required", "properties"]:
                clean_schema[key] = value
            elif isinstance(value, (dict, list)):
                clean_schema[key] = clean_gemini_schema(value)
            else:
                clean_schema[key] = value
                
        return clean_schema
    elif isinstance(schema, list):
        return [clean_gemini_schema(item) for item in schema]
    else:
        return schema


def parse_tool_result_content(content):
    """
    Parse tool result content into a standardized format.
    
    Args:
        content: The tool result content
        
    Returns:
        Properly formatted content
    """
    if isinstance(content, str):
        try:
            # Attempt to parse as JSON
            parsed = json.loads(content)
            return parsed
        except json.JSONDecodeError:
            # Return as-is if not valid JSON
            return content
    else:
        # Return as-is if not a string
        return content