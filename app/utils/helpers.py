"""
Helper functions for Claude Code Proxy.
"""
import os
import json
import logging
from typing import Any, Dict, List, Union, Optional

# Environment variables and configuration
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
BIG_MODEL = os.environ.get("BIG_MODEL", "")
SMALL_MODEL = os.environ.get("SMALL_MODEL", "")

logger = logging.getLogger(__name__)


def compute_cost(model: str, in_tokens: int, out_tokens: int) -> float | None:
    """
    Compute the cost of a request based on token usage.
    
    Args:
        model: The model name
        in_tokens: Input token count
        out_tokens: Output token count
        
    Returns:
        Estimated cost in USD or None if pricing not available
    """
    # Standard Claude 3 pricing
    if "claude-3" in model:
        if "haiku" in model:
            return (in_tokens / 1000) * 0.00025 + (out_tokens / 1000) * 0.00125
        elif "sonnet" in model:
            return (in_tokens / 1000) * 0.003 + (out_tokens / 1000) * 0.015
        elif "opus" in model:
            return (in_tokens / 1000) * 0.015 + (out_tokens / 1000) * 0.075
    
    # GPT-4 pricing
    if "gpt-4" in model:
        return (in_tokens / 1000) * 0.03 + (out_tokens / 1000) * 0.06
    
    # GPT-3.5 pricing
    if "gpt-3.5" in model:
        return (in_tokens / 1000) * 0.0015 + (out_tokens / 1000) * 0.002
    
    # Return None for unknown models
    return None


def clean_gemini_schema(schema: Any) -> Any:
    """
    Clean and adapt JSON schema for Gemini compatibility.
    
    Args:
        schema: The original schema
        
    Returns:
        Cleaned schema suitable for Gemini
    """
    if isinstance(schema, dict):
        # Remove problematic keywords
        for key in list(schema.keys()):
            if key in ["additionalProperties", "type", "required", "properties"]:
                continue
                
            # Handle nested structures
            if isinstance(schema[key], (dict, list)) and key not in ["type", "enum"]:
                schema[key] = clean_gemini_schema(schema[key])
                
        return schema
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


def convert_anthropic_to_litellm(anthropic_request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert Anthropic request format to LiteLLM format.
    
    Args:
        anthropic_request: Request in Anthropic format
        
    Returns:
        Request in LiteLLM format
    """
    # Implementation depends on the specific conversion logic in your application
    # This is a placeholder that should be customized based on your actual code
    return {
        "model": anthropic_request.get("model", ""),
        "messages": anthropic_request.get("messages", []),
        # Add other fields as needed
    }