"""
Test fixtures for Claude Code Proxy tests.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.environ.get("ANTHROPIC_API_KEY")
PROXY_API_URL = "http://localhost:8082/v1/messages"
API_VERSION = "2023-06-01"
DIRECT_MODEL = "openai/gpt-4o"  # Using known-supported models
PREFIXED_MODEL = "openai/gpt-4o-mini"  # Test with explicit provider prefix

# Headers
api_headers = {
    "x-api-key": API_KEY,
    "anthropic-version": API_VERSION,
    "content-type": "application/json",
}

# Tool definitions
calculator_tool = {
    "name": "calculator",
    "description": "Evaluate mathematical expressions",
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate"
            }
        },
        "required": ["expression"]
    }
}

weather_tool = {
    "name": "weather",
    "description": "Get weather information for a location",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city or location to get weather for"
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature units"
            }
        },
        "required": ["location"]
    }
}

search_tool = {
    "name": "search",
    "description": "Search for information on the web",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string", 
                "description": "The search query"
            }
        },
        "required": ["query"]
    }
}

# Test scenarios
TEST_SCENARIOS = {
    # Simple text response
    "simple_direct": {
        "model": DIRECT_MODEL,
        "max_tokens": 300,
        "messages": [
            {"role": "user", "content": "Hello, world! Can you tell me about Paris in 2-3 sentences?"}
        ]
    },
    
    # Test with explicit provider prefix
    "simple_prefixed": {
        "model": PREFIXED_MODEL,
        "max_tokens": 300,
        "messages": [
            {"role": "user", "content": "Hello, world! Can you tell me about London in 2-3 sentences?"}
        ]
    },
    
    # Basic tool use
    "calculator": {
        "model": DIRECT_MODEL,
        "max_tokens": 300,
        "messages": [
            {"role": "user", "content": "What is 135 + 7.5 divided by 2.5?"}
        ],
        "tools": [calculator_tool],
        "tool_choice": {"type": "auto"}
    },
    
    # Simple streaming test
    "simple_stream": {
        "model": DIRECT_MODEL,
        "max_tokens": 100,
        "stream": True,
        "messages": [
            {"role": "user", "content": "Count from 1 to 5, with one number per line."}
        ]
    },
    
    # Streaming with provider prefix
    "prefixed_stream": {
        "model": PREFIXED_MODEL,
        "max_tokens": 300,  # Increased to avoid token limitation errors
        "stream": True,
        "messages": [
            {"role": "user", "content": "Count from 5 to 10, with one number per line."}
        ]
    }
}

# Required event types for Anthropic streaming responses
REQUIRED_EVENT_TYPES = {
    "message_start", 
    "content_block_start", 
    "content_block_delta", 
    "content_block_stop", 
    "message_delta", 
    "message_stop"
}