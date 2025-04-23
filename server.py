from fastapi import FastAPI, Request, HTTPException
import uvicorn
import logging
import json
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Union, Literal
import httpx
import os
from fastapi.responses import JSONResponse, StreamingResponse
import litellm
import uuid
import time
from dotenv import load_dotenv
import re
from datetime import datetime
import sys

# Load environment variables from .env file
load_dotenv()

# Cost per token fallback rates
PRICES = {
    "openai/o3":      {"input": 10.0/1e6, "output": 40.0/1e6},
    "openai/o4-mini": {"input": 1.10/1e6, "output": 4.40/1e6},
}

def compute_cost(model: str, in_tokens: int, out_tokens: int) -> float | None:
    """Return approximate USD cost for a request.

    Priority order:
    1. Hard‑coded fallback prices in the *PRICES* table (overrides anything else).
    2. LiteLLM’s built‑in ``cost_per_token`` helper for every other model.
    3. ``None`` if no pricing data are available.
    """
    # 1) Hard‑coded overrides
    for key, rates in PRICES.items():
        if model.startswith(key):
            return in_tokens * rates["input"] + out_tokens * rates["output"]

    # 2) Ask LiteLLM for the pricing of any other model
    try:
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=in_tokens,
            completion_tokens=out_tokens,
        )
        total = (prompt_cost or 0) + (completion_cost or 0)
        return total if total > 0 else None  # treat 0 as "unknown / free"
    except Exception:
        # 3) Unsupported model or unexpected error – silently give up
        return None

# Configure logging based on environment variable
log_level_str = os.environ.get("LOGLEVEL", "WARN").upper()
log_level = getattr(logging, log_level_str, logging.WARN)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)
logger.info(f"Logger initialized with level: {log_level_str}")

# Check if we should show detailed model information
SHOW_MODEL_DETAILS = os.environ.get("SHOW_MODEL_DETAILS", "").lower() == "true"
if SHOW_MODEL_DETAILS:
    logger.info("Model details logging enabled")

# Enable LiteLLM debugging only in full debug mode
if log_level == logging.DEBUG:
    import litellm
    litellm._turn_on_debug()
    logger.info("LiteLLM debug mode enabled")

# Configure uvicorn to be quieter
import uvicorn
# Tell uvicorn's loggers to be quiet
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

# Create a filter to block any log messages containing specific strings
class MessageFilter(logging.Filter):
    def filter(self, record):
        # Block messages containing these strings
        blocked_phrases = [
            "LiteLLM completion()",
            "HTTP Request:", 
            "selected model name for cost calculation",
            "utils.py",
            "cost_calculator"
        ]
        
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            for phrase in blocked_phrases:
                if phrase in record.msg:
                    return False
        return True

# Apply the filter to the root logger to catch all messages
root_logger = logging.getLogger()
root_logger.addFilter(MessageFilter())

# Custom formatter for model mapping logs
class ColorizedFormatter(logging.Formatter):
    """Custom formatter to highlight model mappings"""
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    def format(self, record):
        if record.levelno == logging.debug and "MODEL MAPPING" in record.msg:
            # Apply colors and formatting to model mapping logs
            return f"{self.BOLD}{self.GREEN}{record.msg}{self.RESET}"
        return super().format(record)

# Apply custom formatter to console handler
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setFormatter(ColorizedFormatter('%(asctime)s - %(levelname)s - %(message)s'))

app = FastAPI()

# Get API keys from environment
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Get model mapping configuration from environment
# These should include provider prefixes (e.g., "openai/gpt-4.1")
BIG_MODEL = os.environ.get("BIG_MODEL", "openai/gpt-4.1")
SMALL_MODEL = os.environ.get("SMALL_MODEL", "openai/gpt-4.1-mini")
THINKER_MODEL = os.environ.get("THINKER_MODEL", "openai/gpt-4o")

# Helper function to clean schema for Gemini
def clean_gemini_schema(schema: Any) -> Any:
    """Recursively removes unsupported fields from a JSON schema for Gemini."""
    if isinstance(schema, dict):
        # Remove specific keys unsupported by Gemini tool parameters
        schema.pop("additionalProperties", None)
        schema.pop("default", None)

        # Handle Union types - Gemini only supports Optional types
        if "type" in schema and isinstance(schema["type"], list):
            # Check if this is an Optional type (list with "null" as one of the types)
            types = schema["type"]
            if "null" in types and len(types) == 2:
                # Keep the non-null type for Optional[Type]
                non_null_type = next(t for t in types if t != "null")
                schema["type"] = non_null_type
                # Add nullable flag
                schema["nullable"] = True
            else:
                # Non-Optional union types not supported - convert to string
                logger.debug(f"Converting unsupported union type {types} to string for Gemini compatibility")
                schema["type"] = "string"
                schema.pop("items", None)  # Remove items if present
                schema.pop("properties", None)  # Remove properties if present

        # Check for unsupported 'format' in string types
        if schema.get("type") == "string" and "format" in schema:
            allowed_formats = {"enum", "date-time"}
            if schema["format"] not in allowed_formats:
                logger.debug(f"Removing unsupported format '{schema['format']}' for string type in Gemini schema.")
                schema.pop("format")

        # Special handling for anyOf, oneOf, allOf - not well supported by Gemini
        for union_key in ["anyOf", "oneOf", "allOf"]:
            if union_key in schema:
                logger.debug(f"Converting unsupported {union_key} to string type for Gemini compatibility")
                schema.pop(union_key)
                schema["type"] = "string"
                schema.pop("items", None)
                schema.pop("properties", None)

        # Recursively clean nested schemas (properties, items, etc.)
        for key, value in list(schema.items()): # Use list() to allow modification during iteration
            schema[key] = clean_gemini_schema(value)
    elif isinstance(schema, list):
        # Recursively clean items in a list
        return [clean_gemini_schema(item) for item in schema]
    return schema

# Models for Anthropic API requests
class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str

class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]

class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]

class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]

class SystemContent(BaseModel):
    type: Literal["text"]
    text: str

class Message(BaseModel):
    role: Literal["user", "assistant"] 
    content: Union[str, List[Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult]]]

class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

class ThinkingConfig(BaseModel):
    enabled: bool = True
    type: Optional[str] = None
    budget_tokens: Optional[int] = None

class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None
    original_model: Optional[str] = None  # Will store the original model name
    
    @field_validator('model')
    def validate_model_field(cls, v, info): # Renamed to avoid conflict
        original_model = v
        new_model = v # Default to original value

        logger.debug(f"📋 MODEL VALIDATION: Original='{original_model}', BIG='{BIG_MODEL}', SMALL='{SMALL_MODEL}'")

        # Remove provider prefixes for easier matching
        clean_v = v
        if clean_v.startswith('anthropic/'):
            clean_v = clean_v[10:]
        elif clean_v.startswith('openai/'):
            clean_v = clean_v[7:]
        elif clean_v.startswith('gemini/'):
            clean_v = clean_v[7:]

        # --- Mapping Logic --- START ---
        mapped = False
        
        # Special handling for passthrough mode - preserve original model
        if BIG_MODEL == "passthrough" and ('claude' in clean_v.lower() or v.startswith('anthropic/')):
            # Keep the original model for Anthropic models when in passthrough mode
            logger.debug(f"Passthrough mode: preserving original model '{v}'")
            new_model = v
            if not v.startswith('anthropic/') and 'claude' in v.lower():
                new_model = f"anthropic/{v}"
                logger.debug(f"Adding anthropic/ prefix to model: '{v}' -> '{new_model}'")
            mapped = True
        # Map Haiku to SMALL_MODEL
        elif 'haiku' in clean_v.lower():
            new_model = SMALL_MODEL
            mapped = True
        # Map Sonnet to BIG_MODEL
        elif 'sonnet' in clean_v.lower():
            new_model = BIG_MODEL
            mapped = True
        # --- Mapping Logic --- END ---

        if mapped:
            # Check if thinking is in the values dictionary for reasoning level
            values = info.data
            thinking_budget = None
            if isinstance(values, dict) and 'thinking' in values and values['thinking']:
                thinking_budget = getattr(values['thinking'], 'budget_tokens', None)
            
            # Add reasoning tier indicators if present
            reasoning_level = ""
            if thinking_budget:
                reasoning_level = " (high)" if thinking_budget > 4096 else " (medium)"
            
            logger.debug(f"📌 MODEL MAPPING: '{original_model}{reasoning_level}' ➡️ '{new_model}{reasoning_level}'")
        else:
            # Don't modify the model name if it's already in provider/model format
            if '/' in v:
                new_model = v  # Keep as is if it already has a provider prefix
                logger.debug(f"Using provider-specified model: '{v}'")
            else:
                # For models without a prefix, add openai/ prefix
                new_model = f"openai/{v}"
                logger.debug(f"Adding openai/ prefix to model: '{v}' -> '{new_model}'")
                mapped = True # Mark as mapped

        # Store the original model in the values dictionary
        values = info.data
        if isinstance(values, dict):
            values['original_model'] = original_model

        return new_model

class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None  # Will store the original model name
    
    @field_validator('model')
    def validate_model_token_count(cls, v, info): # Renamed to avoid conflict
        # Use the same logic as MessagesRequest validator
        # NOTE: Pydantic validators might not share state easily if not class methods
        # Re-implementing the logic here for clarity, could be refactored
        original_model = v
        new_model = v # Default to original value

        logger.debug(f"📋 TOKEN COUNT VALIDATION: Original='{original_model}', BIG='{BIG_MODEL}', SMALL='{SMALL_MODEL}'")

        # Remove provider prefixes for easier matching
        clean_v = v
        if clean_v.startswith('anthropic/'):
            clean_v = clean_v[10:]
        elif clean_v.startswith('openai/'):
            clean_v = clean_v[7:]
        elif clean_v.startswith('gemini/'):
            clean_v = clean_v[7:]

        # --- Mapping Logic --- START ---
        mapped = False
        
        # Special handling for passthrough mode - preserve original model
        if BIG_MODEL == "passthrough" and ('claude' in clean_v.lower() or v.startswith('anthropic/')):
            # Keep the original model for Anthropic models when in passthrough mode
            logger.debug(f"Passthrough mode: preserving original model '{v}'")
            new_model = v
            if not v.startswith('anthropic/') and 'claude' in v.lower():
                new_model = f"anthropic/{v}"
                logger.debug(f"Adding anthropic/ prefix to model: '{v}' -> '{new_model}'")
            mapped = True
        # Map Haiku to SMALL_MODEL
        elif 'haiku' in clean_v.lower():
            new_model = SMALL_MODEL
            mapped = True
        # Map Sonnet to BIG_MODEL
        elif 'sonnet' in clean_v.lower():
            new_model = BIG_MODEL
            mapped = True
        # --- Mapping Logic --- END ---

        if mapped:
            # Check if thinking is in the values dictionary for reasoning level
            values = info.data
            thinking_budget = None
            if isinstance(values, dict) and 'thinking' in values and values['thinking']:
                thinking_budget = getattr(values['thinking'], 'budget_tokens', None)
            
            # Add reasoning tier indicators if present
            reasoning_level = ""
            if thinking_budget:
                reasoning_level = " (high)" if thinking_budget > 4096 else " (medium)"
            
            logger.debug(f"📌 TOKEN COUNT MAPPING: '{original_model}{reasoning_level}' ➡️ '{new_model}{reasoning_level}'")
        else:
            # Don't modify the model name if it's already in provider/model format
            if '/' in v:
                new_model = v  # Keep as is if it already has a provider prefix
                logger.debug(f"Using provider-specified model: '{v}'")
            else:
                # For models without a prefix, add openai/ prefix
                new_model = f"openai/{v}"
                logger.debug(f"Adding openai/ prefix to model: '{v}' -> '{new_model}'")
                mapped = True # Mark as mapped

        # Store the original model in the values dictionary
        values = info.data
        if isinstance(values, dict):
            values['original_model'] = original_model

        return new_model

class TokenCountResponse(BaseModel):
    input_tokens: int

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: Optional[float] = None

class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Get request details
    method = request.method
    path = request.url.path
    
    # Log only basic request details at debug level
    logger.debug(f"Request: {method} {path}")
    
    # Process the request and get the response
    response = await call_next(request)
    
    return response

# Not using validation function as we're using the environment API key

def parse_tool_result_content(content):
    """Helper function to properly parse and normalize tool result content."""
    if content is None:
        return "No content provided"
        
    if isinstance(content, str):
        return content
        
    if isinstance(content, list):
        result = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                result += item.get("text", "") + "\n"
            elif isinstance(item, str):
                result += item + "\n"
            elif isinstance(item, dict):
                if "text" in item:
                    result += item.get("text", "") + "\n"
                else:
                    try:
                        result += json.dumps(item) + "\n"
                    except:
                        result += str(item) + "\n"
            else:
                try:
                    result += str(item) + "\n"
                except:
                    result += "Unparseable content\n"
        return result.strip()
        
    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        try:
            return json.dumps(content)
        except:
            return str(content)
            
    # Fallback for any other type
    try:
        return str(content)
    except:
        return "Unparseable content"

def convert_anthropic_to_litellm(anthropic_request: MessagesRequest) -> Dict[str, Any]:
    """Convert Anthropic API request format to LiteLLM format (which follows OpenAI)."""
    # LiteLLM already handles Anthropic models when using the format model="anthropic/claude-3-opus-20240229"
    # So we just need to convert our Pydantic model to a dict in the expected format
    
    messages = []
    
    # Add system message if present
    if anthropic_request.system:
        # Handle different formats of system messages
        if isinstance(anthropic_request.system, str):
            # Simple string format
            messages.append({"role": "system", "content": anthropic_request.system})
        elif isinstance(anthropic_request.system, list):
            # List of content blocks
            system_text = ""
            for block in anthropic_request.system:
                if hasattr(block, 'type') and block.type == "text":
                    system_text += block.text + "\n\n"
                elif isinstance(block, dict) and block.get("type") == "text":
                    system_text += block.get("text", "") + "\n\n"
            
            if system_text:
                messages.append({"role": "system", "content": system_text.strip()})
    
    # Add conversation messages
    for idx, msg in enumerate(anthropic_request.messages):
        content = msg.content
        if isinstance(content, str):
            messages.append({"role": msg.role, "content": content})
        else:
            # Special handling for tool_result in user messages
            # OpenAI/LiteLLM format expects the assistant to call the tool, 
            # and the user's next message to include the result as plain text
            if msg.role == "user" and any(block.type == "tool_result" for block in content if hasattr(block, "type")):
                # For user messages with tool_result, split into separate messages
                text_content = ""
                
                # Extract all text parts and concatenate them
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            text_content += block.text + "\n"
                        elif block.type == "tool_result":
                            # Add tool result as a message by itself - simulate the normal flow
                            tool_id = block.tool_use_id if hasattr(block, "tool_use_id") else ""
                            
                            # Handle different formats of tool result content
                            result_content = ""
                            if hasattr(block, "content"):
                                if isinstance(block.content, str):
                                    result_content = block.content
                                elif isinstance(block.content, list):
                                    # If content is a list of blocks, extract text from each
                                    for content_block in block.content:
                                        if hasattr(content_block, "type") and content_block.type == "text":
                                            result_content += content_block.text + "\n"
                                        elif isinstance(content_block, dict) and content_block.get("type") == "text":
                                            result_content += content_block.get("text", "") + "\n"
                                        elif isinstance(content_block, dict):
                                            # Handle any dict by trying to extract text or convert to JSON
                                            if "text" in content_block:
                                                result_content += content_block.get("text", "") + "\n"
                                            else:
                                                try:
                                                    result_content += json.dumps(content_block) + "\n"
                                                except:
                                                    result_content += str(content_block) + "\n"
                                elif isinstance(block.content, dict):
                                    # Handle dictionary content
                                    if block.content.get("type") == "text":
                                        result_content = block.content.get("text", "")
                                    else:
                                        try:
                                            result_content = json.dumps(block.content)
                                        except:
                                            result_content = str(block.content)
                                else:
                                    # Handle any other type by converting to string
                                    try:
                                        result_content = str(block.content)
                                    except:
                                        result_content = "Unparseable content"
                            
                            # In OpenAI format, tool results come from the user (rather than being content blocks)
                            text_content += f"Tool result for {tool_id}:\n{result_content}\n"
                
                # Add as a single user message with all the content
                messages.append({"role": "user", "content": text_content.strip()})
            else:
                # Regular handling for other message types
                processed_content = []
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            processed_content.append({"type": "text", "text": block.text})
                        elif block.type == "image":
                            processed_content.append({"type": "image", "source": block.source})
                        elif block.type == "tool_use":
                            # Handle tool use blocks if needed
                            processed_content.append({
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input": block.input
                            })
                        elif block.type == "tool_result":
                            # Handle different formats of tool result content
                            processed_content_block = {
                                "type": "tool_result",
                                "tool_use_id": block.tool_use_id if hasattr(block, "tool_use_id") else ""
                            }
                            
                            # Process the content field properly
                            if hasattr(block, "content"):
                                if isinstance(block.content, str):
                                    # If it's a simple string, create a text block for it
                                    processed_content_block["content"] = [{"type": "text", "text": block.content}]
                                elif isinstance(block.content, list):
                                    # If it's already a list of blocks, keep it
                                    processed_content_block["content"] = block.content
                                else:
                                    # Default fallback
                                    processed_content_block["content"] = [{"type": "text", "text": str(block.content)}]
                            else:
                                # Default empty content
                                processed_content_block["content"] = [{"type": "text", "text": ""}]
                                
                            processed_content.append(processed_content_block)
                
                messages.append({"role": msg.role, "content": processed_content})
    
    # Use client-provided max_tokens value by default
    max_tokens = anthropic_request.max_tokens
    
    # For logging only - don't modify the value
    if SHOW_MODEL_DETAILS:
        logger.info(f"Client requested max_tokens: {max_tokens}")
    
    # Create LiteLLM request dict
    litellm_request = {
        "model": anthropic_request.model,  # t understands "anthropic/claude-x" format
        "messages": messages,
        "stream": anthropic_request.stream,
    }
    
    # Special handling for o3 and o4 models
    if anthropic_request.model.startswith("openai/o3") or anthropic_request.model.startswith("openai/o4"):
        # OpenAI o3/o4 models use max_completion_tokens instead of max_tokens
        # For Claude Code, we should fully respect the client's token request
        litellm_request["max_completion_tokens"] = max_tokens
        
        if SHOW_MODEL_DETAILS:
            logger.info(f"Using max_completion_tokens={max_tokens} for {anthropic_request.model}")
        
        # Override system prompt for o3 models to experiment with its behavior
        if anthropic_request.model.startswith("openai/o3"):
            # Log the original system message if any
            original_system = None
            system_msg_idx = -1
            
            # Find any existing system message
            for idx, msg in enumerate(litellm_request["messages"]):
                if msg.get("role") == "system":
                    system_msg_idx = idx
                    original_system = msg.get("content")
                    if original_system:
                        # Print the full system prompt split into reasonably sized chunks for logging
                        logger.info(f"📥 ORIGINAL SYSTEM PROMPT (FULL):")
                        # Split into chunks of around 1000 chars for readability in logs
                        chunk_size = 1000
                        for i in range(0, len(original_system), chunk_size):
                            chunk = original_system[i:i+chunk_size]
                            logger.info(f"SYSTEM PROMPT PART {i//chunk_size + 1}: {chunk}")
                    break
            
            # Instead of replacing, let's modify the existing system prompt by injecting personality guidelines
            if original_system:
                # Locate the tone instruction about being concise
                concise_pattern = r"You should be concise, direct, and to the point"
                
                replacement = "You should be concise, direct, to the point, and also friendly and a good coworker"
                
                modified_system_prompt = re.sub(concise_pattern, replacement, original_system)
                
                # Remove overly strict brevity constraints
                brevity_patterns = [
                    r"You MUST answer concisely with fewer than 4 lines[^\n]*\n?",
                    r"One word answers are best\.?:?[^\n]*\n?"
                ]
                for pat in brevity_patterns:
                    modified_system_prompt = re.sub(pat, "", modified_system_prompt, flags=re.IGNORECASE)
                
                # Inject softer, friendlier guidance
                personality_addendum = """

ADDITIONAL GUIDANCE:
- Use a warm, conversational, empathetic tone.
- Offer positive affirmations and examples.
- Invite follow-up questions and collaboration.
- Remain supportive, clear, and patient.
- Show eagerness to implement changes and iterate quickly."""
                
                if "ADDITIONAL GUIDANCE:" not in modified_system_prompt:
                    modified_system_prompt += personality_addendum
                
                # Check if we actually made a change
                if modified_system_prompt != original_system:
                    logger.info(f"🔄 Modified system prompt: Added friendliness and personality guidance")
                    
                    # Update the system message with our modified version
                    litellm_request["messages"][system_msg_idx]["content"] = modified_system_prompt
                else:
                    logger.info(f"⚠️ Could not modify system prompt - pattern not found")
            else:
                # No system prompt found, create a simple one
                simple_prompt = "You are Claude, an AI assistant. You should be concise, direct, to the point, and also friendly and a good coworker. Provide useful, informative responses."
                litellm_request["messages"].insert(0, {"role": "system", "content": simple_prompt})
                logger.info(f"➕ Added a simple system prompt (no original found)")
                
            # Log all messages being sent (truncated for readability)
            for i, msg in enumerate(litellm_request["messages"]):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if isinstance(content, str):
                    content_preview = content[:100] + "..." if len(content) > 100 else content
                    logger.info(f"📩 MESSAGE {i}: role={role}, content={content_preview}")
        
        # o3 and o4 models don't support custom temperature - they only support the default (1.0)
        # Only add temperature if it's the default value of 1.0
        if anthropic_request.temperature == 1.0:
            litellm_request["temperature"] = 1.0
    else:
        # For other models, use standard parameters
        litellm_request["max_tokens"] = max_tokens
        litellm_request["temperature"] = anthropic_request.temperature
    
    # Add optional parameters if present
    if anthropic_request.stop_sequences:
        litellm_request["stop"] = anthropic_request.stop_sequences
    
    if anthropic_request.top_p:
        litellm_request["top_p"] = anthropic_request.top_p
    
    if anthropic_request.top_k:
        litellm_request["top_k"] = anthropic_request.top_k
    
    # Convert tools to OpenAI format
    if anthropic_request.tools:
        openai_tools = []
        is_gemini_model = anthropic_request.model.startswith("gemini/")

        for tool in anthropic_request.tools:
            # Convert to dict if it's a pydantic model
            if hasattr(tool, 'dict'):
                tool_dict = tool.dict()
            else:
                # Ensure tool_dict is a dictionary, handle potential errors if 'tool' isn't dict-like
                try:
                    tool_dict = dict(tool) if not isinstance(tool, dict) else tool
                except (TypeError, ValueError):
                     logger.error(f"Could not convert tool to dict: {tool}")
                     continue # Skip this tool if conversion fails

            # Clean the schema if targeting a Gemini model
            input_schema = tool_dict.get("input_schema", {})
            if is_gemini_model:
                 logger.debug(f"Cleaning schema for Gemini tool: {tool_dict.get('name')}")
                 input_schema = clean_gemini_schema(input_schema)

            # Create OpenAI-compatible function tool
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool_dict["name"],
                    "description": tool_dict.get("description", ""),
                    "parameters": input_schema # Use potentially cleaned schema
                }
            }
            openai_tools.append(openai_tool)

        litellm_request["tools"] = openai_tools
    
    # Convert tool_choice to OpenAI format if present
    if anthropic_request.tool_choice:
        if hasattr(anthropic_request.tool_choice, 'dict'):
            tool_choice_dict = anthropic_request.tool_choice.dict()
        else:
            tool_choice_dict = anthropic_request.tool_choice
            
        # Handle Anthropic's tool_choice format
        choice_type = tool_choice_dict.get("type")
        if choice_type == "auto":
            litellm_request["tool_choice"] = "auto"
        elif choice_type == "any":
            litellm_request["tool_choice"] = "any"
        elif choice_type == "tool" and "name" in tool_choice_dict:
            litellm_request["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice_dict["name"]}
            }
        else:
            # Default to auto if we can't determine
            litellm_request["tool_choice"] = "auto"
    
    return litellm_request

def convert_litellm_to_anthropic(litellm_response: Union[Dict[str, Any], Any], 
                                 original_request: MessagesRequest) -> MessagesResponse:
    """Convert LiteLLM (OpenAI format) response to Anthropic API response format."""
    
    # Enhanced response extraction with better error handling
    try:
        # Get the clean model name to check capabilities
        clean_model = original_request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/"):]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/"):]
        
        # Check if this is a Claude model (which supports content blocks)
        is_claude_model = clean_model.startswith("claude-")
        
        # Handle ModelResponse object from LiteLLM
        if hasattr(litellm_response, 'choices') and hasattr(litellm_response, 'usage'):
            # Extract data from ModelResponse object directly
            choices = litellm_response.choices
            message = choices[0].message if choices and len(choices) > 0 else None
            content_text = message.content if message and hasattr(message, 'content') else ""
            tool_calls = message.tool_calls if message and hasattr(message, 'tool_calls') else None
            finish_reason = choices[0].finish_reason if choices and len(choices) > 0 else "stop"
            usage_info = litellm_response.usage
            response_id = getattr(litellm_response, 'id', f"msg_{uuid.uuid4()}")
        else:
            # For backward compatibility - handle dict responses
            # If response is a dict, use it, otherwise try to convert to dict
            try:
                response_dict = litellm_response if isinstance(litellm_response, dict) else litellm_response.dict()
            except AttributeError:
                # If .dict() fails, try to use model_dump or __dict__ 
                try:
                    response_dict = litellm_response.model_dump() if hasattr(litellm_response, 'model_dump') else litellm_response.__dict__
                except AttributeError:
                    # Fallback - manually extract attributes
                    response_dict = {
                        "id": getattr(litellm_response, 'id', f"msg_{uuid.uuid4()}"),
                        "choices": getattr(litellm_response, 'choices', [{}]),
                        "usage": getattr(litellm_response, 'usage', {})
                    }
                    
            # Extract the content from the response dict
            choices = response_dict.get("choices", [{}])
            message = choices[0].get("message", {}) if choices and len(choices) > 0 else {}
            content_text = message.get("content", "")
            tool_calls = message.get("tool_calls", None)
            finish_reason = choices[0].get("finish_reason", "stop") if choices and len(choices) > 0 else "stop"
            usage_info = response_dict.get("usage", {})
            response_id = response_dict.get("id", f"msg_{uuid.uuid4()}")
        
        # Create content list for Anthropic format
        content = []
        
        # Add text content block if present (text might be None or empty for pure tool call responses)
        if content_text is not None and content_text != "":
            content.append({"type": "text", "text": content_text})
        
        # Add tool calls if present (tool_use in Anthropic format) - only for Claude models
        if tool_calls and is_claude_model:
            logger.debug(f"Processing tool calls: {tool_calls}")
            
            # Convert to list if it's not already
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]
                
            for idx, tool_call in enumerate(tool_calls):
                logger.debug(f"Processing tool call {idx}: {tool_call}")
                
                # Extract function data based on whether it's a dict or object
                if isinstance(tool_call, dict):
                    function = tool_call.get("function", {})
                    tool_id = tool_call.get("id", f"tool_{uuid.uuid4()}")
                    name = function.get("name", "")
                    arguments = function.get("arguments", "{}")
                else:
                    function = getattr(tool_call, "function", None)
                    tool_id = getattr(tool_call, "id", f"tool_{uuid.uuid4()}")
                    name = getattr(function, "name", "") if function else ""
                    arguments = getattr(function, "arguments", "{}") if function else "{}"
                
                # Convert string arguments to dict if needed
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse tool arguments as JSON: {arguments}")
                        arguments = {"raw": arguments}
                
                logger.debug(f"Adding tool_use block: id={tool_id}, name={name}, input={arguments}")
                
                content.append({
                    "type": "tool_use",
                    "id": tool_id,
                    "name": name,
                    "input": arguments
                })
        elif tool_calls and not is_claude_model:
            # For non-Claude models, convert tool calls to text format
            logger.debug(f"Converting tool calls to text for non-Claude model: {clean_model}")
            
            # We'll append tool info to the text content
            tool_text = "\n\nTool usage:\n"
            
            # Convert to list if it's not already
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]
                
            for idx, tool_call in enumerate(tool_calls):
                # Extract function data based on whether it's a dict or object
                if isinstance(tool_call, dict):
                    function = tool_call.get("function", {})
                    tool_id = tool_call.get("id", f"tool_{uuid.uuid4()}")
                    name = function.get("name", "")
                    arguments = function.get("arguments", "{}")
                else:
                    function = getattr(tool_call, "function", None)
                    tool_id = getattr(tool_call, "id", f"tool_{uuid.uuid4()}")
                    name = getattr(function, "name", "") if function else ""
                    arguments = getattr(function, "arguments", "{}") if function else "{}"
                
                # Convert string arguments to dict if needed
                if isinstance(arguments, str):
                    try:
                        args_dict = json.loads(arguments)
                        arguments_str = json.dumps(args_dict, indent=2)
                    except json.JSONDecodeError:
                        arguments_str = arguments
                else:
                    arguments_str = json.dumps(arguments, indent=2)
                
                tool_text += f"Tool: {name}\nArguments: {arguments_str}\n\n"
            
            # Add or append tool text to content
            if content and content[0]["type"] == "text":
                content[0]["text"] += tool_text
            else:
                content.append({"type": "text", "text": tool_text})
        
        # Get usage information - extract values safely from object or dict
        if isinstance(usage_info, dict):
            prompt_tokens = usage_info.get("prompt_tokens", 0)
            completion_tokens = usage_info.get("completion_tokens", 0)
        else:
            prompt_tokens = getattr(usage_info, "prompt_tokens", 0)
            completion_tokens = getattr(usage_info, "completion_tokens", 0)
        
        # Map OpenAI finish_reason to Anthropic stop_reason
        stop_reason = None
        if finish_reason == "stop":
            stop_reason = "end_turn"
        elif finish_reason == "length":
            stop_reason = "max_tokens"
        elif finish_reason == "tool_calls":
            stop_reason = "tool_use"
        else:
            stop_reason = "end_turn"  # Default
        
        # Make sure content is never empty
        if not content:
            content.append({"type": "text", "text": ""})
        
        # Create Anthropic-style response
        anthropic_response = MessagesResponse(
            id=response_id,
            model=original_request.model,
            role="assistant",
            content=content,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=Usage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens
            )
        )
        
        return anthropic_response
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_message = f"Error converting response: {str(e)}\n\nFull traceback:\n{error_traceback}"
        logger.error(error_message)
        
        # In case of any error, create a fallback response
        return MessagesResponse(
            id=f"msg_{uuid.uuid4()}",
            model=original_request.model,
            role="assistant",
            content=[{"type": "text", "text": f"Error converting response: {str(e)}. Please check server logs."}],
            stop_reason="end_turn",
            usage=Usage(input_tokens=0, output_tokens=0)
        )

async def handle_streaming(response_generator, original_request: MessagesRequest):
    """Handle streaming responses from LiteLLM and convert to Anthropic format."""
    try:
        # Send message_start event
        message_id = f"msg_{uuid.uuid4().hex[:24]}"  # Format similar to Anthropic's IDs
        
        message_data = {
            'type': 'message_start',
            'message': {
                'id': message_id,
                'type': 'message',
                'role': 'assistant',
                'model': original_request.model,
                'content': [],
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {
                    'input_tokens': 0,
                    'cache_creation_input_tokens': 0,
                    'cache_read_input_tokens': 0,
                    'output_tokens': 0
                }
            }
        }
        yield f"event: message_start\ndata: {json.dumps(message_data)}\n\n"
        
        # Content block index for the first text block
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
        
        # Send a ping to keep the connection alive (Anthropic does this)
        yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"
        
        tool_index = None
        current_tool_call = None
        tool_content = ""
        accumulated_text = ""  # Track accumulated text content
        text_sent = False  # Track if we've sent any text content
        text_block_closed = False  # Track if text block is closed
        input_tokens = 0
        output_tokens = 0
        has_sent_stop_reason = False
        last_tool_index = 0
        
        # Process each chunk
        async for chunk in response_generator:
            try:

                
                # Check if this is the end of the response with usage data
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    if hasattr(chunk.usage, 'prompt_tokens'):
                        input_tokens = chunk.usage.prompt_tokens
                    if hasattr(chunk.usage, 'completion_tokens'):
                        output_tokens = chunk.usage.completion_tokens
                
                # Handle text content
                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    
                    # Get the delta from the choice
                    if hasattr(choice, 'delta'):
                        delta = choice.delta
                    else:
                        # If no delta, try to get message
                        delta = getattr(choice, 'message', {})
                    
                    # Check for finish_reason to know when we're done
                    finish_reason = getattr(choice, 'finish_reason', None)
                    
                    # Process text content
                    delta_content = None
                    
                    # Handle different formats of delta content
                    if hasattr(delta, 'content'):
                        delta_content = delta.content
                    elif isinstance(delta, dict) and 'content' in delta:
                        delta_content = delta['content']
                    
                    # Accumulate text content
                    if delta_content is not None and delta_content != "":
                        accumulated_text += delta_content
                        
                        # Always emit text deltas if no tool calls started
                        if tool_index is None and not text_block_closed:
                            text_sent = True
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta_content}})}\n\n"
                    
                    # Process tool calls
                    delta_tool_calls = None
                    
                    # Handle different formats of tool calls
                    if hasattr(delta, 'tool_calls'):
                        delta_tool_calls = delta.tool_calls
                    elif isinstance(delta, dict) and 'tool_calls' in delta:
                        delta_tool_calls = delta['tool_calls']
                    
                    # Process tool calls if any
                    if delta_tool_calls:
                        # First tool call we've seen - need to handle text properly
                        if tool_index is None:
                            # If we've been streaming text, close that text block
                            if text_sent and not text_block_closed:
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            # If we've accumulated text but not sent it, we need to emit it now
                            # This handles the case where the first delta has both text and a tool call
                            elif accumulated_text and not text_sent and not text_block_closed:
                                # Send the accumulated text
                                text_sent = True
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}})}\n\n"
                                # Close the text block
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            # Close text block even if we haven't sent anything - models sometimes emit empty text blocks
                            elif not text_block_closed:
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                                
                        # Convert to list if it's not already
                        if not isinstance(delta_tool_calls, list):
                            delta_tool_calls = [delta_tool_calls]
                        
                        for tool_call in delta_tool_calls:
                            # Get the index of this tool call (for multiple tools)
                            current_index = None
                            if isinstance(tool_call, dict) and 'index' in tool_call:
                                current_index = tool_call['index']
                            elif hasattr(tool_call, 'index'):
                                current_index = tool_call.index
                            else:
                                current_index = 0
                            
                            # Check if this is a new tool or a continuation
                            if tool_index is None or current_index != tool_index:
                                # New tool call - create a new tool_use block
                                tool_index = current_index
                                last_tool_index += 1
                                anthropic_tool_index = last_tool_index
                                
                                # Extract function info
                                if isinstance(tool_call, dict):
                                    function = tool_call.get('function', {})
                                    name = function.get('name', '') if isinstance(function, dict) else ""
                                    tool_id = tool_call.get('id', f"toolu_{uuid.uuid4().hex[:24]}")
                                else:
                                    function = getattr(tool_call, 'function', None)
                                    name = getattr(function, 'name', '') if function else ''
                                    tool_id = getattr(tool_call, 'id', f"toolu_{uuid.uuid4().hex[:24]}")
                                
                                # Start a new tool_use block
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': anthropic_tool_index, 'content_block': {'type': 'tool_use', 'id': tool_id, 'name': name, 'input': {}}})}\n\n"
                                current_tool_call = tool_call
                                tool_content = ""
                            
                            # Extract function arguments
                            arguments = None
                            if isinstance(tool_call, dict) and 'function' in tool_call:
                                function = tool_call.get('function', {})
                                arguments = function.get('arguments', '') if isinstance(function, dict) else ''
                            elif hasattr(tool_call, 'function'):
                                function = getattr(tool_call, 'function', None)
                                arguments = getattr(function, 'arguments', '') if function else ''
                            
                            # If we have arguments, send them as a delta
                            if arguments:
                                # Try to detect if arguments are valid JSON or just a fragment
                                try:
                                    # If it's already a dict, use it
                                    if isinstance(arguments, dict):
                                        args_json = json.dumps(arguments)
                                    else:
                                        # Otherwise, try to parse it
                                        json.loads(arguments)
                                        args_json = arguments
                                except (json.JSONDecodeError, TypeError):
                                    # If it's a fragment, treat it as a string
                                    args_json = arguments
                                
                                # Add to accumulated tool content
                                tool_content += args_json if isinstance(args_json, str) else ""
                                
                                # Send the update
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': anthropic_tool_index, 'delta': {'type': 'input_json_delta', 'partial_json': args_json}})}\n\n"
                    
                    # Process finish_reason - end the streaming response
                    if finish_reason and not has_sent_stop_reason:
                        has_sent_stop_reason = True
                        
                        # Close any open tool call blocks
                        if tool_index is not None:
                            for i in range(1, last_tool_index + 1):
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"
                        
                        # If we accumulated text but never sent or closed text block, do it now
                        if not text_block_closed:
                            if accumulated_text and not text_sent:
                                # Send the accumulated text
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}})}\n\n"
                            # Close the text block
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                        
                        # Map OpenAI finish_reason to Anthropic stop_reason
                        stop_reason = "end_turn"
                        if finish_reason == "length":
                            stop_reason = "max_tokens"
                        elif finish_reason == "tool_calls":
                            stop_reason = "tool_use"
                        elif finish_reason == "stop":
                            stop_reason = "end_turn"
                        
                        # Send message_delta with stop reason and usage
                        usage = {"input_tokens": input_tokens, "output_tokens": output_tokens, "cost_usd": compute_cost(original_request.model, input_tokens, output_tokens)}
                        
                        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': usage})}\n\n"
                        
                        # Send message_stop event
                        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
                        
                        # Send final [DONE] marker to match Anthropic's behavior
                        yield "data: [DONE]\n\n"
                        return
            except Exception as e:
                # Log error but continue processing other chunks
                logger.error(f"Error processing chunk: {str(e)}")
                continue
        
        # If we didn't get a finish reason, close any open blocks
        if not has_sent_stop_reason:
            # Close any open tool call blocks
            if tool_index is not None:
                for i in range(1, last_tool_index + 1):
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"
            
            # Close the text content block
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
            
            # Send final message_delta with usage
            usage = {"input_tokens": input_tokens, "output_tokens": output_tokens, "cost_usd": compute_cost(original_request.model, input_tokens, output_tokens)}
            
            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': usage})}\n\n"
            
            # Send message_stop event
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
            
            # Send final [DONE] marker to match Anthropic's behavior
            yield "data: [DONE]\n\n"
    
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_message = f"Error in streaming: {str(e)}\n\nFull traceback:\n{error_traceback}"
        logger.error(error_message)
        
        # Send error message_delta
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'error', 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
        
        # Send message_stop event
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        
        # Send final [DONE] marker
        yield "data: [DONE]\n\n"

@app.post("/v1/messages")
async def create_message(
    request: MessagesRequest,
    raw_request: Request
):
    try:
        # print the body here
        body = await raw_request.body()
    
        # Parse the raw body as JSON since it's bytes
        body_json = json.loads(body.decode('utf-8'))
        original_model = body_json.get("model", "unknown")
        
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
        
        logger.debug(f"📊 PROCESSING REQUEST: Model={request.model}, Stream={request.stream}")
        
        # Detect thinker mode and patch model if needed
        thinking_budget = None
        if hasattr(request, 'thinking') and request.thinking:
            thinking_budget = getattr(request.thinking, 'budget_tokens', None)
        reasoning_level = None
        if thinking_budget:
            reasoning_level = "high" if thinking_budget > 4096 else "medium"
            patched_model = THINKER_MODEL
            logger.info(f"THINKER mode triggered: routing to THINKER_MODEL ('{THINKER_MODEL}') with thinking budget {thinking_budget}")
            request.model = THINKER_MODEL
            # Optionally add a tag for OpenAI reasoning models
            if 'openai/o3' in THINKER_MODEL or 'openai/o4' in THINKER_MODEL or 'gpt-4o' in THINKER_MODEL:
                # Add reasoning tag if backend supports it
                if not hasattr(request, 'metadata') or request.metadata is None:
                    request.metadata = {}
                request.metadata["reasoning"] = True
                request.metadata["thinking_budget"] = thinking_budget
        # Check if we need direct passthrough to Anthropic API
        is_direct_anthropic = (BIG_MODEL == "passthrough" and 
                             (request.model.startswith("anthropic/") or "claude" in request.model.lower()))
        
        # Handle direct passthrough to Anthropic API (bypassing LiteLLM)
        if is_direct_anthropic:
            logger.info(f"📌 TRUE PASSTHROUGH: sending request directly to Anthropic API for {request.model}")
            
            # Ensure model has anthropic/ prefix if needed
            model_name = request.model
            if not model_name.startswith("anthropic/") and "claude" in model_name.lower():
                model_name = f"anthropic/{model_name}"
                logger.debug(f"Added anthropic/ prefix to model: {request.model} -> {model_name}")
            
            # For streaming requests, use direct Anthropic API streaming
            if request.stream:
                try:
                    # Create direct Anthropic request
                    api_url = "https://api.anthropic.com/v1/messages"
                    
                    # Dump the original request to JSON (preserving Anthropic format)
                    # Use dictionary to ensure we don't modify the original request's model
                    request_dict = request.dict(exclude_none=True, exclude_unset=True)
                    request_dict["model"] = model_name.replace("anthropic/", "")  # Remove prefix for API
                    
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
                    
                    headers = {
                        "x-api-key": ANTHROPIC_API_KEY,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    }
                    
                    # Log the beautiful request
                    num_tools = len(request.tools) if request.tools else 0
                    reasoning_level = "high" if thinking_budget and thinking_budget > 4096 else ("medium" if thinking_budget else None)
                    log_request_beautifully(
                        "POST", 
                        raw_request.url.path, 
                        request.model, 
                        "DIRECT to Anthropic",
                        len(request.messages),
                        num_tools,
                        200,
                        reasoning_level
                    )
                    
                    # Use httpx for async streaming
                    async def direct_anthropic_stream():
                        async with httpx.AsyncClient() as client:
                            async with client.stream("POST", api_url, json=request_dict, headers=headers) as response:
                                # Forward status code if there's an error
                                if response.status_code != 200:
                                    error_body = await response.aread()
                                    raise HTTPException(
                                        status_code=response.status_code,
                                        detail=f"Anthropic API error: {error_body.decode()}"
                                    )
                                
                                # Stream the response directly
                                async for chunk in response.aiter_bytes():
                                    yield chunk
                    
                    # Return the streaming response directly
                    return StreamingResponse(
                        direct_anthropic_stream(),
                        media_type="text/event-stream"
                    )
                
                except Exception as e:
                    logger.error(f"Error in direct Anthropic streaming: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Error in direct Anthropic API call: {str(e)}")
                
            else:
                # For non-streaming requests, use direct Anthropic API
                try:
                    # Create direct Anthropic request
                    api_url = "https://api.anthropic.com/v1/messages"
                    
                    # Dump the original request to JSON (preserving Anthropic format)
                    request_dict = request.dict(exclude_none=True, exclude_unset=True)
                    request_dict["model"] = model_name.replace("anthropic/", "")  # Remove prefix for API
                    
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
                    
                    headers = {
                        "x-api-key": ANTHROPIC_API_KEY,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    }
                    
                    # Log the beautiful request
                    num_tools = len(request.tools) if request.tools else 0
                    reasoning_level = "high" if thinking_budget and thinking_budget > 4096 else ("medium" if thinking_budget else None)
                    log_request_beautifully(
                        "POST", 
                        raw_request.url.path, 
                        request.model, 
                        "DIRECT to Anthropic",
                        len(request.messages),
                        num_tools,
                        200,
                        reasoning_level
                    )
                    
                    # Use httpx for async request
                    async with httpx.AsyncClient() as client:
                        response = await client.post(api_url, json=request_dict, headers=headers)
                        
                        # Check for errors
                        if response.status_code != 200:
                            raise HTTPException(
                                status_code=response.status_code,
                                detail=f"Anthropic API error: {response.text}"
                            )
                        
                        # Return the response directly
                        return response.json()
                
                except Exception as e:
                    logger.error(f"Error in direct Anthropic API call: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Error in direct Anthropic API call: {str(e)}")
        
        # Standard non-passthrough flow using LiteLLM
        # Convert Anthropic request to LiteLLM format
        litellm_request = convert_anthropic_to_litellm(request)
        
        # Determine which API key to use based on the model prefix
        if request.model.startswith("openai/"):
            litellm_request["api_key"] = OPENAI_API_KEY
            logger.debug(f"Using OpenAI API key for model: {request.model}")
        elif request.model.startswith("gemini/"):
            litellm_request["api_key"] = GEMINI_API_KEY
            logger.debug(f"Using Gemini API key for model: {request.model}")
        elif request.model.startswith("anthropic/") or "claude" in request.model.lower():
            litellm_request["api_key"] = ANTHROPIC_API_KEY
            logger.debug(f"Using Anthropic API key for model: {request.model}")
        else:
            # Default to OpenAI API key for models without a provider prefix
            litellm_request["api_key"] = OPENAI_API_KEY
            logger.debug(f"Using OpenAI API key (default) for model: {request.model}")
        
        # For OpenAI models - modify request format to work with limitations
        if "openai" in litellm_request["model"] and "messages" in litellm_request:
            logger.debug(f"Processing OpenAI model request: {litellm_request['model']}")
            
            # For OpenAI models, we need to convert content blocks to simple strings
            # and handle other requirements
            for i, msg in enumerate(litellm_request["messages"]):
                # Special case - handle message content directly when it's a list of tool_result
                # This is a specific case we're seeing in the error
                if "content" in msg and isinstance(msg["content"], list):
                    is_only_tool_result = True
                    for block in msg["content"]:
                        if not isinstance(block, dict) or block.get("type") != "tool_result":
                            is_only_tool_result = False
                            break
                    
                    if is_only_tool_result and len(msg["content"]) > 0:
                        logger.warning(f"Found message with only tool_result content - special handling required")
                        # Extract the content from all tool_result blocks
                        all_text = ""
                        for block in msg["content"]:
                            all_text += "Tool Result:\n"
                            result_content = block.get("content", [])
                            
                            # Handle different formats of content
                            if isinstance(result_content, list):
                                for item in result_content:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        all_text += item.get("text", "") + "\n"
                                    elif isinstance(item, dict):
                                        # Fall back to string representation of any dict
                                        try:
                                            item_text = item.get("text", json.dumps(item))
                                            all_text += item_text + "\n"
                                        except:
                                            all_text += str(item) + "\n"
                            elif isinstance(result_content, str):
                                all_text += result_content + "\n"
                            else:
                                try:
                                    all_text += json.dumps(result_content) + "\n"
                                except:
                                    all_text += str(result_content) + "\n"
                        
                        # Replace the list with extracted text
                        litellm_request["messages"][i]["content"] = all_text.strip() or "..."
                        logger.warning(f"Converted tool_result to plain text: {all_text.strip()[:200]}...")
                        continue  # Skip normal processing for this message
                
                # 1. Handle content field - normal case
                if "content" in msg:
                    # Check if content is a list (content blocks)
                    if isinstance(msg["content"], list):
                        # Convert complex content blocks to simple string
                        text_content = ""
                        for block in msg["content"]:
                            if isinstance(block, dict):
                                # Handle different content block types
                                if block.get("type") == "text":
                                    text_content += block.get("text", "") + "\n"
                                
                                # Handle tool_result content blocks - extract nested text
                                elif block.get("type") == "tool_result":
                                    tool_id = block.get("tool_use_id", "unknown")
                                    text_content += f"[Tool Result ID: {tool_id}]\n"
                                    
                                    # Extract text from the tool_result content
                                    result_content = block.get("content", [])
                                    if isinstance(result_content, list):
                                        for item in result_content:
                                            if isinstance(item, dict) and item.get("type") == "text":
                                                text_content += item.get("text", "") + "\n"
                                            elif isinstance(item, dict):
                                                # Handle any dict by trying to extract text or convert to JSON
                                                if "text" in item:
                                                    text_content += item.get("text", "") + "\n"
                                                else:
                                                    try:
                                                        text_content += json.dumps(item) + "\n"
                                                    except:
                                                        text_content += str(item) + "\n"
                                    elif isinstance(result_content, dict):
                                        # Handle dictionary content
                                        if result_content.get("type") == "text":
                                            text_content += result_content.get("text", "") + "\n"
                                        else:
                                            try:
                                                text_content += json.dumps(result_content) + "\n"
                                            except:
                                                text_content += str(result_content) + "\n"
                                    elif isinstance(result_content, str):
                                        text_content += result_content + "\n"
                                    else:
                                        try:
                                            text_content += json.dumps(result_content) + "\n"
                                        except:
                                            text_content += str(result_content) + "\n"
                                
                                # Handle tool_use content blocks
                                elif block.get("type") == "tool_use":
                                    tool_name = block.get("name", "unknown")
                                    tool_id = block.get("id", "unknown")
                                    tool_input = json.dumps(block.get("input", {}))
                                    text_content += f"[Tool: {tool_name} (ID: {tool_id})]\nInput: {tool_input}\n\n"
                                
                                # Handle image content blocks
                                elif block.get("type") == "image":
                                    text_content += "[Image content - not displayed in text format]\n"
                        
                        # Make sure content is never empty for OpenAI models
                        if not text_content.strip():
                            text_content = "..."
                        
                        litellm_request["messages"][i]["content"] = text_content.strip()
                    # Also check for None or empty string content
                    elif msg["content"] is None:
                        litellm_request["messages"][i]["content"] = "..." # Empty content not allowed
                
                # 2. Remove any fields OpenAI doesn't support in messages
                for key in list(msg.keys()):
                    if key not in ["role", "content", "name", "tool_call_id", "tool_calls"]:
                        logger.warning(f"Removing unsupported field from message: {key}")
                        del msg[key]
            
            # 3. Final validation - check for any remaining invalid values and dump full message details
            for i, msg in enumerate(litellm_request["messages"]):
                # Log the message format for debugging
                logger.debug(f"Message {i} format check - role: {msg.get('role')}, content type: {type(msg.get('content'))}")
                
                # If content is still a list or None, replace with placeholder
                if isinstance(msg.get("content"), list):
                    logger.warning(f"CRITICAL: Message {i} still has list content after processing: {json.dumps(msg.get('content'))}")
                    # Last resort - stringify the entire content as JSON
                    litellm_request["messages"][i]["content"] = f"Content as JSON: {json.dumps(msg.get('content'))}"
                elif msg.get("content") is None:
                    logger.warning(f"Message {i} has None content - replacing with placeholder")
                    litellm_request["messages"][i]["content"] = "..." # Fallback placeholder
        
        # Log request information based on logging settings
        if log_level == logging.DEBUG or SHOW_MODEL_DETAILS:
            # Create a summary of the request
            requested_model = request.model
            actual_model = litellm_request.get('model')
            is_streaming = litellm_request.get('stream', False)
            msg_count = len(litellm_request.get('messages', []))
            max_tokens_val = litellm_request.get('max_completion_tokens', litellm_request.get('max_tokens', 'unknown'))
            
            # Show a focused summary
            request_summary = f"📤 REQUEST: From='{requested_model}' To='{actual_model}', Stream={is_streaming}, MaxTokens={max_tokens_val}, Messages={msg_count}"
            
            if SHOW_MODEL_DETAILS:
                # If we're showing model details, log at INFO level
                logger.info(request_summary)
                
                # Optionally show a preview of the last message
                if msg_count > 0 and "messages" in litellm_request:
                    last_msg = litellm_request["messages"][-1]
                    role = last_msg.get("role", "unknown")
                    content = last_msg.get("content", "")
                    if isinstance(content, str):
                        preview = content[:100] + ("..." if len(content) > 100 else "")
                        logger.info(f"Last message ({role}): {preview}")
            
            # In full debug mode, log the detailed request
            if log_level == logging.DEBUG:
                # Create a safe copy of the request for logging with sensitive/large data truncated
                log_request = litellm_request.copy()
                
                # Only log a sample of the messages to avoid overwhelming logs
                if "messages" in log_request:
                    sample_messages = []
                    for i, msg in enumerate(log_request["messages"]):
                        # For each message, include role and a preview of content
                        sample_msg = {"role": msg.get("role", "unknown")}
                        
                        # Handle different content formats
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            # Show first 100 chars of string content
                            sample_msg["content"] = content[:100] + ("..." if len(content) > 100 else "")
                        elif isinstance(content, list):
                            # For content blocks, just show count and types
                            types = [block.get("type", "unknown") if isinstance(block, dict) else type(block).__name__ 
                                    for block in content[:5]]
                            sample_msg["content"] = f"[{len(content)} blocks: {', '.join(types)}...]"
                        
                        sample_messages.append(sample_msg)
                    
                    # Replace full messages with sample
                    log_request["messages"] = sample_messages
                
                # Log the request details
                logger.debug(f"DETAILED REQUEST: {json.dumps(log_request, indent=2)}")
        else:
            # Minimal info for normal mode
            logger.debug(f"Request for model: {litellm_request.get('model')}, stream: {litellm_request.get('stream', False)}")
        
        # Handle streaming mode
        if request.stream:
            # Use LiteLLM for streaming
            num_tools = len(request.tools) if request.tools else 0
            reasoning_level = "high" if thinking_budget and thinking_budget > 4096 else ("medium" if thinking_budget else None)
            
            log_request_beautifully(
                "POST", 
                raw_request.url.path, 
                display_model, 
                litellm_request.get('model'),
                len(litellm_request['messages']),
                num_tools,
                200,  # Assuming success at this point
                reasoning_level
            )
            # Ensure we use the async version for streaming
            response_generator = await litellm.acompletion(**litellm_request)
            
            return StreamingResponse(
                handle_streaming(response_generator, request),
                media_type="text/event-stream"
            )
        else:
            # Use LiteLLM for regular completion
            num_tools = len(request.tools) if request.tools else 0
            
            log_request_beautifully(
                "POST", 
                raw_request.url.path, 
                display_model, 
                litellm_request.get('model'),
                len(litellm_request['messages']),
                num_tools,
                200,  # Assuming success at this point
                reasoning_level
            )
            start_time = time.time()
            litellm_response = litellm.completion(**litellm_request)
            logger.debug(f"✅ RESPONSE RECEIVED: Model={litellm_request.get('model')}, Time={time.time() - start_time:.2f}s")
            
            # Convert LiteLLM response to Anthropic format
            anthropic_response = convert_litellm_to_anthropic(litellm_response, request)
            # Compute and attach cost
            in_toks = anthropic_response.usage.input_tokens
            out_toks = anthropic_response.usage.output_tokens
            anthropic_response.usage.cost_usd = compute_cost(request.model, in_toks, out_toks)
            
            # Log response info based on logging settings
            if log_level == logging.DEBUG or SHOW_MODEL_DETAILS:
                # Extract the actual model used from the response
                actual_model = None
                model_response_details = {}
                
                # Extract detailed model info - this helps identify exactly which model responded
                if hasattr(litellm_response, "__dict__"):
                    for field in ["model", "id", "object", "created", "system_fingerprint", "model_version"]:
                        if hasattr(litellm_response, field):
                            model_response_details[field] = getattr(litellm_response, field)
                
                # Also try dict access for non-object responses
                if isinstance(litellm_response, dict):
                    for field in ["model", "id", "object", "created", "system_fingerprint", "model_version"]:
                        if field in litellm_response and field not in model_response_details:
                            model_response_details[field] = litellm_response[field]
                
                # Log the full model details JSON
                if model_response_details:
                    logger.info(f"🔍 DETAILED MODEL INFO: {json.dumps(model_response_details)}")
                
                # Set actual_model for normal logging flow
                if "model" in model_response_details:
                    actual_model = model_response_details["model"]
                elif hasattr(litellm_response, "model"):
                    actual_model = getattr(litellm_response, "model")
                elif isinstance(litellm_response, dict) and "model" in litellm_response:
                    actual_model = litellm_response["model"]
                
                # Extract content length - first try to get from anthropic_response
                response_content = ""
                if hasattr(anthropic_response, "content") and anthropic_response.content:
                    for block in anthropic_response.content:
                        if hasattr(block, "type") and block.type == "text" and hasattr(block, "text"):
                            response_content += block.text
                
                # If nothing from anthropic_response, try litellm_response
                if not response_content and hasattr(litellm_response, "choices") and litellm_response.choices:
                    if hasattr(litellm_response.choices[0], "message") and hasattr(litellm_response.choices[0].message, "content"):
                        response_content = litellm_response.choices[0].message.content
                
                # Show just the key information at INFO level
                content_preview = response_content[:50] + "..." if response_content and len(response_content) > 50 else response_content
                response_summary = f"📥 RESPONSE: Requested='{request.model}', Actual='{actual_model}', Content={len(response_content)} chars"
                
                if SHOW_MODEL_DETAILS:
                    logger.info(f"{response_summary}\nPreview: {content_preview}")
                else:
                    logger.debug(f"{response_summary}\nPreview: {content_preview}")
                
                # In full debug mode, log the detailed response
                if log_level == logging.DEBUG:
                    # Create safe copy for logging with potential large fields truncated
                    log_response = {}
                    if hasattr(anthropic_response, "dict"):
                        log_response = anthropic_response.dict()
                    else:
                        # Try to extract key fields if not a Pydantic model
                        log_response = {
                            "id": getattr(anthropic_response, "id", None),
                            "model": getattr(anthropic_response, "model", None), 
                            "role": getattr(anthropic_response, "role", None),
                            "stop_reason": getattr(anthropic_response, "stop_reason", None),
                        }
                    
                    # Truncate content for logging
                    if "content" in log_response:
                        if isinstance(log_response["content"], list):
                            for i, block in enumerate(log_response["content"]):
                                if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                                    text = block["text"]
                                    if len(text) > 500:
                                        log_response["content"][i]["text"] = text[:500] + "... [truncated]"
                    
                    logger.debug(f"DETAILED RESPONSE: {json.dumps(log_response, indent=2)}")
            
            return anthropic_response
                
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        
        # Capture as much info as possible about the error
        error_details = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": error_traceback
        }
        
        # Check for LiteLLM-specific attributes
        for attr in ['message', 'status_code', 'response', 'llm_provider', 'model']:
            if hasattr(e, attr):
                error_details[attr] = getattr(e, attr)
        
        # Check for additional exception details in dictionaries
        if hasattr(e, '__dict__'):
            for key, value in e.__dict__.items():
                if key not in error_details and key not in ['args', '__traceback__']:
                    error_details[key] = str(value)
        
        # Make error details JSON serializable by converting problematic objects to strings
        for key, value in list(error_details.items()):
            try:
                # Test if value is JSON serializable
                json.dumps({key: value})
            except (TypeError, OverflowError):
                # If not serializable, convert to string
                error_details[key] = f"[Non-serializable {type(value).__name__}]: {str(value)}"
        
        # Log all error details
        logger.error(f"Error processing request: {json.dumps(error_details, indent=2)}")
        
        # Format error for response
        error_message = f"Error: {str(e)}"
        if 'message' in error_details and error_details['message']:
            error_message += f"\nMessage: {error_details['message']}"
        if 'response' in error_details and error_details['response']:
            error_message += f"\nResponse: {error_details['response']}"
        
        # Return detailed error
        status_code = error_details.get('status_code', 500)
        raise HTTPException(status_code=status_code, detail=error_message)

@app.post("/v1/messages/count_tokens")
async def count_tokens(
    request: TokenCountRequest,
    raw_request: Request
):
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
                request_dict["model"] = model_name.replace("anthropic/", "")  # Remove prefix for API
                
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
                
                headers = {
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                }
                
                # Use httpx for async request
                async with httpx.AsyncClient() as client:
                    response = await client.post(api_url, json=request_dict, headers=headers)
                    
                    # Check for errors
                    if response.status_code != 200:
                        raise HTTPException(
                            status_code=response.status_code,
                            detail=f"Anthropic token counting error: {response.text}"
                        )
                    
                    # Return the token count directly
                    token_count_data = response.json()
                    return TokenCountResponse(input_tokens=token_count_data.get("input_tokens", 0))
            
            except Exception as e:
                logger.error(f"Error in direct Anthropic token counting: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error in direct Anthropic token counting: {str(e)}")
        
        # Standard non-passthrough flow using LiteLLM
        converted_request = convert_anthropic_to_litellm(
            MessagesRequest(
                model=request.model,
                max_tokens=100,  # Arbitrary value not used for token counting
                messages=request.messages,
                system=request.system,
                tools=request.tools,
                tool_choice=request.tool_choice,
                thinking=request.thinking
            )
        )
        
        # Use LiteLLM's token_counter function
        try:
            # Import token_counter function
            from litellm import token_counter
            
            # Log the request beautifully
            num_tools = len(request.tools) if request.tools else 0
            
            # Check for thinking budget in the request
            thinking_budget = None
            if hasattr(request, 'thinking') and request.thinking:
                thinking_budget = getattr(request.thinking, 'budget_tokens', None)
            reasoning_level = "high" if thinking_budget and thinking_budget > 4096 else ("medium" if thinking_budget else None)
            
            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                converted_request.get('model'),
                len(converted_request['messages']),
                num_tools,
                200,  # Assuming success at this point
                reasoning_level
            )
            
            # Add the appropriate API key based on the model prefix
            if request.model.startswith("openai/"):
                api_key = OPENAI_API_KEY
                logger.debug(f"Using OpenAI API key for token counting: {request.model}")
            elif request.model.startswith("gemini/"):
                api_key = GEMINI_API_KEY
                logger.debug(f"Using Gemini API key for token counting: {request.model}")
            elif request.model.startswith("anthropic/") or "claude" in request.model.lower():
                api_key = ANTHROPIC_API_KEY
                logger.debug(f"Using Anthropic API key for token counting: {request.model}")
            else:
                # Default to OpenAI API key for models without a provider prefix
                api_key = OPENAI_API_KEY
                logger.debug(f"Using OpenAI API key (default) for token counting: {request.model}")
            
            # Count tokens
            token_count = token_counter(
                model=converted_request["model"],
                messages=converted_request["messages"]
            )
            
            # Return Anthropic-style response
            return TokenCountResponse(input_tokens=token_count)
            
        except ImportError:
            logger.error("Could not import token_counter from litellm")
            # Fallback to a simple approximation
            return TokenCountResponse(input_tokens=1000)  # Default fallback
            
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error counting tokens: {str(e)}\n{error_traceback}")
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Anthropic Proxy for LiteLLM"}

# Define ANSI color codes for terminal output
class Colors:
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    DIM = "\033[2m"
def log_request_beautifully(method, path, claude_model, openai_model, num_messages, num_tools, status_code, reasoning_level=None):
    """Log requests in a beautiful, twitter-friendly format showing Claude to OpenAI mapping."""
    # Format the Claude model name nicely
    claude_display = f"{Colors.CYAN}{claude_model}{Colors.RESET}"
    
    # Extract endpoint name
    endpoint = path
    if "?" in endpoint:
        endpoint = endpoint.split("?")[0]
    
    # Extract just the OpenAI model name without provider prefix
    openai_display = openai_model
    if "/" in openai_display:
        openai_display = openai_display.split("/")[-1]
    openai_display = f"{Colors.GREEN}{openai_display}{Colors.RESET}"
    
    # Append reasoning tier if provided
    if reasoning_level:
        # Colorize the reasoning level for better visibility
        level_color = Colors.YELLOW if reasoning_level == "medium" else Colors.RED
        reasoning_display = f" ({level_color}{reasoning_level}{Colors.RESET})"
        claude_display += reasoning_display
        openai_display += reasoning_display
    
    # Format tools and messages
    tools_str = f"{Colors.MAGENTA}{num_tools} tools{Colors.RESET}"
    messages_str = f"{Colors.BLUE}{num_messages} messages{Colors.RESET}"
    
    # Format status code
    status_str = f"{Colors.GREEN}✓ {status_code} OK{Colors.RESET}" if status_code == 200 else f"{Colors.RED}✗ {status_code}{Colors.RESET}"
    
    # Put it all together in a clear, beautiful format
    log_line = f"{Colors.BOLD}{method} {endpoint}{Colors.RESET} {status_str}"
    model_line = f"{claude_display} → {openai_display} {tools_str} {messages_str}"
    
    # Print to console
    print(log_line)
    print(model_line)
    sys.stdout.flush()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Run with: uvicorn server:app --reload --host 0.0.0.0 --port 8082")
        sys.exit(0)
    
    # Configure uvicorn to run with minimal logs
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="error")