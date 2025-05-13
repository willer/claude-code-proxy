"""
Pydantic models for the Claude Code Proxy API.
"""
from typing import Any, Dict, List, Literal, Optional, Union
import logging
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

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
    content: Union[str, List[Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult]], Any]

    @model_validator(mode='before')
    @classmethod
    def validate_content(cls, data):
        """Validate and normalize the content field to handle various formats."""
        if not isinstance(data, dict):
            return data

        content = data.get('content')

        # If content is already a string or properly structured list, return as is
        if isinstance(content, str) or content is None:
            return data

        # Handle malformed content that might be a list of improperly structured objects
        if isinstance(content, list):
            try:
                # Check if each item has required structure or convert to proper format
                normalized_content = []
                for item in content:
                    if isinstance(item, dict) and 'type' in item:
                        # Item seems properly structured, keep as is
                        normalized_content.append(item)
                    elif isinstance(item, dict) and 'text' in item:
                        # Convert to proper text block
                        normalized_content.append({"type": "text", "text": item["text"]})
                    elif isinstance(item, str):
                        # Convert string to text block
                        normalized_content.append({"type": "text", "text": item})
                    else:
                        # Try to keep item if it's dict-like
                        if isinstance(item, dict):
                            normalized_content.append(item)

                # Replace content with normalized version if we have items
                if normalized_content:
                    data['content'] = normalized_content

            except Exception as e:
                logger.warning(f"Error normalizing message content: {str(e)}")
                # Convert entire content to string if normalization fails
                if content:
                    data['content'] = str(content)

        return data

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

    @model_validator(mode='before')
    @classmethod
    def validate_messages(cls, data):
        """Preprocess the messages to ensure content is properly formatted."""
        if not isinstance(data, dict):
            return data

        # Normalize messages if present
        if 'messages' in data and isinstance(data['messages'], list):
            try:
                normalized_messages = []
                for msg in data['messages']:
                    if not isinstance(msg, dict):
                        continue

                    # Ensure message has role
                    if 'role' not in msg:
                        continue

                    # Handle content field specifically
                    if 'content' in msg:
                        content = msg['content']

                        # Convert string content to proper format
                        if isinstance(content, str):
                            # Keep as is, string is valid
                            pass
                        # Handle list content that may need normalization
                        elif isinstance(content, list):
                            normalized_content = []
                            for item in content:
                                if isinstance(item, dict) and 'type' in item:
                                    # Item seems properly structured, keep as is
                                    normalized_content.append(item)
                                elif isinstance(item, dict) and 'text' in item:
                                    # Add type field if missing
                                    item_copy = dict(item)
                                    item_copy['type'] = 'text'
                                    normalized_content.append(item_copy)
                                elif isinstance(item, str):
                                    # Convert string to text block
                                    normalized_content.append({"type": "text", "text": item})
                                else:
                                    # For unknown format, try to make best guess
                                    if isinstance(item, dict):
                                        # If it has keys but not proper structure, add it anyway
                                        # The model validator for Message will handle further normalization
                                        normalized_content.append(item)

                            # Replace content with normalized version
                            if normalized_content:
                                msg['content'] = normalized_content
                            else:
                                # If normalization results in empty list, convert to string
                                msg['content'] = str(content)

                        # Handle dict content (which is not an expected format by default)
                        elif isinstance(content, dict):
                            # Convert single dict to a list with one item
                            if 'type' in content:
                                msg['content'] = [content]
                            elif 'text' in content:
                                msg['content'] = [{"type": "text", "text": content['text']}]
                            else:
                                # Unknown dict format, convert to string
                                msg['content'] = str(content)

                    normalized_messages.append(msg)

                # Replace messages with normalized version
                if normalized_messages:
                    data['messages'] = normalized_messages

            except Exception as e:
                logger.warning(f"Error preprocessing messages in request: {str(e)}")
                # Keep original messages if preprocessing fails

        return data

class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None  # Will store the original model name

    @model_validator(mode='before')
    @classmethod
    def validate_messages(cls, data):
        """Reuse the same validation logic as MessagesRequest."""
        if not isinstance(data, dict):
            return data

        # Apply the same message normalization as in MessagesRequest
        if 'messages' in data and isinstance(data['messages'], list):
            try:
                normalized_messages = []
                for msg in data['messages']:
                    if not isinstance(msg, dict):
                        continue

                    # Ensure message has role
                    if 'role' not in msg:
                        continue

                    # Handle content field specifically
                    if 'content' in msg:
                        content = msg['content']

                        # Convert string content to proper format
                        if isinstance(content, str):
                            # Keep as is, string is valid
                            pass
                        # Handle list content that may need normalization
                        elif isinstance(content, list):
                            normalized_content = []
                            for item in content:
                                if isinstance(item, dict) and 'type' in item:
                                    # Item seems properly structured, keep as is
                                    normalized_content.append(item)
                                elif isinstance(item, dict) and 'text' in item:
                                    # Add type field if missing
                                    item_copy = dict(item)
                                    item_copy['type'] = 'text'
                                    normalized_content.append(item_copy)
                                elif isinstance(item, str):
                                    # Convert string to text block
                                    normalized_content.append({"type": "text", "text": item})
                                else:
                                    # For unknown format, try to make best guess
                                    if isinstance(item, dict):
                                        # If it has keys but not proper structure, add it anyway
                                        normalized_content.append(item)

                            # Replace content with normalized version
                            if normalized_content:
                                msg['content'] = normalized_content
                            else:
                                # If normalization results in empty list, convert to string
                                msg['content'] = str(content)

                        # Handle dict content (which is not an expected format by default)
                        elif isinstance(content, dict):
                            # Convert single dict to a list with one item
                            if 'type' in content:
                                msg['content'] = [content]
                            elif 'text' in content:
                                msg['content'] = [{"type": "text", "text": content['text']}]
                            else:
                                # Unknown dict format, convert to string
                                msg['content'] = str(content)

                    normalized_messages.append(msg)

                # Replace messages with normalized version
                if normalized_messages:
                    data['messages'] = normalized_messages

            except Exception as e:
                logger.warning(f"Error preprocessing messages in token count request: {str(e)}")
                # Keep original messages if preprocessing fails

        return data

class TokenCountResponse(BaseModel):
    input_tokens: int

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int

class MessagesResponse(BaseModel):
    id: str
    model: str
    type: str = "message"
    role: str
    content: List[Dict[str, Any]]
    usage: Usage
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None