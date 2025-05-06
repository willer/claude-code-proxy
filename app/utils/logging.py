"""
Logging configuration for Claude Code Proxy.
"""
import logging
import os
from datetime import datetime
from typing import Optional

# Configure logging
log_level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_name, logging.INFO)


class MessageFilter(logging.Filter):
    """Filter to prevent sensitive data from being logged."""
    
    def filter(self, record):
        message = record.getMessage()
        # Add any filtering logic here
        return True


class ColorizedFormatter(logging.Formatter):
    """Formatter that adds color to logs based on level."""
    
    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',   # Green
        'WARNING': '\033[33m', # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[41m\033[37m',  # White on Red background
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_message = super().format(record)
        color_code = self.COLORS.get(record.levelname, '')
        return f"{color_code}{log_message}{self.RESET}"


def setup_logging():
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Set up logging to file
    log_file = f"logs/claude-code-proxy-{datetime.now().strftime('%Y-%m-%d')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Set up console logging with color
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = ColorizedFormatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add message filter to both handlers
    message_filter = MessageFilter()
    file_handler.addFilter(message_filter)
    console_handler.addFilter(message_filter)
    
    # Set up root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def log_request_beautifully(method, path, claude_model, openai_model, num_messages, num_tools, status_code, reasoning_level=None):
    """Format and log API requests in a readable way."""
    logger = logging.getLogger()
    
    model_display = claude_model or openai_model or "unknown"
    tools_str = f" with {num_tools} tools" if num_tools else ""
    reasoning_str = f" ({reasoning_level})" if reasoning_level else ""
    
    # Create a colorful, formatted log message
    log_message = (
        f"\n{'='*80}\n"
        f"📡 {method} {path} | "
        f"📦 model: {model_display}{reasoning_str} | "
        f"💬 msgs: {num_messages}{tools_str} | "
        f"🔢 status: {status_code}\n"
        f"{'='*80}"
    )
    
    logger.info(log_message)