#!/bin/bash

# Set up logging directory and file
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/server_$(date +'%Y-%m-%d_%H-%M-%S').log"

# Force colors to be preserved even when output is piped
export CLICOLOR_FORCE=1
#export PYTHONIOENCODING=utf-8

# Enable verbose mode to see model and request info
#export LOGLEVEL=INFO
#export LOGLEVEL=DEBUG
export SHOW_MODEL_DETAILS=true

# Set default model configuration
export THINKER_MODEL="${BIG_MODEL:-openai/o3}"
export BIG_MODEL="${BIG_MODEL:-openai/o4-mini}"
export SMALL_MODEL="${SMALL_MODEL:-openai/gpt-4o}"

# Print server startup info
echo "Starting server with BIG_MODEL=$BIG_MODEL, SMALL_MODEL=$SMALL_MODEL"
echo "Logging to $LOG_FILE"

# Run server and log output
uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload #2>&1 | tee -a "$LOG_FILE"
