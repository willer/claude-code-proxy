# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Run Commands
- Run server: `uv run uvicorn app:app --host 0.0.0.0 --port 8082 --reload`
- Run server using script: `./run-server.sh`
- Run tests: `python tests.py`
- Run specific tests: `python tests.py --simple` or `python tests.py --tools-only`
- Run streaming/non-streaming tests: `python tests.py --streaming-only` or `python tests.py --no-streaming`

## Style Guidelines
- Use FastAPI and Pydantic for API interactions
- Follow PEP 8 conventions with appropriate typing annotations
- Import order: standard library → third-party → local modules
- Use descriptive variable names and docstrings for functions
- Follow error handling patterns: try/except with detailed logging
- Log errors with appropriate context using logging module
- Maintain model mapping architecture between Anthropic/OpenAI/Gemini
- For model logging, use (medium) and (high) indicators when thinking/reasoning budgets are active

## Project Structure
- app/: Main Python package
  - __init__.py: FastAPI application initialization
  - middleware.py: Request/response middleware
  - error_handlers.py: Exception handlers
  - models/: Pydantic data models
  - routes/: API endpoint handlers
    - messages.py: Main messages endpoint
    - token_count.py: Token counting endpoint
  - utils/: Utility functions
    - auth.py: Authentication utilities
    - helpers.py: Helper functions
    - logging.py: Logging configuration
    - model_conversion.py: Model format conversion utilities
  - tests/: Test fixtures and utilities
- app.py: Application entry point
- tests.py: Test suite for API functionality
- pyproject.toml: Project dependencies and metadata
- run-server.sh: Convenience script to start the server