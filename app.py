"""
Main entry point for the Claude Code Proxy application.
This script is used to run the FastAPI application using Uvicorn.
"""
import os
import uvicorn
from app import app

if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8082))
    
    # Run the application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )