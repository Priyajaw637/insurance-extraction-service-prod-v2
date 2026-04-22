import os
import socket
import subprocess
import uuid
import uvicorn
# from dotenv import load_dotenv
from app.config import ENV_PROJECT

# Load environment variables from .env file
# load_dotenv()

if __name__ == "__main__":
    # unique_worker_name = f"worker_{uuid.uuid4().hex[:8]}@{socket.gethostname()}"
    # celery_cmd = [
    #     "celery",
    #     "-A", "app.modules.Celery.config.celery",  # Point directly to the Celery app instance
    #     "worker",
    #     "--loglevel=INFO",
    #     "-n", unique_worker_name
    # ]
    # celery_process = subprocess.Popen(celery_cmd)

    # Get configuration from environment
    host = ENV_PROJECT.HOST
    port = ENV_PROJECT.PORT
    reload = ENV_PROJECT.RELOAD

    # Start Uvicorn server for FastAPI
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"  # Added proper logging
    )
