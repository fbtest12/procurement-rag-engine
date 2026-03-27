"""
Entry point for the Procurement RAG Engine API server.

Usage:
    python main.py
    uvicorn src.api.app:create_app --factory --reload
"""

import logging
import uvicorn
from dotenv import load_dotenv

load_dotenv()

from src.api.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

if __name__ == "__main__":
    settings = get_settings()

    uvicorn.run(
        "src.api.app:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level=settings.log_level.lower(),
    )
