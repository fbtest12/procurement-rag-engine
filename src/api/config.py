"""
Application configuration via environment variables.

Uses pydantic-settings for type-safe config with sensible defaults.
All sensitive values (API keys) are loaded from environment variables
and never hardcoded.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # LLM Configuration
    llm_provider: str = "openrouter"
    llm_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    openrouter_api_key: str = ""

    # Vector Store Configuration
    collection_name: str = "procurement_docs"
    chroma_persist_dir: str = "./data/chromadb"

    # Chunking Configuration
    chunking_strategy: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval Configuration
    default_top_k: int = 5
    default_score_threshold: float = 0.3

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
