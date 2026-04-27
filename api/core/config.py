"""Application settings loaded from environment variables.

Uses pydantic-settings so every value can be overridden via a .env file or
shell environment. Example .env::

    DB_URL=postgresql+asyncpg://user:pass@localhost:5432/afm
    MODEL_NAME=microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
    MODEL_VERSION=1.0.0
    API_KEY=changeme
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Database
    DB_URL: str = "postgresql+asyncpg://afm:afm@localhost:5432/afm"

    # Embedding model
    MODEL_NAME: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    MODEL_VERSION: str = "1.0.0"

    # Search defaults
    IMAGE_WEIGHT: float = 0.6  # text weight = 1 - IMAGE_WEIGHT

    # API security
    API_KEY: str = "changeme"

    # Vector store
    HNSW_EF_SEARCH: int = 64
    TOP_K_DEFAULT: int = 5


settings = Settings()
