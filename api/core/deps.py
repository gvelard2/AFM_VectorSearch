"""FastAPI dependency injection for shared services.

Import ``get_encoder`` and ``get_vector_store`` as FastAPI ``Depends`` targets.
Both return cached singletons so models and DB connections are initialised once
at startup rather than per-request.
"""

from __future__ import annotations

import functools
from typing import Annotated

from fastapi import Depends

from api.core.config import settings
from services.encoder import CLIPEncoder, get_encoder as _get_encoder
from services.vector_store import VectorStore


def get_encoder() -> CLIPEncoder:
    """Return the process-level CLIPEncoder singleton."""
    return _get_encoder()


@functools.lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """Return a VectorStore connected to the configured PostgreSQL/pgvector DB.

    The instance is cached for the lifetime of the process. On first call it
    connects to the DB and creates the table + HNSW index if they don't exist.
    """
    return VectorStore(settings.DB_URL)


EncoderDep = Annotated[CLIPEncoder, Depends(get_encoder)]
VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]
