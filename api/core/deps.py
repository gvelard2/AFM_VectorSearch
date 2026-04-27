"""FastAPI dependency injection for shared services.

Import ``get_encoder`` and ``get_vector_store`` as FastAPI ``Depends`` targets.
Both return cached singletons so models and DB connections are initialised once
at startup rather than per-request.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from services.encoder import CLIPEncoder
from services.vector_store import VectorStore


def get_encoder() -> CLIPEncoder:
    """Return the process-level CLIPEncoder singleton.

    The encoder is constructed lazily on first call and cached via
    ``functools.lru_cache`` inside ``services.encoder``.
    """
    return CLIPEncoder.instance()


def get_vector_store() -> VectorStore:
    """Return a VectorStore connected to the configured PostgreSQL/pgvector DB."""
    raise NotImplementedError(
        "get_vector_store: construct and return a VectorStore using settings.DB_URL. "
        "Consider using a module-level singleton or an async lifespan connection pool."
    )


EncoderDep = Annotated[CLIPEncoder, Depends(get_encoder)]
VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]
