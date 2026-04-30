"""FastAPI application factory for the AFM Similarity Search API.

Run locally::

    uvicorn api.main:app --reload --port 8000

The ``/health`` endpoint is always available without authentication.
All other endpoints require an ``X-API-Key`` header matching ``settings.API_KEY``.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.core.config import settings
from api.core.deps import verify_api_key
from api.routers import ingest, search

app = FastAPI(
    title="AFM Similarity Search",
    description=(
        "Multimodal similarity search over AFM .ibw files using "
        "BiomedCLIP embeddings and pgvector."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    ingest.router,
    prefix="/ingest",
    tags=["ingestion"],
    dependencies=[Depends(verify_api_key)],
)
app.include_router(
    search.router,
    tags=["search"],
    dependencies=[Depends(verify_api_key)],
)


@app.get("/health", tags=["health"])
async def health() -> dict:
    """Liveness probe — returns 200 with service info."""
    return {
        "status": "ok",
        "model_name": settings.MODEL_NAME,
        "model_version": settings.MODEL_VERSION,
    }
