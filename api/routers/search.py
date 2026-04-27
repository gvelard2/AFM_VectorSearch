"""Search endpoints — POST /search and GET /sample/{id}."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from api.core.config import settings
from api.core.deps import EncoderDep, VectorStoreDep
from api.models.schemas import SearchRequest, SearchResponse, SearchHit

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search_scans(
    encoder: EncoderDep,
    store: VectorStoreDep,
    request: SearchRequest,
    file: UploadFile | None = File(None, description="Optional .ibw query image"),
) -> SearchResponse:
    """Find the top-k most similar AFM scans.

    At least one of *file* (image query) or ``request.text`` must be provided.
    When both are supplied the embeddings are fused using ``request.image_weight``.

    Raises:
        HTTPException 400: If neither file nor text is provided.
        HTTPException 500: On search errors.
    """
    raise NotImplementedError(
        "search_scans: validate that file or request.text is present, "
        "build query embedding (embed_image and/or embed_text, then fuse), "
        "call store.search(vector, top_k=request.top_k, filters=request.filters), "
        "and map results to SearchHit instances."
    )


@router.get("/sample/{sample_id}")
async def get_sample(
    sample_id: str,
    store: VectorStoreDep,
) -> JSONResponse:
    """Retrieve metadata for a single ingested sample by ID.

    Raises:
        HTTPException 404: If no sample with *sample_id* exists.
    """
    raise NotImplementedError(
        "get_sample: query the DB for the row with this sample_id and return it, "
        "or raise HTTPException(404) if not found."
    )
