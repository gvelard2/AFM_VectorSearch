"""Search endpoints — POST /search and GET /sample/{id}."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from api.core.deps import EncoderDep, VectorStoreDep
from api.models.schemas import AFMMetadata, SearchHit, SearchResponse
from ingestion.parsers.ibw import parse_ibw
from ingestion.preprocessing import preprocess

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search_scans(
    encoder: EncoderDep,
    store: VectorStoreDep,
    text: str | None = Form(None, description="Free-text query description"),
    top_k: int = Form(5, ge=1, le=100),
    image_weight: float = Form(0.6, ge=0.0, le=1.0),
    filters: str | None = Form(None, description="JSON-encoded filter dict, e.g. '{\"material\": \"SrTiO3\"}'"),
    file: UploadFile | None = File(None, description="Optional .ibw query image"),
) -> SearchResponse:
    """Find the top-k most similar AFM scans.

    Accepts multipart/form-data. At least one of *file* or *text* must be
    provided. When both are given the embeddings are fused using *image_weight*.
    """
    if file is None and not text:
        raise HTTPException(status_code=400, detail="Provide at least one of: file, text")

    img_vec = None
    txt_vec = None

    if file is not None:
        if not (file.filename or "").endswith(".ibw"):
            raise HTTPException(status_code=400, detail="Uploaded file must be a .ibw file")
        contents = await file.read()
        with tempfile.NamedTemporaryFile(suffix=".ibw", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = Path(tmp.name)
        try:
            try:
                array, _ = parse_ibw(tmp_path)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Failed to parse .ibw: {exc}")
            image = preprocess(array)
            img_vec = encoder.embed_image(image)
        finally:
            tmp_path.unlink(missing_ok=True)

    if text:
        txt_vec = encoder.embed_text(text)

    if img_vec is not None and txt_vec is not None:
        query_vec = encoder.fuse(img_vec, txt_vec, image_weight=image_weight)
    elif img_vec is not None:
        query_vec = img_vec
    else:
        query_vec = txt_vec

    filter_dict = json.loads(filters) if filters else None
    rows = store.search(query_vec, top_k=top_k, filters=filter_dict)

    hits = [
        SearchHit(
            sample_id=row["sample_id"],
            filename=row["filename"],
            score=row["score"],
            model_version=row["model_version"],
            metadata=AFMMetadata(
                material=row.get("material"),
                substrate=row.get("substrate"),
                technique=row.get("technique"),
                scan_size_um=row.get("scan_size_um"),
                raw_text=row.get("raw_text") or "",
            ),
        )
        for row in rows
    ]
    return SearchResponse(query_text=text, results=hits)


@router.get("/sample/{sample_id}")
async def get_sample(
    sample_id: str,
    store: VectorStoreDep,
) -> JSONResponse:
    """Retrieve metadata for a single ingested sample by ID.

    Raises:
        HTTPException 404: If no sample with *sample_id* exists.
    """
    row = store.get(sample_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found")
    if row.get("created_at") is not None:
        row["created_at"] = row["created_at"].isoformat()
    return JSONResponse(content=row)
