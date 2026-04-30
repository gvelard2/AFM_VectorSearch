"""POST /ingest — upload an .ibw file and ingest it into the vector store."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.core.config import settings
from api.core.deps import EncoderDep, VectorStoreDep
from api.models.schemas import AFMMetadata, IngestResponse
from ingestion.parsers.ibw import parse_ibw
from ingestion.preprocessing import preprocess
from ingestion.record import build_record

router = APIRouter()


@router.post("", response_model=IngestResponse)
async def ingest_scan(
    encoder: EncoderDep,
    store: VectorStoreDep,
    file: UploadFile = File(..., description="An .ibw AFM scan file"),
    text: str = Form(..., description="Free-text description of the sample"),
    sample_id: str | None = Form(None, description="Optional caller-supplied sample ID"),
) -> IngestResponse:
    """Ingest an uploaded .ibw file through the full pipeline.

    Steps: parse → preprocess → embed (image + text, fused) → NER → upsert.

    Raises:
        HTTPException 400: If the file is not a valid .ibw or fails to parse.
        HTTPException 500: On unexpected errors.
    """
    if not (file.filename or "").endswith(".ibw"):
        raise HTTPException(status_code=400, detail="File must be a .ibw file")

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
        txt_vec = encoder.embed_text(text)
        fused = encoder.fuse(img_vec, txt_vec, image_weight=settings.IMAGE_WEIGHT)

        try:
            from ingestion.ner import extract_metadata
            metadata = extract_metadata(text)
        except Exception:
            metadata = AFMMetadata(raw_text=text)

        sid = sample_id or Path(file.filename).stem
        record = build_record(
            sample_id=sid,
            embedding=fused,
            metadata=metadata,
            filename=file.filename,
            model_version=settings.MODEL_NAME,
        )
        store.upsert(fused, record)
    finally:
        tmp_path.unlink(missing_ok=True)

    return IngestResponse(
        sample_id=sid,
        filename=file.filename,
        model_version=settings.MODEL_NAME,
    )
