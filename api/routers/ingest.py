"""POST /ingest — upload an .ibw file and ingest it into the vector store."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from api.core.config import settings
from api.core.deps import EncoderDep, VectorStoreDep
from api.models.schemas import IngestResponse

router = APIRouter()


@router.post("", response_model=IngestResponse)
async def ingest_scan(
    encoder: EncoderDep,
    store: VectorStoreDep,
    file: UploadFile = File(..., description="An .ibw AFM scan file"),
    text: str = Form(..., description="Free-text description of the sample"),
    sample_id: str | None = Form(None, description="Optional caller-supplied sample ID"),
) -> IngestResponse:
    """Ingest an uploaded .ibw file.

    Pipeline:
        1. Save the upload to a temporary path.
        2. Parse with ``ingestion.parsers.ibw.parse_ibw``.
        3. Preprocess with ``ingestion.preprocessing.preprocess``.
        4. Embed image + text, fuse at ``settings.IMAGE_WEIGHT``.
        5. Extract NER metadata from *text*.
        6. Build and upsert the record.

    Raises:
        HTTPException 400: If the file is not a valid .ibw.
        HTTPException 500: On unexpected ingestion errors.
    """
    raise NotImplementedError(
        "ingest_scan: implement the full ingestion pipeline using the injected "
        "encoder and store. Use tempfile.NamedTemporaryFile to write the upload "
        "before parsing. Derive sample_id from filename if not supplied."
    )
