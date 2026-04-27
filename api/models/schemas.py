"""Pydantic schemas for the AFM Similarity Search API.

All models use strict typing and are version-agnostic with respect to the
embedding model — model_version is stored on every ingestion record.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class AFMMetadata(BaseModel):
    """Structured metadata extracted from a researcher's free-text description
    by the MatBERT NER pipeline.
    """

    material: Optional[str] = Field(None, description="Primary material (e.g. 'SrTiO3')")
    substrate: Optional[str] = Field(None, description="Substrate material (e.g. 'STO')")
    technique: Optional[str] = Field(
        None, description="Imaging technique (e.g. 'PFM', 'contact mode')"
    )
    scan_size_um: Optional[float] = Field(
        None, description="Scan size in micrometres, if parseable from text"
    )
    raw_text: str = Field(..., description="Original free-text description")


class IngestRequest(BaseModel):
    """Body for POST /ingest (multipart handled by FastAPI; text fields here)."""

    text: str = Field(..., description="Free-text description of the AFM sample")
    sample_id: Optional[str] = Field(
        None,
        description="Optional caller-supplied ID. Auto-generated from filename if omitted.",
    )


class IngestResponse(BaseModel):
    """Response returned after a successful ingestion."""

    sample_id: str
    filename: str
    model_version: str
    message: str = "Ingested successfully"


class SearchRequest(BaseModel):
    """Body for POST /search."""

    text: Optional[str] = Field(None, description="Free-text query description")
    top_k: int = Field(5, ge=1, le=100, description="Number of results to return")
    image_weight: float = Field(
        0.6, ge=0.0, le=1.0, description="Weight for image embedding in fusion (text = 1 - image_weight)"
    )
    filters: Optional[dict] = Field(
        None, description="Optional SQL metadata filters, e.g. {'material': 'SrTiO3'}"
    )


class SearchHit(BaseModel):
    """A single result from a similarity search."""

    sample_id: str
    filename: str
    score: float = Field(..., description="Cosine similarity score [0, 1]")
    metadata: AFMMetadata
    model_version: str


class SearchResponse(BaseModel):
    """Response returned by POST /search."""

    query_text: Optional[str]
    results: list[SearchHit]
