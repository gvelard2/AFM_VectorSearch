"""Build ingestion records for storage in the vector database.

An ingestion record is a flat dict containing the fused embedding vector,
structured metadata, provenance fields, and the model version that produced
the embedding. It maps 1-to-1 with a row in the ``afm_scans`` pgvector table.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from api.models.schemas import AFMMetadata


def build_record(
    sample_id: str,
    embedding: np.ndarray,
    metadata: AFMMetadata,
    filename: str,
    model_version: str,
    image: Image.Image | None = None,
) -> dict:
    """Assemble a storable record from an embedding and its metadata.

    Args:
        sample_id: Unique identifier for this scan (e.g. ``"GV013_0001"``).
        embedding: 1-D float32 numpy array of shape ``(512,)`` — the fused
            image+text embedding produced by ``CLIPEncoder.fuse()``.
        metadata: Structured metadata extracted by the NER pipeline.
        filename: Original ``.ibw`` filename (basename only, no path).
        model_version: Identifier of the embedding model, e.g.
            ``"microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"``.
            Stored on every record to support future model migrations.

    Returns:
        A dict with keys ``sample_id``, ``filename``, ``model_version``,
        ``embedding`` (as a float32 numpy array for VectorStore.upsert),
        and all ``AFMMetadata`` fields flattened to the top level.

    Raises:
        ValueError: If *embedding* does not have shape ``(512,)``.
    """
    if embedding.shape != (512,):
        raise ValueError(
            f"Expected embedding shape (512,), got {embedding.shape}"
        )

    buf = io.BytesIO()
    if image is not None:
        image.save(buf, format="PNG")
    image_png: bytes | None = buf.getvalue() or None

    record = {
        "sample_id":     sample_id,
        "filename":      filename,
        "model_version": model_version,
        "embedding":     embedding.astype(np.float32),
        "image_png":     image_png,
    }
    record.update(metadata.model_dump())
    return record
