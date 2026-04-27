"""Build ingestion records for storage in the vector database.

An ingestion record is a flat dict containing the fused embedding vector,
structured metadata, provenance fields, and the model version that produced
the embedding. It maps 1-to-1 with a row in the ``afm_scans`` pgvector table.
"""

from __future__ import annotations

import numpy as np

from api.models.schemas import AFMMetadata


def build_record(
    sample_id: str,
    embedding: np.ndarray,
    metadata: AFMMetadata,
    filename: str,
    model_version: str,
) -> dict:
    """Assemble a storable record from an embedding and its metadata.

    Args:
        sample_id: Unique identifier for this scan (e.g. ``"GV013_0001"``).
        embedding: 1-D float32 numpy array — the fused image+text embedding
            produced by ``CLIPEncoder``. Shape must be ``(512,)``.
        metadata: Structured metadata extracted by the NER pipeline.
        filename: Original ``.ibw`` filename (basename only, no path).
        model_version: Identifier of the embedding model used, e.g.
            ``"microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"``.
            Required on every record to support future model migrations.

    Returns:
        A dict with keys: ``sample_id``, ``filename``, ``model_version``,
        ``embedding`` (as a Python list), and all ``AFMMetadata`` fields
        flattened to the top level.

    Raises:
        ValueError: If *embedding* does not have shape ``(512,)``.
    """
    raise NotImplementedError(
        "build_record: validate embedding shape, convert embedding to list, "
        "merge metadata.model_dump() into the record dict alongside "
        "sample_id, filename, and model_version."
    )
