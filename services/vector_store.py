"""PostgreSQL + pgvector store for AFM scan embeddings.

The ``VectorStore`` class manages all interactions with the ``afm_scans`` table,
including upsertion, nearest-neighbour search, and deletion. The table uses an
HNSW index on the ``embedding`` column for sub-millisecond ANN queries.

Expected table schema (SQL)::

    CREATE TABLE afm_scans (
        sample_id    TEXT PRIMARY KEY,
        filename     TEXT NOT NULL,
        model_version TEXT NOT NULL,
        embedding    vector(512),
        material     TEXT,
        substrate    TEXT,
        technique    TEXT,
        scan_size_um REAL,
        raw_text     TEXT,
        created_at   TIMESTAMPTZ DEFAULT now()
    );

    CREATE INDEX ON afm_scans USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
"""

from __future__ import annotations

import numpy as np


class VectorStore:
    """Thin async-friendly wrapper around a pgvector-enabled PostgreSQL database.

    Args:
        db_url: asyncpg-compatible connection string, e.g.
            ``"postgresql+asyncpg://user:pass@host:5432/dbname"``.
    """

    def __init__(self, db_url: str) -> None:
        raise NotImplementedError(
            "__init__: store db_url; initialise a SQLAlchemy async engine and "
            "session factory using create_async_engine(db_url)."
        )

    def upsert(self, embedding: np.ndarray, metadata: dict) -> None:
        """Insert or update a scan record in the database.

        Args:
            embedding: 1-D float32 numpy array of shape ``(512,)``.
            metadata: Dict produced by ``ingestion.record.build_record()``.
                Must contain ``sample_id`` as the primary key.

        Raises:
            ValueError: If *embedding* has wrong shape.
            RuntimeError: On database errors.
        """
        raise NotImplementedError(
            "upsert: convert embedding to pgvector format, build an INSERT ... "
            "ON CONFLICT (sample_id) DO UPDATE statement via SQLAlchemy, and execute."
        )

    def search(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[dict]:
        """Find the top-k most similar scans by cosine similarity.

        Args:
            vector: 1-D float32 query embedding of shape ``(512,)``.
            top_k: Number of results to return.
            filters: Optional dict of column-value equality filters applied
                as SQL WHERE clauses (e.g. ``{"material": "SrTiO3"}``).

        Returns:
            List of dicts, each containing all ``afm_scans`` columns plus a
            ``score`` key (cosine similarity in [0, 1]), ordered by score desc.
        """
        raise NotImplementedError(
            "search: build a SELECT ... ORDER BY embedding <=> :vector LIMIT :k "
            "query using pgvector's cosine distance operator, apply any filters, "
            "and return results as a list of dicts."
        )

    def delete(self, sample_id: str) -> None:
        """Remove a scan record from the database.

        Args:
            sample_id: Primary key of the record to delete.

        Raises:
            KeyError: If no record with *sample_id* exists.
        """
        raise NotImplementedError(
            "delete: execute DELETE FROM afm_scans WHERE sample_id = :id "
            "and raise KeyError if rowcount == 0."
        )
