"""PostgreSQL + pgvector store and in-memory mock for AFM scan embeddings.

Two classes are provided:

``VectorStore``
    Connects to a real PostgreSQL + pgvector instance. Used by the ingestion
    CLI and (via a thread-pool wrapper) by the FastAPI backend. Automatically
    creates the ``afm_scans`` table and HNSW index on first connection.

``VectorStoreMock``
    Pure in-memory implementation using a list of dicts and numpy cosine
    similarity. Implements the same interface as ``VectorStore`` so tests
    and local development can run without Docker.

Table schema managed by this module::

    CREATE TABLE afm_scans (
        sample_id     TEXT PRIMARY KEY,
        filename      TEXT NOT NULL,
        model_version TEXT NOT NULL,
        embedding     vector(512),
        material      TEXT,
        substrate     TEXT,
        technique     TEXT,
        scan_size_um  REAL,
        raw_text      TEXT,
        created_at    TIMESTAMPTZ DEFAULT now()
    );

    CREATE INDEX ON afm_scans USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
"""

from __future__ import annotations

from typing import Any

import numpy as np
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

# Columns returned by SELECT * on afm_scans (order matches table definition)
_COLUMNS = (
    "sample_id",
    "filename",
    "model_version",
    "embedding",
    "material",
    "substrate",
    "technique",
    "scan_size_um",
    "raw_text",
    "created_at",
)

_CREATE_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS afm_scans (
    sample_id     TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    model_version TEXT NOT NULL,
    embedding     vector(512),
    material      TEXT,
    substrate     TEXT,
    technique     TEXT,
    scan_size_um  REAL,
    raw_text      TEXT,
    created_at    TIMESTAMPTZ DEFAULT now()
)
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS afm_scans_hnsw_idx
ON afm_scans USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64)
"""

_UPSERT = """
INSERT INTO afm_scans
    (sample_id, filename, model_version, embedding,
     material, substrate, technique, scan_size_um, raw_text)
VALUES
    (%(sample_id)s, %(filename)s, %(model_version)s, %(embedding)s,
     %(material)s, %(substrate)s, %(technique)s, %(scan_size_um)s, %(raw_text)s)
ON CONFLICT (sample_id) DO UPDATE SET
    filename      = EXCLUDED.filename,
    model_version = EXCLUDED.model_version,
    embedding     = EXCLUDED.embedding,
    material      = EXCLUDED.material,
    substrate     = EXCLUDED.substrate,
    technique     = EXCLUDED.technique,
    scan_size_um  = EXCLUDED.scan_size_um,
    raw_text      = EXCLUDED.raw_text
"""


def _url_to_dsn(db_url: str) -> str:
    """Convert a SQLAlchemy-style URL to a psycopg2-compatible URL.

    Strips the ``+asyncpg`` or ``+psycopg2`` driver suffix so psycopg2
    can parse it directly::

        postgresql+asyncpg://user:pass@host:5432/db
        →  postgresql://user:pass@host:5432/db
    """
    return db_url.replace("+asyncpg", "").replace("+psycopg2", "")


class VectorStore:
    """Synchronous PostgreSQL + pgvector store for AFM scan embeddings.

    Args:
        db_url: SQLAlchemy-style connection string from ``settings.DB_URL``.
            Both ``postgresql+asyncpg://`` and ``postgresql://`` are accepted.

    Example::

        store = VectorStore("postgresql+asyncpg://afm:afm@localhost:5432/afm")
        store.upsert(embedding, record)
        results = store.search(query_vector, top_k=5)
    """

    def __init__(self, db_url: str) -> None:
        self._dsn = _url_to_dsn(db_url)
        self._ensure_schema()

    def _connect(self) -> psycopg2.extensions.connection:
        conn = psycopg2.connect(self._dsn)
        register_vector(conn)
        return conn

    def _ensure_schema(self) -> None:
        """Create the pgvector extension, table, and HNSW index if they don't exist.

        The extension must be created in a plain connection (without register_vector)
        first, because register_vector requires the vector type to already exist in
        the database. After the extension is committed we can use _connect() normally.
        """
        # Step 1: create extension in a plain connection (no register_vector)
        bootstrap = psycopg2.connect(self._dsn)
        try:
            with bootstrap.cursor() as cur:
                cur.execute(_CREATE_EXTENSION)
            bootstrap.commit()
        finally:
            bootstrap.close()

        # Step 2: now that the vector type exists, create table + index normally
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(_CREATE_TABLE)
                cur.execute(_CREATE_INDEX)
            conn.commit()

    def upsert(self, embedding: np.ndarray, metadata: dict) -> None:
        """Insert or update a scan record.

        Args:
            embedding: 1-D float32 array of shape ``(512,)``.
            metadata: Dict from ``ingestion.record.build_record()``.
                Must contain ``sample_id``.

        Raises:
            ValueError: If embedding shape is not ``(512,)``.
        """
        if embedding.shape != (512,):
            raise ValueError(f"Expected embedding shape (512,), got {embedding.shape}")

        params = {
            "sample_id":     metadata["sample_id"],
            "filename":      metadata["filename"],
            "model_version": metadata["model_version"],
            "embedding":     embedding,
            "material":      metadata.get("material"),
            "substrate":     metadata.get("substrate"),
            "technique":     metadata.get("technique"),
            "scan_size_um":  metadata.get("scan_size_um"),
            "raw_text":      metadata.get("raw_text"),
        }
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(_UPSERT, params)
            conn.commit()

    def search(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[dict]:
        """Return top-k most similar scans by cosine similarity.

        Args:
            vector: 1-D float32 query embedding of shape ``(512,)``.
            top_k: Number of results.
            filters: Optional equality filters, e.g. ``{"material": "SrTiO3"}``.
                Only simple column = value filters are supported.

        Returns:
            List of dicts with all ``afm_scans`` columns plus ``score``
            (cosine similarity, higher = more similar), sorted descending.
        """
        where_clause = ""
        where_params: list[Any] = []
        if filters:
            clauses = [f"{col} = %s" for col in filters]
            where_clause = "WHERE " + " AND ".join(clauses)
            where_params = list(filters.values())

        query = f"""
            SELECT
                sample_id, filename, model_version,
                material, substrate, technique, scan_size_um, raw_text,
                created_at,
                1 - (embedding <=> %s::vector) AS score
            FROM afm_scans
            {where_clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        params = [vector, *where_params, vector, top_k]

        with self._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, params)
                rows = cur.fetchall()

        return [dict(row) for row in rows]

    def get(self, sample_id: str) -> dict | None:
        """Return a single scan record by sample_id, or None if not found."""
        query = """
            SELECT sample_id, filename, model_version,
                   material, substrate, technique, scan_size_um, raw_text, created_at
            FROM afm_scans WHERE sample_id = %s
        """
        with self._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, (sample_id,))
                row = cur.fetchone()
        return dict(row) if row else None

    def delete(self, sample_id: str) -> None:
        """Delete a scan record by sample_id.

        Raises:
            KeyError: If no record with *sample_id* exists.
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM afm_scans WHERE sample_id = %s", (sample_id,)
                )
                if cur.rowcount == 0:
                    conn.rollback()
                    raise KeyError(f"No record found with sample_id={sample_id!r}")
            conn.commit()


class VectorStoreMock:
    """In-memory drop-in replacement for ``VectorStore``.

    Uses a plain Python list and numpy cosine similarity. No database or
    Docker required. Intended for unit tests and offline development.

    Example::

        store = VectorStoreMock()
        store.upsert(embedding, record)
        results = store.search(query_vector, top_k=3)
    """

    def __init__(self) -> None:
        self._records: list[dict] = []

    def upsert(self, embedding: np.ndarray, metadata: dict) -> None:
        """Insert or update a record in-memory."""
        if embedding.shape != (512,):
            raise ValueError(f"Expected embedding shape (512,), got {embedding.shape}")

        record = {**metadata, "embedding": embedding.copy()}
        for i, existing in enumerate(self._records):
            if existing["sample_id"] == metadata["sample_id"]:
                self._records[i] = record
                return
        self._records.append(record)

    def search(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[dict]:
        """Return top-k most similar records by cosine similarity."""
        candidates = self._records
        if filters:
            candidates = [
                r for r in candidates
                if all(r.get(k) == v for k, v in filters.items())
            ]

        if not candidates:
            return []

        scores = []
        for record in candidates:
            emb = record["embedding"]
            norm = np.linalg.norm(vector) * np.linalg.norm(emb)
            score = float(np.dot(vector, emb) / norm) if norm > 0 else 0.0
            scores.append((score, record))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [
            {k: v for k, v in rec.items() if k != "embedding"} | {"score": score}
            for score, rec in scores[:top_k]
        ]

    def get(self, sample_id: str) -> dict | None:
        """Return a single record by sample_id, or None if not found."""
        for record in self._records:
            if record["sample_id"] == sample_id:
                return {k: v for k, v in record.items() if k != "embedding"}
        return None

    def delete(self, sample_id: str) -> None:
        """Delete a record by sample_id."""
        for i, record in enumerate(self._records):
            if record["sample_id"] == sample_id:
                self._records.pop(i)
                return
        raise KeyError(f"No record found with sample_id={sample_id!r}")

    def __len__(self) -> int:
        return len(self._records)
