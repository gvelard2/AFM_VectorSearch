# System Architecture

## Overview

AFM Similarity Search is a multimodal retrieval system. Users submit an `.ibw`
file and a text description; the system returns the most similar scans from the
corpus ranked by fused cosine similarity.

## Data flow

```
Researcher
    │
    │  .ibw file + text description
    ▼
┌───────────────────────────────────────────────────────────┐
│  FastAPI  (/ingest  or  /search)                          │
└───────────┬───────────────────────────────┬───────────────┘
            │                               │
            ▼                               ▼
  ┌──────────────────┐           ┌────────────────────┐
  │  ingestion/      │           │  services/         │
  │  parsers/ibw.py  │           │  encoder.py        │
  │  preprocessing   │──image──► │  BiomedCLIP        │──►  512-d image vec
  │  ner.py          │──text───► │  (open_clip)       │──►  512-d text vec
  │  record.py       │           └────────────────────┘
  └──────────────────┘                     │
            │                              │  fuse (60/40)
            │                              ▼
            │                   ┌────────────────────┐
            │                   │  512-d fused vec   │
            │                   └────────┬───────────┘
            │                            │
            ▼                            ▼
  ┌──────────────────────────────────────────────────────┐
  │  PostgreSQL + pgvector                               │
  │  Table: afm_scans                                    │
  │  Index: HNSW on embedding (vector_cosine_ops)        │
  └──────────────────────────────────────────────────────┘
            │
            │  top-k results (cosine sim + metadata)
            ▼
  ┌──────────────────┐
  │  Streamlit UI    │   (demo)
  │  React UI        │   (production)
  └──────────────────┘
```

## Component responsibilities

| Component              | Responsibility                                              |
|------------------------|-------------------------------------------------------------|
| `ingestion/parsers/`   | Read `.ibw` binary, extract height channel + metadata note |
| `ingestion/preprocessing.py` | Plane-level, smooth, normalise → PIL image          |
| `ingestion/ner.py`     | MatBERT NER: text → structured `AFMMetadata`               |
| `ingestion/record.py`  | Assemble the DB row dict from embedding + metadata          |
| `ingestion/run.py`     | CLI: orchestrates single-file or batch ingestion            |
| `services/encoder.py`  | BiomedCLIP singleton: `embed_image`, `embed_text`, `fuse`  |
| `services/vector_store.py` | pgvector CRUD: `upsert`, `search`, `delete`            |
| `api/`                 | FastAPI app: `/health`, `/ingest`, `/search`, `/sample/{id}`|
| `ui/app.py`            | Streamlit demo with Search and Ingest tabs                  |
| `deploy/`              | Docker Compose (local), Helm chart + Nautilus K8s Jobs      |

## Embedding fusion

Image and text embeddings are each L2-normalised to unit length before fusion.
The default 60/40 weighting was chosen empirically; it is tunable per-request
via `SearchRequest.image_weight` and globally via `Settings.IMAGE_WEIGHT`.

```
fused = 0.6 * image_emb + 0.4 * text_emb
fused = fused / ||fused||
```

## Deployment topology

```
Local dev:   docker compose up  →  postgres (pgvector) + api + streamlit

Kubernetes:  Helm chart         →  Deployment(api) + Service
                                   Job(ingestion, GPU)
                                   PVC (Ceph, ReadWriteMany)

Nautilus NRP: batch_ingest_job.yaml  →  K8s Job requesting nvidia.com/gpu: 1
                                        mounts Ceph PVC at /data
```
