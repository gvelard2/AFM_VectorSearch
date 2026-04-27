# Fork Guide — Deploying AFM Similarity Search for Your Lab

This guide walks through everything a new lab needs to stand up their own
instance of AFM Similarity Search, from forking the repo to running their
first similarity search.

## Step 1 — Fork and clone

```bash
# Fork on GitHub, then:
git clone https://github.com/<your-org>/afm-search.git
cd afm-search
```

## Step 2 — Configure your environment

Copy the example env file and fill in your values:

```bash
cp .env.example .env
```

Key variables:

| Variable        | Description                                              | Default                                                |
|-----------------|----------------------------------------------------------|--------------------------------------------------------|
| `DB_URL`        | asyncpg connection string to your PostgreSQL instance    | `postgresql+asyncpg://afm:afm@localhost:5432/afm`      |
| `MODEL_NAME`    | open_clip model identifier                               | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` |
| `MODEL_VERSION` | Semantic version string stored on every DB record        | `1.0.0`                                                |
| `API_KEY`       | Shared secret for `X-API-Key` header                     | `changeme` — **change this in production**             |

## Step 3 — Set up the database

Ensure you have a PostgreSQL 15+ instance with the pgvector extension enabled:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

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

CREATE INDEX ON afm_scans
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

Or just run `docker compose up postgres` and the table will be created by the
API on first startup (once migration support is added — see roadmap).

## Step 4 — Install dependencies

```bash
# Install PyTorch first (platform-specific):
# https://pytorch.org/get-started/locally/

pip install -r requirements.txt
pre-commit install
```

## Step 5 — Ingest your data

Drop your `.ibw` files into `data/` and run:

```bash
python -m ingestion.run --batch-dir data/ --text "your sample description"
```

For large datasets, use the Kubernetes batch job (see Step 7).

## Step 6 — Run locally

```bash
# Option A: bare Python
uvicorn api.main:app --reload
streamlit run ui/app.py

# Option B: Docker Compose (recommended)
docker compose up --build
```

API docs: http://localhost:8000/docs
UI: http://localhost:8501

## Step 7 — Deploy on Nautilus NRP (optional)

1. Push your Docker images to a registry accessible from Nautilus:
   ```bash
   docker build -f deploy/Dockerfile.ingestion -t ghcr.io/<your-org>/afm-search-ingestion .
   docker push ghcr.io/<your-org>/afm-search-ingestion
   ```
2. Create a K8s Secret for your DB credentials:
   ```bash
   kubectl create secret generic afm-search-secrets \
     --from-literal=db-url="postgresql+asyncpg://..."
   ```
3. Edit `deploy/nautilus/batch_ingest_job.yaml` — set your namespace and PVC name.
4. Submit the job:
   ```bash
   kubectl apply -f deploy/nautilus/batch_ingest_job.yaml
   ```

## Step 8 — Customise the embedding model

To swap BiomedCLIP for a fine-tuned model:

1. Publish your model to HuggingFace as `<your-org>/afm-clip`.
2. Update `MODEL_NAME` in `.env` and `values.yaml`.
3. Bump `MODEL_VERSION` — all new ingestion records will carry the new version,
   enabling you to filter or re-index by model version later.

## Roadmap for forkers

- [ ] SQL migration scripts (Alembic)
- [ ] Authentication beyond a single shared API key
- [ ] HuggingFace Spaces Gradio demo deployment
- [ ] React production UI
