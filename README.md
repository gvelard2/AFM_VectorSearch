# AFM Similarity Search

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com)

A multimodal similarity search tool for Atomic Force Microscopy (AFM) images. Researchers upload an `.ibw` file and a free-text description (e.g. "SrTiO₃ thin film on SrTiO₃ substrate, PFM lateral channel") and get back the most visually and semantically similar AFM images from a shared corpus.

**Live demo:** [https://afm-search.nrp-nautilus.io](https://afm-search.nrp-nautilus.io)

## How it works

Raw `.ibw` files are parsed with `igor2`/`afmformats` and converted to numpy arrays. The height channel is plane-leveled, Gaussian-smoothed, and normalized to a PIL image. [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) (trained on 15M scientific image-text pairs) encodes both the image and the researcher-supplied text into 512-dimensional embeddings. A MatSciBERT NER pipeline extracts structured metadata fields (material, substrate, technique, scan size) from the free-text description. Image and text embeddings are fused at a tunable 60/40 ratio and stored in PostgreSQL with the pgvector extension (HNSW index). At query time the same fusion pipeline runs on the uploaded file and description, and the top-k nearest neighbors are returned with their metadata.

The FastAPI backend exposes `/ingest` and `/search` endpoints; a Streamlit app provides a browser UI. Batch ingestion runs on [Nautilus NRP](https://nationalresearchplatform.org/) as GPU-accelerated Kubernetes Jobs.

## Quick start

### Option 1 — Docker Compose (local, recommended for development)

```bash
git clone https://github.com/gvelard2/AFM_VectorSearch.git
cd AFM_VectorSearch

# Start postgres + API + UI
docker compose -f deploy/docker-compose.yml up --build
```

- API docs: http://localhost:8000/docs
- UI: http://localhost:8501

> **PyTorch:** Docker images install PyTorch CPU automatically. For local dev without Docker, install PyTorch first per the [official guide](https://pytorch.org/get-started/locally/), then `pip install -r requirements.txt`.

### Option 2 — Ingest your own data

```bash
# Single file
python -m ingestion.run --file Data/myscan.ibw --text "SrTiO3 thin film, tapping mode"

# All files from a CSV with per-file descriptions
python -m ingestion.run --csv corpus_descriptions.csv --data-dir Data/

# Dry run (parse + preprocess only, no DB write)
python -m ingestion.run --csv corpus_descriptions.csv --data-dir Data/ --dry-run
```

### Option 3 — Search via API

```bash
# Text query
curl -X POST http://localhost:8000/search \
  -H "X-API-Key: afm-search-2024" \
  -F "text=SrTiO3 tapping mode 2um scan" \
  -F "top_k=5"

# Fetch a specific sample
curl http://localhost:8000/sample/GV0130001 \
  -H "X-API-Key: afm-search-2024"
```

## Data

`Data/` contains 35 prototype `.ibw` files collected on a Bruker/Asylum AFM covering thin film oxides (SrTiO₃, GdScO₃, NdScO₃, PrScO₃) and PFM measurements. File names encode the sample ID and scan index. See [`Data/README.md`](Data/README.md) for provenance details and [`DATA_LICENSE.md`](DATA_LICENSE.md) for licensing terms.

## Deployment on Nautilus NRP

```bash
# Create credentials secret
kubectl create secret generic afm-secrets \
  --from-literal=db-user=afm \
  --from-literal=db-password=<password> \
  --from-literal=db-url=postgresql+asyncpg://afm:<password>@postgres:5432/afm \
  --from-literal=api-key=<api-key> \
  --namespace=<your-namespace>

# Deploy
kubectl apply -f deploy/nautilus/pvc.yaml
kubectl apply -f deploy/nautilus/postgres.yaml
kubectl apply -f deploy/nautilus/api.yaml
kubectl apply -f deploy/nautilus/streamlit.yaml
kubectl apply -f deploy/nautilus/ingress.yaml
```

See [`docs/FORK_GUIDE.md`](docs/FORK_GUIDE.md) for a full walkthrough.

## Contributing

1. Fork the repo and create a feature branch.
2. Install pre-commit hooks: `pre-commit install`
3. Run the test suite: `pytest`
4. Open a PR — CI must pass (ruff lint, mypy type check, pytest) before merge.

## Acknowledgements

This work used resources available through the National Research Platform (NRP) at the University of California, San Diego. NRP has been developed, and is supported in part, by funding from National Science Foundation, from awards 1730158, 1540112, 1541349, 1826967, 2112167, 2100237, and 2120019, as well as additional funding from community partners.

## Citation

If you use this tool in published research, please cite using the metadata in [`CITATION.cff`](CITATION.cff).

## License

Code: [MIT](LICENSE). Data: see [DATA_LICENSE.md](DATA_LICENSE.md).
