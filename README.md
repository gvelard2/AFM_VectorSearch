# AFM Similarity Search

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com)

A multimodal similarity search tool for Atomic Force Microscopy (AFM) images. Researchers upload an `.ibw` file and a free-text description (e.g. "SrTiO₃ thin film on SrTiO₃ substrate, PFM lateral channel") and get back the most visually and semantically similar AFM images from a shared corpus.

## Architecture overview

Raw `.ibw` files are parsed with `igor2`/`afmformats` and converted to numpy arrays. The height channel is plane-leveled, Gaussian-smoothed, and normalized to a PIL image. BiomedCLIP (trained on 15 M scientific image-text pairs) encodes both the image and the researcher-supplied text into 512-dimensional embeddings. A MatBERT NER pipeline extracts structured metadata fields (material, substrate, technique, scan size) from the free-text description. Image and text embeddings are fused at a tunable 60/40 ratio and stored in PostgreSQL with the pgvector extension (HNSW index). At query time the same fusion pipeline runs on the uploaded file and description, and the top-k nearest neighbors are returned with their metadata. The FastAPI backend exposes `/ingest` and `/search` endpoints; a Streamlit app provides a browser UI for demos.

Batch ingestion jobs run on [Nautilus NRP](https://nationalresearchplatform.org/) as Kubernetes Jobs with GPU acceleration. Persistent storage uses a Ceph PVC. See `deploy/nautilus/` for standalone job specs and `deploy/helm/` for the full Helm chart.

## Quick start

### Option 1 — Docker Compose (local)

```bash
git clone https://github.com/<your-org>/afm-search.git
cd afm-search
cp .env.example .env          # fill in DB_URL and API_KEY
docker compose up --build
```

- API docs: http://localhost:8000/docs
- UI:       http://localhost:8501

### Option 2 — Helm on Kubernetes

```bash
helm install afm-search ./deploy/helm/afm-search -f deploy/helm/afm-search/values.yaml
```

> **PyTorch:** Docker images install PyTorch automatically. For local dev, install it manually per the [official guide](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`.

## Data

`data/` contains prototype `.ibw` files collected on a Bruker/Asylum AFM. File names encode the sample ID and scan index (e.g. `GV013_0001.ibw`). See [`data/README.md`](data/README.md) for provenance details and [`DATA_LICENSE.md`](DATA_LICENSE.md) for licensing terms.

To add your own files, drop `.ibw` files into `data/` and run:

```bash
python -m ingestion.run --batch-dir data/ --text "your sample description"
```

## Contributing

1. Fork the repo and create a feature branch.
2. Install pre-commit hooks: `pre-commit install`
3. Run the test suite: `pytest`
4. Open a PR — CI must pass (ruff lint, mypy type check, pytest) before merge.

See [`docs/FORK_GUIDE.md`](docs/FORK_GUIDE.md) if you are adapting this system for your own lab.

## Citation

If you use this tool in published research, please cite using the metadata in [`CITATION.cff`](CITATION.cff).

## License

Code: [MIT](LICENSE). Data: see [DATA_LICENSE.md](DATA_LICENSE.md).
