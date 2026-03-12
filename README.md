# AFM Vector Search

A multimodal similarity search system for Atomic Force Microscopy (AFM) data. The goal is to enable researchers to query a database of AFM scans using either an image (upload a `.ibw` file) or natural language (describe a surface, material, or imaging condition) and retrieve the most similar scans from the dataset.

## How It Works

AFM scans (`.ibw` files) are processed through an ingestion pipeline that extracts height map data, parses instrument metadata, and generates embeddings using OpenAI's CLIP model. Both an image embedding (from the preprocessed height map) and a text embedding (from structured metadata + user-supplied context) are stored in a vector database. At query time, a new scan or text query is embedded and compared against the database using cosine similarity.

```
.ibw files
    │
    ├─► Ingestion Pipeline   →   image embedding (512-d)
    │                        →   text embedding  (512-d)
    │                        →   structured metadata
    │
    └─► Vector Database (Qdrant / Neo4j)
            │
            └─► Similarity Search
                    ├─► Image query  →  find visually similar scans
                    └─► Text query   →  find scans matching a description
```

## Repository Structure

```
AFM_VectorSearch/
├── AFM_Ingestion_Pipeline.ipynb   # Stage 1: ingestion pipeline (see below)
├── GV0130001.ibw                  # Sample AFM scan (Asylum Research format)
├── GV0130001_ingestion.npz        # Output embeddings from sample scan
├── GV0130001_metadata.json        # Output structured metadata from sample scan
└── README.md
```

## Stage 1 — Ingestion Pipeline (`AFM_Ingestion_Pipeline.ipynb`)

This notebook is the first stage of the pipeline. It is intentionally structured as a **visualization notebook** — every processing step is rendered with plots and diagnostic output so the logic can be inspected and validated before being converted into a production script.

**Once validated, this notebook will be translated into `afm_ingest.py` — a standalone executable script that batch-processes a folder of `.ibw` files without any visualization overhead.**

### What the notebook does

| Stage | Description |
|---|---|
| **Load** | Read a `.ibw` binary wave file using `igor2`, extract all 4 channels (Height, Deflection, Amplitude, Phase) and the raw metadata note |
| **Visualize raw channels** | Display all 4 channels side by side to confirm the file loaded correctly |
| **Parse metadata** | Decode the IBW note block into a structured dictionary; extract key fields (scan size, rate, imaging mode, date, operator) |
| **Build semantic text** | Combine parsed metadata with a user-supplied sample description into a single text string for embedding |
| **Preprocess height map** | Plane-level → Gaussian smooth → percentile normalize → resize to 224×224 RGB (CLIP input format) |
| **Visualize preprocessing** | Show each preprocessing step with histograms to confirm normalization |
| **Image embedding** | Pass the preprocessed PIL image through CLIP ViT-B/32 image encoder → 512-d unit vector |
| **Text embedding** | Tokenize and encode the semantic text string through CLIP text encoder → 512-d unit vector |
| **Similarity probe** | Compare the image embedding against several test text queries to validate semantic alignment |
| **Package record** | Bundle embeddings + metadata into a single ingestion record |
| **Save outputs** | Write `_ingestion.npz` (embeddings) and `_metadata.json` (structured metadata) for the next pipeline stage |

### Output files per scan

- `<filename>_ingestion.npz` — NumPy archive containing `image_embedding` (512,), `text_embedding` (512,), and `preprocessed_array` (224×224)
- `<filename>_metadata.json` — JSON with structured metadata, user text, cosine similarity score, and provenance info

## Environment Setup

Dependencies are managed in a dedicated virtual environment (`vsearch_env`).

```bash
# Activate the environment
C:\Users\<you>\envs\vsearch_env\Scripts\activate

# Key dependencies
pip install numpy pandas matplotlib scipy scikit-image Pillow igor2
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install git+https://github.com/openai/CLIP.git
```

## Roadmap

- [ ] `afm_ingest.py` — batch ingestion script (converted from notebook)
- [ ] `02_vector_db_ingestion.ipynb` — populate Qdrant with a folder of scans
- [ ] `03_similarity_search.ipynb` — query interface (image + text)
- [ ] Web UI for drag-and-drop search
