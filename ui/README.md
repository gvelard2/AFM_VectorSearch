# UI — Streamlit Demo

The `ui/app.py` Streamlit application provides a browser-based interface with
two tabs: **Search** (query the corpus by image + text) and **Ingest** (add a
new scan to the database).

## Running locally

```bash
# From the repo root, with the API already running:
streamlit run ui/app.py
```

The app defaults to `http://localhost:8000` for the API. Override with:

```bash
API_BASE_URL=http://my-api-host:8000 API_KEY=secret streamlit run ui/app.py
```

## Running via Docker Compose

```bash
docker compose up streamlit
```

The service is exposed on port `8501` by default.

## Production UI

For production deployment, a React frontend is planned. The Streamlit app is
for demos and internal lab use only.
