"""Streamlit demo UI for AFM Similarity Search.

Run::

    streamlit run ui/app.py

Environment variables:
    API_BASE_URL: Base URL of the FastAPI backend (default: http://localhost:8000).
    API_KEY:      API key for the X-API-Key header (default: changeme).
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "changeme")
HEADERS = {"X-API-Key": API_KEY}

st.set_page_config(page_title="AFM Similarity Search", layout="wide")
st.title("AFM Similarity Search")

tab_search, tab_ingest = st.tabs(["Search", "Ingest"])

# ---------------------------------------------------------------------------
# Search tab
# ---------------------------------------------------------------------------
with tab_search:
    st.header("Search the corpus")
    query_file = st.file_uploader("Upload a query .ibw file (optional)", type=["ibw"])
    query_text = st.text_area(
        "Describe your sample (optional)",
        placeholder="e.g. SrTiO3 thin film on STO substrate, PFM lateral",
    )
    top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)
    image_weight = st.slider(
        "Image vs. text weight", min_value=0.0, max_value=1.0, value=0.6, step=0.05
    )

    if st.button("Search"):
        if not query_file and not query_text:
            st.error("Provide at least one of: query file or description text.")
        else:
            # TODO: implement search call to POST /search
            st.warning("Search endpoint integration not yet implemented.")

# ---------------------------------------------------------------------------
# Ingest tab
# ---------------------------------------------------------------------------
with tab_ingest:
    st.header("Ingest a new scan")
    ingest_file = st.file_uploader("Upload an .ibw file", type=["ibw"], key="ingest_uploader")
    ingest_text = st.text_area(
        "Sample description",
        placeholder="e.g. GdScO3 substrate, 5×5 µm scan, contact mode",
        key="ingest_text",
    )
    ingest_sample_id = st.text_input(
        "Sample ID (optional — derived from filename if blank)"
    )

    if st.button("Ingest"):
        if not ingest_file or not ingest_text:
            st.error("Both a file and a description are required.")
        else:
            # TODO: implement ingest call to POST /ingest
            st.warning("Ingest endpoint integration not yet implemented.")
