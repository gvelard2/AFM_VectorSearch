"""Streamlit demo UI for AFM Similarity Search.

Run::

    streamlit run ui/app.py

Environment variables:
    API_BASE_URL: Base URL of the FastAPI backend (default: http://localhost:8000).
    API_KEY:      API key for the X-API-Key header (default: changeme).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Ensure the project root is on sys.path so ingestion/services are importable
# regardless of which directory Streamlit is launched from.
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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
def _ibw_preview(uploaded_file) -> None:
    """Render a viridis height-map preview for an uploaded .ibw file."""
    from ingestion.parsers.ibw import parse_ibw
    from ingestion.preprocessing import preprocess

    with tempfile.NamedTemporaryFile(suffix=".ibw", delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = Path(tmp.name)
    try:
        array, _ = parse_ibw(tmp_path)
        image = preprocess(array)
        st.image(image, caption=f"Height Map — {uploaded_file.name}", use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render preview: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)


with tab_search:
    st.header("Search the corpus")
    upload_col, preview_col = st.columns([2, 1])
    with upload_col:
        query_file = st.file_uploader("Upload a query .ibw file (optional)", type=["ibw"])
    with preview_col:
        if query_file:
            _ibw_preview(query_file)
    query_text = st.text_area(
        "Describe your sample (optional)",
        placeholder="e.g. SrTiO3 thin film on STO substrate, PFM lateral",
    )
    top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)
    image_weight = st.slider("Image vs. text weight", min_value=0.0, max_value=1.0, value=0.6, step=0.05)

    with st.expander("Filters (optional)"):
        filter_technique = st.selectbox("Technique", options=["Any", "AC Mode", "PFM Mode"])
        filter_instrument = st.selectbox("Instrument model", options=["Any", "MFP3D"])
        filter_scan_size = st.selectbox("Scan size (µm)", options=["Any", "1.0", "2.0", "5.0", "10.0"])

    if st.button("Search"):
        if not query_file and not query_text:
            st.error("Provide at least one of: query file or description text.")
        else:
            with st.spinner("Searching..."):
                form_data = {"top_k": str(top_k), "image_weight": str(image_weight)}
                if query_text:
                    form_data["text"] = query_text

                filters: dict = {}
                if filter_technique != "Any":
                    filters["technique"] = filter_technique
                if filter_instrument != "Any":
                    filters["instrument_model"] = filter_instrument
                if filter_scan_size != "Any":
                    filters["scan_size_um"] = float(filter_scan_size)
                if filters:
                    import json

                    form_data["filters"] = json.dumps(filters)

                files = (
                    {"file": (query_file.name, query_file.getvalue(), "application/octet-stream")} if query_file else {}
                )
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/search",
                        headers=HEADERS,
                        data=form_data,
                        files=files,
                        timeout=60,
                    )
                    resp.raise_for_status()
                    results = resp.json()["results"]
                except requests.HTTPError as e:
                    st.error(f"API error {e.response.status_code}: {e.response.text}")
                    results = []
                except requests.RequestException as e:
                    st.error(f"Could not reach API at {API_BASE_URL}: {e}")
                    results = []

            if results:
                st.success(f"{len(results)} result(s) found.")
                for hit in results:
                    meta = hit["metadata"]
                    with st.expander(f"{hit['sample_id']}  —  score {hit['score']:.4f}"):
                        col1, col2, col3, col_img = st.columns([2, 2, 2, 1])
                        with col1:
                            st.markdown(f"**Filename:** `{hit['filename']}`")
                            st.markdown(f"**Score:** `{hit['score']:.4f}`")
                            st.markdown(f"**Model:** `{hit['model_version']}`")
                            st.markdown("---")
                            st.markdown("**Sample info (NER)**")
                            st.markdown(f"**Material:** {meta.get('material') or '—'}")
                            st.markdown(f"**Substrate:** {meta.get('substrate') or '—'}")
                        with col2:
                            st.markdown("**Scan geometry**")
                            st.markdown(f"**Technique:** {meta.get('technique') or '—'}")
                            st.markdown(f"**Scan size:** {meta.get('scan_size_um') or '—'} µm")
                            st.markdown(f"**Scan lines:** {meta.get('scan_lines') or '—'} px")
                            st.markdown(f"**Scan points:** {meta.get('scan_points') or '—'} px")
                            st.markdown(f"**Scan rate:** {meta.get('scan_rate_hz') or '—'} Hz")
                            st.markdown(f"**Scan angle:** {meta.get('scan_angle_deg') or '—'}°")
                        with col3:
                            st.markdown("**Instrument / cantilever**")
                            st.markdown(f"**Instrument:** {meta.get('instrument_model') or '—'}")
                            st.markdown(f"**Scan date:** {meta.get('scan_date') or '—'}")
                            st.markdown(f"**Drive freq:** {meta.get('drive_frequency_hz') or '—'} Hz")
                            st.markdown(f"**Drive amp:** {meta.get('drive_amplitude_v') or '—'} V")
                            st.markdown(f"**Spring const:** {meta.get('spring_constant') or '—'} N/m")
                            st.markdown(f"**Tip voltage:** {meta.get('tip_voltage_v') or '—'} V")
                        with col_img:
                            img_resp = requests.get(
                                f"{API_BASE_URL}/sample/{hit['sample_id']}/image",
                                headers=HEADERS,
                                timeout=10,
                            )
                            if img_resp.status_code == 200:
                                st.image(
                                    img_resp.content,
                                    caption=f"Height Map — {hit['filename']}",
                                    use_container_width=True,
                                )
                        st.caption(meta.get("raw_text", ""))
            elif results is not None:
                st.info("No results returned.")

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
    ingest_sample_id = st.text_input("Sample ID (optional — derived from filename if blank)")

    if st.button("Ingest"):
        if not ingest_file or not ingest_text:
            st.error("Both a file and a description are required.")
        else:
            with st.spinner("Ingesting..."):
                form_data = {"text": ingest_text}
                if ingest_sample_id:
                    form_data["sample_id"] = ingest_sample_id
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/ingest",
                        headers=HEADERS,
                        data=form_data,
                        files={"file": (ingest_file.name, ingest_file.getvalue(), "application/octet-stream")},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    st.success(
                        f"Ingested `{result['filename']}` as `{result['sample_id']}` "
                        f"using model `{result['model_version']}`."
                    )
                except requests.HTTPError as e:
                    st.error(f"API error {e.response.status_code}: {e.response.text}")
                except requests.RequestException as e:
                    st.error(f"Could not reach API at {API_BASE_URL}: {e}")
