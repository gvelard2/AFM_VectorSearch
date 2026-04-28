"""Named Entity Recognition for AFM sample descriptions.

Uses the LBNL CEDER group MatBERT model fine-tuned for materials science NER.
Extracts structured metadata fields from free-text researcher descriptions.

Model configuration
-------------------
Set ``NER_MODEL_NAME`` in your ``.env`` file to the HuggingFace model ID of
the CEDER group's fine-tuned NER checkpoint. The pretrained base is::

    lbnlp/MatBERT-pretrained-512

For token-classification (NER) you need a fine-tuned version. Check the CEDER
group's HuggingFace page (https://huggingface.co/lbnlp) for the correct NER
checkpoint and update NER_MODEL_NAME accordingly.

Entity label mapping
--------------------
The CEDER MatBERT NER schema uses these labels:

    MAT  → material name         → AFMMetadata.material
    SPL  → sample / substrate    → AFMMetadata.substrate
    CMT  → characterisation      → AFMMetadata.technique
           method / technique
    DSC  → descriptor            → (ignored; informational only)
    PRO  → property              → (ignored)
    SMT  → symmetry / phase      → (ignored)
    APL  → application           → (ignored)

``scan_size_um`` is extracted via regex (NER models handle numeric values
poorly) — patterns like "5 µm", "5um", "5 micron", "5x5".
"""

from __future__ import annotations

import functools
import re
from typing import Optional

from transformers import Pipeline, pipeline

from api.models.schemas import AFMMetadata

# ---------------------------------------------------------------------------
# Scan-size extraction — regex is more reliable than NER for numeric values
# ---------------------------------------------------------------------------

# Matches: "5 µm", "5um", "5 micron", "5microns", "5 μm", "5x5 µm", "5 × 5 um"
_SCAN_SIZE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:[xX×]\s*\d+(?:\.\d+)?\s*)?"
    r"(?:µm|μm|um|micron|microns)",
    re.IGNORECASE,
)


def _extract_scan_size(text: str) -> Optional[float]:
    """Return the first numeric scan-size value found in *text*, in µm."""
    match = _SCAN_SIZE_RE.search(text)
    if match:
        return float(match.group(1))
    return None


# ---------------------------------------------------------------------------
# NER pipeline — loaded once per process via lru_cache
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _get_ner_pipeline(model_name: str) -> Pipeline:
    """Load and cache the HuggingFace NER pipeline.

    Args:
        model_name: HuggingFace model ID (from ``settings.NER_MODEL_NAME``).

    Returns:
        A ``transformers.Pipeline`` configured for token classification with
        ``aggregation_strategy="simple"`` to merge subword tokens into spans.

    Raises:
        RuntimeError: If the model cannot be downloaded or loaded.
    """
    try:
        return pipeline(
            "ner",
            model=model_name,
            aggregation_strategy="simple",
            device=-1,  # CPU; set to 0 for GPU
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load NER model {model_name!r}. "
            "Check that NER_MODEL_NAME in your .env points to a valid "
            "HuggingFace token-classification checkpoint. "
            "See https://huggingface.co/lbnlp for CEDER group models."
        ) from exc


# ---------------------------------------------------------------------------
# Entity label → AFMMetadata field mapping
# ---------------------------------------------------------------------------

# Only the first entity of each type is used; preference order matters.
_LABEL_TO_FIELD = {
    "MAT": "material",
    "SPL": "substrate",
    "CMT": "technique",
}


def extract_metadata(text: str) -> AFMMetadata:
    """Run NER over a free-text sample description and return structured metadata.

    The CEDER MatBERT NER pipeline is loaded lazily on first call and cached
    for subsequent calls. Scan size is extracted via regex independently of NER.

    Args:
        text: Raw free-text description provided by the researcher, e.g.
            "SrTiO3 thin film on STO substrate, PFM lateral, 5 µm scan".

    Returns:
        An ``AFMMetadata`` instance. Fields without a recognised entity remain
        ``None``.

    Raises:
        RuntimeError: If the NER model cannot be loaded (wrong model ID,
            no internet access, or incompatible checkpoint format).
    """
    from api.core.config import settings

    ner = _get_ner_pipeline(settings.NER_MODEL_NAME)
    entities = ner(text)

    # Collect the first entity word for each label type we care about
    extracted: dict[str, str] = {}
    for ent in entities:
        label = ent.get("entity_group", "")
        if label in _LABEL_TO_FIELD and _LABEL_TO_FIELD[label] not in extracted:
            extracted[_LABEL_TO_FIELD[label]] = ent["word"].strip()

    return AFMMetadata(
        material=extracted.get("material"),
        substrate=extracted.get("substrate"),
        technique=extracted.get("technique"),
        scan_size_um=_extract_scan_size(text),
        raw_text=text,
    )
