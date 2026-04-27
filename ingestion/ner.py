"""Named Entity Recognition for AFM sample descriptions.

Uses a MatBERT-based transformers pipeline to extract structured materials
metadata from free-text researcher descriptions.
"""

from __future__ import annotations

from api.models.schemas import AFMMetadata


def extract_metadata(text: str) -> AFMMetadata:
    """Run NER over a free-text sample description and return structured metadata.

    The pipeline uses ``m3rg-iitd/matscibert`` (or equivalent MatBERT checkpoint)
    loaded via ``transformers.pipeline("ner", ...)``. Recognised entity types:

        - MAT  → material name (e.g. "SrTiO3")
        - DSC  → descriptor / property (e.g. "thin film")
        - SMT  → symmetry / structure descriptor
        - PRO  → property name
        - CMT  → characterisation method / technique
        - SPL  → sample / substrate identifier

    Entities are aggregated by type and mapped onto the ``AFMMetadata`` fields.

    Args:
        text: Raw free-text description provided by the researcher.

    Returns:
        An ``AFMMetadata`` instance with fields populated from NER output.
        Fields for which no entity was found remain ``None``.

    Raises:
        RuntimeError: If the NER model cannot be loaded.
    """
    raise NotImplementedError(
        "extract_metadata: load the MatBERT NER pipeline via transformers.pipeline(), "
        "run it on `text`, aggregate entity spans by type, and map them to "
        "AFMMetadata fields (material, substrate, technique, scan_size_um)."
    )
