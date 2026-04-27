"""Parser for Igor Binary Wave (.ibw) AFM files.

Uses igor2 to read the binary wave structure and returns the height channel
as a numpy array alongside the raw metadata note as a dict.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def parse_ibw(path: Path) -> tuple[np.ndarray, dict]:
    """Load an .ibw file and return the height channel array and raw metadata.

    The height channel is assumed to be channel index 0 (standard for most
    Asylum/Bruker topography files). If the file contains multiple channels
    the caller should inspect the metadata ``Channel`` field to confirm.

    Args:
        path: Absolute or relative path to the .ibw file.

    Returns:
        A tuple of:
            - height_array: 2-D float64 numpy array (rows × cols), units in metres.
            - metadata: Dict of key-value pairs decoded from the IBW note block.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the wave data cannot be interpreted as a 2-D scan.
    """
    raise NotImplementedError(
        "parse_ibw: implement using igor2.binarywave.load(path) to read the wave, "
        "extract wave_data[..., 0] for the height channel, and decode the "
        "note field from the wave header into a dict of key=value pairs."
    )
