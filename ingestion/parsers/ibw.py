"""Parser for Igor Binary Wave (.ibw) AFM files.

Uses igor2 to read the binary wave structure and returns the height channel
as a numpy array alongside the raw metadata note as a dict.

IBW wave structure (as returned by igor2.binarywave.load):
    ibw['wave']['wData']  — ndarray, shape (rows, cols, channels) or (rows, cols)
    ibw['wave']['note']   — bytes, colon-delimited key:value metadata pairs

Channel ordering (standard Asylum Research / MFP-3D):
    0: Height      ← used for embedding
    1: Deflection
    2: Amplitude
    3: Phase
"""

from __future__ import annotations

from pathlib import Path

import igor2.binarywave
import numpy as np


def _parse_note(note_bytes: bytes) -> dict:
    """Decode the IBW note block into a flat key-value dict.

    The note is a UTF-8 byte string with one ``Key:Value`` pair per line.
    Lines without a colon are skipped. The split is limited to the first
    colon so values containing colons (e.g. timestamps) are preserved.

    Args:
        note_bytes: Raw bytes from ``ibw['wave']['note']``.

    Returns:
        Dict of ``{key: value}`` strings. Both keys and values are stripped
        of leading/trailing whitespace. Empty values are kept.
    """
    meta: dict = {}
    note = note_bytes.decode("utf-8", errors="ignore")
    for line in note.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta


def parse_ibw(path: Path) -> tuple[np.ndarray, dict]:
    """Load an .ibw file and return the height channel array and raw metadata.

    The height channel is channel index 0, which is the standard topography
    channel for Asylum Research MFP-3D and Cypher instruments. Values are
    returned in the instrument's native units (metres).

    Args:
        path: Path to the .ibw file.

    Returns:
        A tuple of:
            - height_array: 2-D float64 numpy array (rows × cols) in metres.
            - metadata: Dict of all key-value pairs from the IBW note block.
              Typical keys include ``ScanSize``, ``ScanRate``, ``ImagingMode``,
              ``Date``, ``Operator``. See ``Data/README.md`` for field details.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the wave data is not 2-D or 3-D, or has zero channels.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"IBW file not found: {path}")

    ibw = igor2.binarywave.load(str(path))
    wave_data: np.ndarray = ibw["wave"]["wData"]
    raw_note: bytes = ibw["wave"]["note"]

    if wave_data.ndim == 2:
        # Some single-channel files are stored as plain 2-D arrays
        height_array = wave_data.astype(np.float64)
    elif wave_data.ndim == 3:
        if wave_data.shape[2] == 0:
            raise ValueError(f"IBW file has 0 channels: {path}")
        height_array = wave_data[:, :, 0].astype(np.float64)
    else:
        raise ValueError(f"Expected 2-D or 3-D wave data, got shape {wave_data.shape}: {path}")

    metadata = _parse_note(raw_note)
    return height_array, metadata
