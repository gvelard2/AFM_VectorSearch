"""Preprocessing for AFM height-map arrays.

Converts a raw 2-D height array (in metres) into a PIL image suitable for
ingestion into BiomedCLIP (224×224 RGB, uint8).
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def preprocess(array: np.ndarray) -> Image.Image:
    """Plane-level, smooth, and normalise a height array to a PIL image.

    Processing steps:
        1. Plane levelling — fit and subtract a least-squares plane to remove
           scanner tilt.
        2. Gaussian smoothing — sigma=1 pixel to suppress single-pixel noise
           without smearing step edges.
        3. Percentile normalisation — map [2nd, 98th] percentile to [0, 255]
           and clip, then cast to uint8.
        4. Resize to 224×224 and convert to RGB (three identical channels).

    Args:
        array: 2-D float64 numpy array of height values (any physical units).

    Returns:
        A 224×224 RGB PIL image ready for BiomedCLIP preprocessing.

    Raises:
        ValueError: If *array* is not 2-D.
    """
    raise NotImplementedError(
        "preprocess: implement plane levelling via numpy least-squares, "
        "Gaussian smoothing via scipy.ndimage.gaussian_filter, "
        "percentile normalisation, and PIL resize to 224×224 RGB."
    )
