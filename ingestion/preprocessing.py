"""Preprocessing for AFM height-map arrays.

Converts a raw 2-D height array (in metres) into a 224×224 RGB PIL image
suitable for ingestion into BiomedCLIP.

Pipeline (mirrors the validated steps in notebooks/AFM_Ingestion_Pipeline.ipynb):

    1. Plane levelling   — subtract least-squares tilt plane (removes scanner tilt)
    2. Gaussian smoothing — sigma=0.8 px (removes single-pixel detector noise)
    3. Percentile normalisation — [2nd, 98th] percentile → [0, 1], clipped
    4. Viridis colourmap — float [0,1] → uint8 RGB (matches CLIP training distribution
       for scientific images better than greyscale)
    5. PIL resize to 224×224 RGB (BICUBIC)
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# Matches the notebook's validated parameters
_GAUSS_SIGMA: float = 0.8
_CLIP_SIZE: int = 224
_PCT_LOW: float = 2.0
_PCT_HIGH: float = 98.0


def _plane_level(array: np.ndarray) -> np.ndarray:
    """Subtract a least-squares tilt plane from *array*.

    Fits ``z = a + b*x + c*y`` to all pixels and returns the residuals.
    This removes linear scanner tilt without affecting surface features.

    Args:
        array: 2-D float array of height values.

    Returns:
        Plane-levelled array, same shape as input.
    """
    rows, cols = array.shape
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    # Design matrix: [1, x, y] for each pixel
    A = np.column_stack([np.ones(rows * cols), x.ravel(), y.ravel()])
    coeffs, _, _, _ = np.linalg.lstsq(A, array.ravel(), rcond=None)
    plane = coeffs[0] + coeffs[1] * x + coeffs[2] * y
    return array - plane


def preprocess(array: np.ndarray) -> Image.Image:
    """Plane-level, smooth, and normalise a height array to a PIL image.

    Args:
        array: 2-D float numpy array of height values (any physical units —
            typically metres from ``parse_ibw``).

    Returns:
        A 224×224 RGB PIL image ready for BiomedCLIP's own preprocessing
        transform (which will further resize, centre-crop, and normalise).

    Raises:
        ValueError: If *array* is not 2-D.
    """
    if array.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {array.shape}")

    # 1. Plane levelling
    levelled = _plane_level(array.astype(np.float64))

    # 2. Gaussian smoothing
    smoothed = gaussian_filter(levelled, sigma=_GAUSS_SIGMA)

    # 3. Percentile normalisation → [0, 1]
    p_low = np.percentile(smoothed, _PCT_LOW)
    p_high = np.percentile(smoothed, _PCT_HIGH)
    span = p_high - p_low
    if span < 1e-12:
        # Flat image (e.g. all zeros) — return blank
        normalised = np.zeros_like(smoothed)
    else:
        normalised = np.clip((smoothed - p_low) / span, 0.0, 1.0)

    # 4. Apply viridis colourmap → uint8 RGB
    #    Using matplotlib's viridis matches the notebook and improves
    #    contrast for BiomedCLIP which expects colour scientific images.
    import matplotlib

    cmap = matplotlib.colormaps["viridis"]
    rgb_float = cmap(normalised)[:, :, :3]  # drop alpha channel
    rgb_uint8 = (rgb_float * 255).astype(np.uint8)
    pil_image = Image.fromarray(rgb_uint8, mode="RGB")

    # 5. Resize to 224×224
    if pil_image.size != (_CLIP_SIZE, _CLIP_SIZE):
        pil_image = pil_image.resize((_CLIP_SIZE, _CLIP_SIZE), resample=Image.Resampling.BICUBIC)

    return pil_image
