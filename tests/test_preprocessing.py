"""Tests for ingestion.preprocessing.preprocess."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from ingestion.preprocessing import preprocess


@pytest.fixture
def synthetic_array() -> np.ndarray:
    """Return a 64×64 float64 array with a simple gradient (no file I/O needed)."""
    rng = np.random.default_rng(seed=0)
    return rng.standard_normal((64, 64))


def test_preprocess_returns_pil_image(synthetic_array: np.ndarray) -> None:
    """preprocess should return a PIL Image."""
    result = preprocess(synthetic_array)
    assert isinstance(result, Image.Image)


def test_preprocess_output_size(synthetic_array: np.ndarray) -> None:
    """Output image should be 224×224."""
    result = preprocess(synthetic_array)
    assert result.size == (224, 224)


def test_preprocess_output_mode(synthetic_array: np.ndarray) -> None:
    """Output image should be RGB."""
    result = preprocess(synthetic_array)
    assert result.mode == "RGB"


def test_preprocess_rejects_1d() -> None:
    """preprocess should raise ValueError for non-2D input."""
    with pytest.raises(ValueError):
        preprocess(np.zeros(512))


def test_preprocess_pixel_range(synthetic_array: np.ndarray) -> None:
    """All pixel values should be in [0, 255]."""
    result = preprocess(synthetic_array)
    arr = np.array(result)
    assert arr.min() >= 0
    assert arr.max() <= 255
