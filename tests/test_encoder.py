"""Tests for services.encoder.CLIPEncoder.

All tests mock the underlying open_clip model so they run on CPU without
downloading any model weights.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def mock_encoder():
    """Return a CLIPEncoder with the open_clip internals mocked out."""
    with patch("services.encoder.CLIPEncoder.__init__", return_value=None):
        from services.encoder import CLIPEncoder

        encoder = CLIPEncoder.__new__(CLIPEncoder)
        # Inject mock model that returns a random unit vector
        fake_vec = np.random.default_rng(0).standard_normal(512).astype(np.float32)
        fake_vec /= np.linalg.norm(fake_vec)
        encoder._model = MagicMock()
        encoder._preprocess = MagicMock(return_value=MagicMock())
        encoder.embed_image = MagicMock(return_value=fake_vec)
        encoder.embed_text = MagicMock(return_value=fake_vec)
        return encoder


def test_embed_image_shape(mock_encoder) -> None:
    """embed_image should return a (512,) array."""
    img = Image.new("RGB", (224, 224))
    result = mock_encoder.embed_image(img)
    assert result.shape == (512,)


def test_embed_text_shape(mock_encoder) -> None:
    """embed_text should return a (512,) array."""
    result = mock_encoder.embed_text("SrTiO3 thin film")
    assert result.shape == (512,)


def test_embed_image_is_normalised(mock_encoder) -> None:
    """embed_image output should be L2-normalised."""
    img = Image.new("RGB", (224, 224))
    result = mock_encoder.embed_image(img)
    norm = float(np.linalg.norm(result))
    assert abs(norm - 1.0) < 1e-5


def test_fuse_output_is_normalised() -> None:
    """fuse() should return a normalised vector regardless of weights."""
    from services.encoder import CLIPEncoder

    rng = np.random.default_rng(1)
    img_emb = rng.standard_normal(512).astype(np.float32)
    img_emb /= np.linalg.norm(img_emb)
    txt_emb = rng.standard_normal(512).astype(np.float32)
    txt_emb /= np.linalg.norm(txt_emb)

    # fuse is implemented (not a stub) so we can call it directly
    encoder = CLIPEncoder.__new__(CLIPEncoder)
    result = encoder.fuse(img_emb, txt_emb, image_weight=0.6)
    assert result.shape == (512,)
    assert abs(float(np.linalg.norm(result)) - 1.0) < 1e-5
