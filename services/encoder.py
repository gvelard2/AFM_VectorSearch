"""BiomedCLIP encoder service.

Wraps ``open_clip`` to produce 512-d L2-normalised embeddings for both PIL
images and text strings. The ``CLIPEncoder`` class is a singleton — call
``CLIPEncoder.instance()`` rather than constructing it directly.
"""

from __future__ import annotations

import functools

import numpy as np
from PIL import Image


class CLIPEncoder:
    """Thin wrapper around BiomedCLIP providing image and text embedding.

    Attributes:
        model_name: HuggingFace / open_clip model identifier.
        device: Torch device string (``"cuda"`` or ``"cpu"``).
    """

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        """Initialise and load the BiomedCLIP model.

        Args:
            model_name: open_clip model tag, e.g.
                ``"microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"``.
            device: Torch device. Defaults to ``"cpu"`` for portability; set to
                ``"cuda"`` for GPU-accelerated ingestion on Nautilus.

        Raises:
            RuntimeError: If the model cannot be downloaded or loaded.
        """
        raise NotImplementedError(
            "__init__: call open_clip.create_model_and_transforms(model_name, ...) "
            "to load the model and preprocessing transform, store them as instance attrs, "
            "and move the model to `device`."
        )

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def instance() -> "CLIPEncoder":
        """Return the process-level singleton encoder.

        The model name is read from ``api.core.config.settings.MODEL_NAME``.
        Device is ``"cuda"`` if available, else ``"cpu"``.
        """
        raise NotImplementedError(
            "instance: import settings, detect torch device, and return "
            "CLIPEncoder(settings.MODEL_NAME, device=device)."
        )

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Encode a PIL image into a 512-d unit vector.

        Args:
            image: A PIL image. Will be preprocessed by the model's own
                transform (resize, normalise) before encoding.

        Returns:
            A 1-D float32 numpy array of shape ``(512,)``, L2-normalised.
        """
        raise NotImplementedError(
            "embed_image: apply self._preprocess(image), run through "
            "self._model.encode_image(), normalise, and return as numpy array."
        )

    def embed_text(self, text: str) -> np.ndarray:
        """Encode a text string into a 512-d unit vector.

        Args:
            text: Raw text to embed (researcher description or search query).

        Returns:
            A 1-D float32 numpy array of shape ``(512,)``, L2-normalised.
        """
        raise NotImplementedError(
            "embed_text: tokenise with open_clip.tokenize([text]), run through "
            "self._model.encode_text(), normalise, and return as numpy array."
        )

    def fuse(
        self,
        image_embedding: np.ndarray,
        text_embedding: np.ndarray,
        image_weight: float = 0.6,
    ) -> np.ndarray:
        """Linearly fuse image and text embeddings and re-normalise.

        Args:
            image_embedding: 512-d L2-normalised image vector.
            text_embedding: 512-d L2-normalised text vector.
            image_weight: Weight for the image embedding in [0, 1].
                The text weight is ``1 - image_weight``.

        Returns:
            A 512-d float32 numpy array, L2-normalised.
        """
        fused = image_weight * image_embedding + (1.0 - image_weight) * text_embedding
        norm = np.linalg.norm(fused)
        if norm == 0.0:
            return fused
        return (fused / norm).astype(np.float32)
