"""BiomedCLIP encoder service.

Wraps ``open_clip`` to produce 512-d L2-normalised embeddings for both PIL
images and text strings.

Usage::

    from services.encoder import get_encoder

    encoder = get_encoder()
    image_vec = encoder.embed_image(pil_image)   # shape (512,)
    text_vec  = encoder.embed_text("SrTiO3 thin film on STO substrate")
    fused_vec = encoder.fuse(image_vec, text_vec, image_weight=0.6)

The encoder is a process-level singleton — the model is downloaded from
HuggingFace (~1.7 GB) on first call and cached in ~/.cache/huggingface/.
Subsequent calls return the cached instance immediately.
"""

from __future__ import annotations

import functools

import numpy as np
import open_clip
import torch
from PIL import Image


class CLIPEncoder:
    """Thin wrapper around BiomedCLIP providing image and text embedding.

    Do not instantiate directly — use ``get_encoder()`` to get the
    process-level singleton.

    Attributes:
        model_name: HuggingFace model identifier passed to open_clip.
        device: Torch device string (``"cpu"`` or ``"cuda"``).
    """

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        """Load BiomedCLIP from HuggingFace via open_clip.

        The ``hf-hub:`` prefix tells open_clip to pull the model config and
        weights directly from HuggingFace. The download is ~1.7 GB and is
        cached in ``~/.cache/huggingface/`` after the first call.

        Args:
            model_name: open_clip model tag. Must include the ``hf-hub:``
                prefix for HuggingFace-hosted models, e.g.
                ``"hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"``.
            device: Torch device. ``"cpu"`` for local dev, ``"cuda"`` on
                Nautilus GPU nodes.

        Raises:
            RuntimeError: If the model cannot be downloaded or loaded.
        """
        self.model_name = model_name
        self.device = device

        try:
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(model_name)
            self._tokenizer = open_clip.get_tokenizer(model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load BiomedCLIP model {model_name!r}. "
                "Check your internet connection and that open_clip_torch is installed."
            ) from exc

        self._model.to(device)
        self._model.eval()

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Encode a PIL image into a 512-d L2-normalised unit vector.

        BiomedCLIP's own preprocessing transform is applied on top of the
        image (resize to 224×224, normalise to ImageNet stats) before the
        vision encoder runs.

        Args:
            image: A PIL image in RGB mode. Typically the output of
                ``ingestion.preprocessing.preprocess()``.

        Returns:
            1-D float32 numpy array of shape ``(512,)``, L2-normalised.
        """
        tensor = self._preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self._model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().squeeze().astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        """Encode a text string into a 512-d L2-normalised unit vector.

        The text is tokenized to a maximum of 256 tokens (BiomedCLIP's
        context length). Longer strings are truncated automatically.

        Args:
            text: Raw text to embed — researcher description or search query.

        Returns:
            1-D float32 numpy array of shape ``(512,)``, L2-normalised.
        """
        tokens = self._tokenizer([text]).to(self.device)
        with torch.no_grad():
            features = self._model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().squeeze().astype(np.float32)

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
                Text weight = ``1 - image_weight``. Default 0.6.

        Returns:
            512-d float32 numpy array, L2-normalised.
        """
        fused = image_weight * image_embedding + (1.0 - image_weight) * text_embedding
        norm = np.linalg.norm(fused)
        if norm == 0.0:
            return fused.astype(np.float32)
        return (fused / norm).astype(np.float32)


@functools.lru_cache(maxsize=1)
def get_encoder() -> CLIPEncoder:
    """Return the process-level CLIPEncoder singleton.

    Loads BiomedCLIP on first call (~1.7 GB download, cached after that).
    Subsequent calls return the cached instance immediately.

    Device selection: ``"cuda"`` if a GPU is available, else ``"cpu"``.

    Returns:
        The singleton ``CLIPEncoder`` instance.
    """
    from api.core.config import settings

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = f"hf-hub:{settings.MODEL_NAME}"
    return CLIPEncoder(model_name, device=device)
