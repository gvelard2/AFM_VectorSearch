"""Tests for ingestion.parsers.ibw.parse_ibw."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ingestion.parsers.ibw import parse_ibw


def test_parse_ibw_returns_tuple(sample_ibw_path: Path) -> None:
    """parse_ibw should return a (ndarray, dict) tuple."""
    result = parse_ibw(sample_ibw_path)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_parse_ibw_array_is_2d(sample_ibw_path: Path) -> None:
    """The height array should be 2-dimensional."""
    array, _ = parse_ibw(sample_ibw_path)
    assert isinstance(array, np.ndarray)
    assert array.ndim == 2


def test_parse_ibw_metadata_is_dict(sample_ibw_path: Path) -> None:
    """Metadata should be a non-empty dict."""
    _, metadata = parse_ibw(sample_ibw_path)
    assert isinstance(metadata, dict)
    assert len(metadata) > 0


def test_parse_ibw_file_not_found() -> None:
    """parse_ibw should raise FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        parse_ibw(Path("/nonexistent/path/file.ibw"))


def test_parse_ibw_array_dtype(sample_ibw_path: Path) -> None:
    """Height array should be a floating-point dtype."""
    array, _ = parse_ibw(sample_ibw_path)
    assert np.issubdtype(array.dtype, np.floating)
