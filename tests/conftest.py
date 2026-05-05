"""Pytest fixtures shared across the test suite.

Fixtures provide:
    - A path to a real sample .ibw file from data/
    - Mock structured metadata
    - A mock 512-d embedding vector
    - A FastAPI TestClient instance
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.models.schemas import AFMMetadata

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "Data"


@pytest.fixture(scope="session")
def sample_ibw_path() -> Path:
    """Return the path to the first .ibw file found in data/."""
    ibw_files = sorted(DATA_DIR.glob("*.ibw"))
    if not ibw_files:
        pytest.skip("No .ibw files found in data/ — skipping file-dependent tests")
    return ibw_files[0]


# ---------------------------------------------------------------------------
# Mock objects
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_metadata() -> AFMMetadata:
    """Return an AFMMetadata instance with both NER and instrument lookup fields."""
    return AFMMetadata(
        material="SrTiO3",
        substrate="STO",
        technique="AC Mode",
        scan_size_um=5.0,
        raw_text="SrTiO3 thin film on STO substrate, AC mode, 5 µm scan",
        scan_rate_hz=1.0,
        scan_angle_deg=0.0,
        scan_lines=256,
        scan_points=256,
        drive_frequency_hz=262150.0,
        drive_amplitude_v=0.51,
        spring_constant=1.0,
        tip_voltage_v=0.0,
        instrument_model="MFP3D",
        scan_date="2024-01-01",
    )


@pytest.fixture
def mock_ibw_meta() -> dict:
    """Return a minimal Asylum IBW note dict for lookup table tests."""
    return {
        "ScanSize": "5e-06",
        "ScanRate": "1.0016",
        "ScanAngle": "0",
        "ScanLines": "256",
        "ScanPoints": "256",
        "ImagingMode": "AC Mode",
        "DriveFrequency": "2.6215e+05",
        "DriveAmplitude": "0.50996",
        "SpringConstant": "1",
        "TipVoltage": "0",
        "MicroscopeModel": "MFP3D",
        "Date": "2024-01-01",
    }


@pytest.fixture
def mock_embedding() -> np.ndarray:
    """Return a random normalised 512-d float32 embedding."""
    rng = np.random.default_rng(seed=42)
    vec = rng.standard_normal(512).astype(np.float32)
    return vec / np.linalg.norm(vec)


# ---------------------------------------------------------------------------
# API test client
# ---------------------------------------------------------------------------


@pytest.fixture
def api_client() -> TestClient:
    """Return a synchronous FastAPI TestClient."""
    from api.main import app

    return TestClient(app)
