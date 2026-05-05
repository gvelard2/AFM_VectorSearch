"""FastAPI endpoint tests using TestClient.

Tests mock the encoder and vector store dependencies so no model weights or
database connection are required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.models.schemas import AFMMetadata, SearchHit


@pytest.fixture
def client_with_mocks() -> TestClient:
    """Return a TestClient with encoder and vector store mocked."""
    from api.main import app
    from api.core import deps

    fake_vec = np.zeros(512, dtype=np.float32)
    mock_encoder = MagicMock()
    mock_encoder.embed_image.return_value = fake_vec
    mock_encoder.embed_text.return_value = fake_vec
    mock_encoder.fuse.return_value = fake_vec

    mock_store = MagicMock()
    mock_store.search.return_value = [
        {
            "sample_id": "GV013_0001",
            "filename": "GV0130001.ibw",
            "score": 0.95,
            "model_version": "1.0.0",
            "material": "SrTiO3",
            "substrate": "STO",
            "technique": "AC Mode",
            "scan_size_um": 5.0,
            "raw_text": "SrTiO3 thin film",
            "scan_rate_hz": 1.0,
            "scan_angle_deg": 0.0,
            "scan_lines": 256,
            "scan_points": 256,
            "drive_frequency_hz": 262150.0,
            "drive_amplitude_v": 0.51,
            "spring_constant": 1.0,
            "tip_voltage_v": 0.0,
            "instrument_model": "MFP3D",
            "scan_date": "2024-01-01",
        }
    ]

    app.dependency_overrides[deps.get_encoder] = lambda: mock_encoder
    app.dependency_overrides[deps.get_vector_store] = lambda: mock_store

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()


def test_health(api_client: TestClient) -> None:
    """GET /health should return 200 with status 'ok'."""
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_includes_model_info(api_client: TestClient) -> None:
    """GET /health should include model_name and model_version."""
    data = api_client.get("/health").json()
    assert "model_name" in data
    assert "model_version" in data


def test_search_requires_text_or_file(client_with_mocks: TestClient) -> None:
    """POST /search with no text and no file should return 400."""
    response = client_with_mocks.post(
        "/search",
        data={"top_k": "5"},
        headers={"X-API-Key": "changeme"},
    )
    assert response.status_code == 400


def test_ingest_requires_file(client_with_mocks: TestClient) -> None:
    """POST /ingest without a file should return 422."""
    response = client_with_mocks.post(
        "/ingest",
        data={"text": "test"},
        headers={"X-API-Key": "changeme"},
    )
    assert response.status_code == 422


def test_search_returns_instrument_lookup_fields(client_with_mocks: TestClient) -> None:
    """POST /search results should include instrument lookup table fields."""
    response = client_with_mocks.post(
        "/search",
        data={"text": "SrTiO3 thin film", "top_k": "1"},
        headers={"X-API-Key": "changeme"},
    )
    assert response.status_code == 200
    hit = response.json()["results"][0]["metadata"]
    assert hit["technique"] == "AC Mode"
    assert hit["scan_size_um"] == 5.0
    assert hit["instrument_model"] == "MFP3D"
    assert hit["scan_lines"] == 256
    assert hit["spring_constant"] == 1.0
    assert hit["tip_voltage_v"] == 0.0
