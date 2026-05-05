"""Tests for ingestion.instrument_lookup.extract_ibw_fields."""

from __future__ import annotations

import pytest

from ingestion.instrument_lookup import extract_ibw_fields


def test_asylum_scan_size_converted_to_um(mock_ibw_meta: dict) -> None:
    """ScanSize in metres should be converted to micrometres."""
    fields = extract_ibw_fields(mock_ibw_meta)
    assert fields["scan_size_um"] == pytest.approx(5.0)


def test_asylum_technique_mapped(mock_ibw_meta: dict) -> None:
    """ImagingMode should map to technique column."""
    fields = extract_ibw_fields(mock_ibw_meta)
    assert fields["technique"] == "AC Mode"


def test_asylum_pfm_technique(mock_ibw_meta: dict) -> None:
    """PFM Mode should be preserved as-is."""
    meta = {**mock_ibw_meta, "ImagingMode": "PFM Mode"}
    fields = extract_ibw_fields(meta)
    assert fields["technique"] == "PFM Mode"


def test_asylum_integer_fields(mock_ibw_meta: dict) -> None:
    """ScanLines and ScanPoints should be parsed as integers."""
    fields = extract_ibw_fields(mock_ibw_meta)
    assert isinstance(fields["scan_lines"], int)
    assert fields["scan_lines"] == 256
    assert isinstance(fields["scan_points"], int)
    assert fields["scan_points"] == 256


def test_asylum_float_fields(mock_ibw_meta: dict) -> None:
    """Numeric fields should be parsed as floats."""
    fields = extract_ibw_fields(mock_ibw_meta)
    assert isinstance(fields["scan_rate_hz"], float)
    assert isinstance(fields["drive_frequency_hz"], float)
    assert isinstance(fields["spring_constant"], float)
    assert isinstance(fields["tip_voltage_v"], float)


def test_asylum_instrument_model(mock_ibw_meta: dict) -> None:
    """MicroscopeModel should map to instrument_model."""
    fields = extract_ibw_fields(mock_ibw_meta)
    assert fields["instrument_model"] == "MFP3D"


def test_asylum_scan_date(mock_ibw_meta: dict) -> None:
    """Date should map to scan_date."""
    fields = extract_ibw_fields(mock_ibw_meta)
    assert fields["scan_date"] == "2024-01-01"


def test_unknown_keys_ignored() -> None:
    """Keys not in any lookup table should not appear in the result."""
    meta = {"UnknownKey": "somevalue", "AnotherUnknown": "123"}
    fields = extract_ibw_fields(meta)
    assert fields == {}


def test_empty_meta_returns_empty_dict() -> None:
    """An empty IBW meta dict should return an empty result."""
    assert extract_ibw_fields({}) == {}


def test_bad_numeric_value_skipped() -> None:
    """A non-numeric value for a numeric field should be silently skipped."""
    meta = {"ScanRate": "not_a_number", "ImagingMode": "AC Mode"}
    fields = extract_ibw_fields(meta)
    assert "scan_rate_hz" not in fields
    assert fields["technique"] == "AC Mode"


def test_all_asylum_fields_present(mock_ibw_meta: dict) -> None:
    """All 12 Asylum keys should produce the expected 12 DB columns."""
    fields = extract_ibw_fields(mock_ibw_meta)
    expected = {
        "scan_size_um", "scan_rate_hz", "scan_angle_deg", "scan_lines",
        "scan_points", "technique", "drive_frequency_hz", "drive_amplitude_v",
        "spring_constant", "tip_voltage_v", "instrument_model", "scan_date",
    }
    assert set(fields.keys()) == expected


def test_real_ibw_file(sample_ibw_path) -> None:
    """extract_ibw_fields should populate key columns from a real .ibw file."""
    from ingestion.parsers.ibw import parse_ibw
    _, ibw_meta = parse_ibw(sample_ibw_path)
    fields = extract_ibw_fields(ibw_meta)
    assert "scan_size_um" in fields
    assert fields["scan_size_um"] > 0
    assert "technique" in fields
    assert fields["technique"] in ("AC Mode", "PFM Mode")
    assert "instrument_model" in fields
