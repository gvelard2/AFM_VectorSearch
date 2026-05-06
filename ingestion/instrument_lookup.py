"""Instrument-specific metadata lookup tables for IBW key → DB column mapping.

Maps raw instrument note keys from IBW files to standardised database column
names, applying unit conversions where necessary.

Currently supported instruments
--------------------------------
Asylum Research (MFP-3D, Cypher) — keys emitted by Asylum Research software.

Adding a new instrument
------------------------
1. Create a ``_<BRAND>_MAP`` dict following the same pattern.
2. Register it in ``_ALL_MAPS``.
3. Add an ``instrument_model`` key so the instrument can be identified.

No training data or NER required — this is a developer-maintained mapping
table based on known instrument output formats.
"""

from __future__ import annotations

from typing import Callable


# ---------------------------------------------------------------------------
# Converter helpers
# ---------------------------------------------------------------------------


def _float(v: str) -> float | None:
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _int(v: str) -> int | None:
    try:
        return int(float(v))  # handles "256.0"
    except (ValueError, TypeError):
        return None


def _scan_size_um(v: str) -> float | None:
    """ScanSize in the IBW note is in metres; convert to µm."""
    f = _float(v)
    return f * 1e6 if f is not None else None


def _str(v: str) -> str | None:
    s = v.strip()
    return s if s else None


# ---------------------------------------------------------------------------
# Asylum Research lookup table
# ---------------------------------------------------------------------------
# fmt: off
_ASYLUM_MAP: dict[str, tuple[str, Callable]] = {
    # IBW key          DB column              converter
    "ScanSize":        ("scan_size_um",       _scan_size_um),  # metres → µm
    "ScanRate":        ("scan_rate_hz",        _float),         # lines/sec
    "ScanAngle":       ("scan_angle_deg",      _float),         # degrees
    "ScanLines":       ("scan_lines",          _int),           # pixels
    "ScanPoints":      ("scan_points",         _int),           # pixels
    "ImagingMode":     ("technique",           _str),           # e.g. "AC Mode", "PFM Mode"
    "DriveFrequency":  ("drive_frequency_hz",  _float),         # Hz
    "DriveAmplitude":  ("drive_amplitude_v",   _float),         # V
    "SpringConstant":  ("spring_constant",     _float),         # N/m
    "TipVoltage":      ("tip_voltage_v",       _float),         # V (relevant for PFM)
    "MicroscopeModel": ("instrument_model",    _str),           # e.g. "MFP3D"
    "Date":            ("scan_date",           _str),           # e.g. "2016-10-12"
}
# fmt: on

# Registry — add new instrument maps here as they are implemented
_ALL_MAPS: list[dict[str, tuple[str, Callable]]] = [
    _ASYLUM_MAP,
    # _BRUKER_MAP,  # placeholder for future Bruker / NanoScope support
    # _PARK_MAP,    # placeholder for future Park Systems support
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_ibw_fields(ibw_meta: dict) -> dict:
    """Apply all registered instrument lookup tables to a raw IBW metadata dict.

    Tries each instrument map in order. Unknown keys are silently ignored.
    Conversion failures leave the column absent from the result (no None
    values are inserted — the caller decides the default).

    Args:
        ibw_meta: Raw key-value dict returned by ``ingestion.parsers.ibw.parse_ibw()``.

    Returns:
        Dict mapping standard DB column names to converted Python values.
        Only successfully parsed keys are included.

    Example::

        _, ibw_meta = parse_ibw(Path("Data/GV0130001.ibw"))
        fields = extract_ibw_fields(ibw_meta)
        # {"scan_size_um": 5.0, "technique": "AC Mode", "scan_lines": 256, ...}
    """
    result: dict = {}
    for instrument_map in _ALL_MAPS:
        for ibw_key, (db_col, convert) in instrument_map.items():
            if ibw_key in ibw_meta and db_col not in result:
                value = convert(ibw_meta[ibw_key])
                if value is not None:
                    result[db_col] = value
    return result
