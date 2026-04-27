# Sample Dataset

This directory contains `.ibw` (Igor Binary Wave) prototype files collected
on a Bruker/Asylum AFM. They serve as reference inputs for testing the
ingestion pipeline and are **not** a curated scientific dataset.

## File naming convention

| Prefix   | Meaning                              |
|----------|--------------------------------------|
| `GV013`  | Growth run / sample identifier       |
| `STO`    | SrTiO₃ substrate                     |
| `GSO`    | GdScO₃ substrate                     |
| `NSO`    | NdScO₃ substrate                     |
| `PSO`    | PrScO₃ substrate                     |
| `_PFM`   | Piezoresponse Force Microscopy mode  |
| `_Lat`   | Lateral (in-plane) PFM channel       |
| `_Vert`  | Vertical (out-of-plane) PFM channel  |
| `0000`   | Scan index within a session          |

## Instrument metadata fields

Each `.ibw` note block encodes the following key fields (among others):

| Field            | Description                           |
|------------------|---------------------------------------|
| `ScanSize`       | Scan size in metres                   |
| `ScanRate`       | Line scan rate in Hz                  |
| `ImagingMode`    | Contact / tapping / PFM               |
| `SpringConstant` | Cantilever spring constant (N/m)      |
| `Date`           | Acquisition date                      |
| `Operator`       | Operator initials                     |

## Adding more files

Drop additional `.ibw` files into this directory and run:

```bash
python -m ingestion.run --batch-dir data/ --text "your sample description"
```

See [`DATA_LICENSE.md`](../DATA_LICENSE.md) for provenance and licensing information.
