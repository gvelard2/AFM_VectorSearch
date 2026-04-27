# Notebooks

This directory contains exploratory Jupyter notebooks used to prototype and
validate the ingestion pipeline. **Do not modify these notebooks** — they are
kept as a reference record of the proof-of-concept work.

## Contents

| Notebook                          | Description                                      |
|-----------------------------------|--------------------------------------------------|
| `AFM_Ingestion_Pipeline.ipynb`    | End-to-end ingestion POC: parse → preprocess → embed → similarity probe |

## Production code

The logic in these notebooks has been (or is being) translated into the
`ingestion/` package. For production use, run:

```bash
python -m ingestion.run --file data/<file>.ibw --text "description"
```

Do not import from notebooks directly. All importable production modules live
in `ingestion/`, `services/`, and `api/`.
