"""CLI entrypoint for the AFM ingestion pipeline.

Usage examples::

    # Ingest a single file
    python -m ingestion.run --file Data/GV0130001.ibw --text "SrTiO3 thin film on STO substrate, tapping mode"

    # Ingest all files from a CSV (one description per file)
    python -m ingestion.run --csv corpus_descriptions.csv --data-dir Data/

    # Batch-ingest a directory (all files share the same description)
    python -m ingestion.run --batch-dir Data/ --text "SrTiO3 on STO substrate"

    # Dry run — parse and preprocess only, no embedding or DB write
    python -m ingestion.run --csv corpus_descriptions.csv --data-dir Data/ --dry-run
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from api.models.schemas import AFMMetadata
from ingestion.parsers.ibw import parse_ibw
from ingestion.preprocessing import preprocess
from ingestion.record import build_record


def _derive_sample_id(path: Path) -> str:
    """Derive a sample ID from the file stem (e.g. 'GV0130001' from 'GV0130001.ibw')."""
    return path.stem


def _load_csv(csv_path: Path, data_dir: Path) -> list[tuple[Path, str]]:
    """Read corpus_descriptions.csv and return (file_path, description) pairs.

    Args:
        csv_path: Path to the CSV file (columns: filename, description).
        data_dir: Directory where the .ibw files live.

    Returns:
        List of (resolved_path, description) tuples for rows where the file exists.
        Rows whose file cannot be found are skipped with a warning.
    """
    pairs: list[tuple[Path, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"].strip()
            description = row["description"].strip()
            if not description:
                print(f"  WARNING: no description for {filename} — skipping", file=sys.stderr)
                continue
            file_path = data_dir / filename
            if not file_path.exists():
                print(f"  WARNING: file not found: {file_path} — skipping", file=sys.stderr)
                continue
            pairs.append((file_path, description))
    return pairs


def ingest_file(path: Path, text: str, *, dry_run: bool = False) -> None:
    """Ingest a single .ibw file through the full pipeline.

    Steps:
        1. Parse .ibw -> height array + raw metadata dict
        2. Preprocess array -> 224x224 RGB PIL image
        3. (dry_run stops here)
        4. Embed image + text via BiomedCLIP
        5. Fuse embeddings (60/40 image/text)
        6. Extract structured metadata via MatSciBERT NER
        7. Build record dict
        8. Upsert into pgvector database

    Args:
        path: Path to the .ibw file.
        text: Researcher-supplied free-text description of the sample.
        dry_run: If True, stop after preprocessing and print a summary
            without embedding or writing to the database.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    from api.core.config import settings

    # Step 1: Parse
    print(f"  [1/6] Parsing {path.name} ...")
    array, ibw_meta = parse_ibw(path)
    print(f"        array shape={array.shape}, dtype={array.dtype}, metadata fields={len(ibw_meta)}")

    # Step 2: Preprocess
    print("  [2/6] Preprocessing height map ...")
    image = preprocess(array)
    print(f"        PIL image: {image.size}, mode={image.mode}")

    if dry_run:
        print("  [dry-run] Skipping embed + DB write. Done.")
        return

    # Step 3: Embed
    print("  [3/6] Embedding ...")
    from services.encoder import get_encoder

    encoder = get_encoder()
    img_vec = encoder.embed_image(image)
    txt_vec = encoder.embed_text(text)
    fused = encoder.fuse(img_vec, txt_vec)
    import numpy as np

    sim = float(np.dot(img_vec, txt_vec))
    print(f"        image-text cosine similarity: {sim:.4f}")

    # Step 4: NER metadata extraction + IBW instrument lookup
    print("  [4/6] Extracting metadata ...")
    try:
        from ingestion.ner import extract_metadata

        metadata = extract_metadata(text)
        print(
            f"        NER — material={metadata.material}, substrate={metadata.substrate}, "
            f"technique={metadata.technique}, scan_size_um={metadata.scan_size_um}"
        )
    except Exception as exc:
        print(f"        WARNING: NER failed ({exc.__class__.__name__}: {exc})")
        print("        Falling back to raw-text-only metadata.")
        metadata = AFMMetadata(raw_text=text)

    # Override/supplement with ground-truth values from the IBW note block
    from ingestion.instrument_lookup import extract_ibw_fields

    ibw_fields = extract_ibw_fields(ibw_meta)
    if ibw_fields:
        metadata = metadata.model_copy(update=ibw_fields)
        print(f"        IBW lookup — {', '.join(f'{k}={v}' for k, v in ibw_fields.items())}")

    # Step 5: Build record
    print("  [5/6] Building record ...")
    sample_id = _derive_sample_id(path)
    record = build_record(
        sample_id=sample_id,
        embedding=fused,
        metadata=metadata,
        filename=path.name,
        model_version=settings.MODEL_NAME,
        image=image,
    )

    # Step 6: Upsert
    print("  [6/6] Writing to database ...")
    from services.vector_store import VectorStore

    store = VectorStore(settings.DB_URL)
    store.upsert(fused, record)
    print(f"        Stored sample_id={sample_id!r}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ingestion.run",
        description="Ingest AFM .ibw files into the similarity search database.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--file",
        type=Path,
        metavar="PATH",
        help="Path to a single .ibw file to ingest.",
    )
    source.add_argument(
        "--batch-dir",
        type=Path,
        metavar="DIR",
        help="Directory to scan recursively for .ibw files (all share --text).",
    )
    source.add_argument(
        "--csv",
        type=Path,
        metavar="CSV",
        help=(
            "Path to a CSV file with columns 'filename' and 'description'. "
            "Each file gets its own description. Use with --data-dir."
        ),
    )
    parser.add_argument(
        "--text",
        type=str,
        metavar="TEXT",
        help="Free-text description (required for --file and --batch-dir).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("Data"),
        metavar="DIR",
        help="Base directory containing .ibw files listed in --csv (default: Data/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and preprocess only; do not embed or write to the database.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Build list of (path, description) pairs depending on input mode
    if args.csv is not None:
        if not args.csv.exists():
            print(f"CSV file not found: {args.csv}", file=sys.stderr)
            return 1
        pairs = _load_csv(args.csv, args.data_dir)
        if not pairs:
            print("No valid entries found in CSV.", file=sys.stderr)
            return 1

    elif args.file is not None:
        if args.text is None:
            print("--text is required when using --file.", file=sys.stderr)
            return 1
        pairs = [(args.file, args.text)]

    else:  # --batch-dir
        if args.text is None:
            print("--text is required when using --batch-dir.", file=sys.stderr)
            return 1
        paths = sorted(args.batch_dir.rglob("*.ibw"))
        if not paths:
            print(f"No .ibw files found in {args.batch_dir}", file=sys.stderr)
            return 1
        pairs = [(p, args.text) for p in paths]

    # Run ingestion
    total = len(pairs)
    failed = 0
    for i, (path, text) in enumerate(pairs, 1):
        print(f"\n[{i}/{total}] {path.name}")
        print(f"  Description: {text[:80]}{'...' if len(text) > 80 else ''}")
        try:
            ingest_file(path, text, dry_run=args.dry_run)
            print("  OK")
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            failed += 1

    print(f"\nDone. {total - failed}/{total} files ingested successfully.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
