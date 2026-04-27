"""CLI entrypoint for the AFM ingestion pipeline.

Usage examples::

    # Ingest a single file
    python -m ingestion.run --file data/GV013_0001.ibw --text "SrTiO3 thin film"

    # Batch-ingest a directory
    python -m ingestion.run --batch-dir data/ --text "SrTiO3 on STO substrate"

    # Dry run (parse and preprocess only — no embedding or DB write)
    python -m ingestion.run --batch-dir data/ --text "test" --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


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
        help="Directory to scan recursively for .ibw files.",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        metavar="TEXT",
        help="Free-text description of the sample (used for NER and text embedding).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and preprocess only; do not embed or write to the database.",
    )
    return parser


def ingest_file(path: Path, text: str, *, dry_run: bool = False) -> None:
    """Ingest a single .ibw file through the full pipeline.

    Args:
        path: Path to the .ibw file.
        text: Researcher-supplied free-text description.
        dry_run: If True, skip embedding and database write steps.

    Raises:
        FileNotFoundError: If *path* does not exist.
        NotImplementedError: Until the pipeline modules are fully implemented.
    """
    raise NotImplementedError(
        "ingest_file: call parsers.ibw.parse_ibw(path), "
        "preprocessing.preprocess(array), "
        "services.encoder.CLIPEncoder().embed_image(image) + embed_text(text), "
        "ingestion.ner.extract_metadata(text), "
        "ingestion.record.build_record(...), "
        "and services.vector_store.VectorStore().upsert(embedding, metadata). "
        "Skip embed + upsert steps when dry_run=True."
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.file is not None:
        paths = [args.file]
    else:
        paths = sorted(args.batch_dir.rglob("*.ibw"))
        if not paths:
            print(f"No .ibw files found in {args.batch_dir}", file=sys.stderr)
            return 1

    for path in paths:
        print(f"Ingesting {path} ...")
        ingest_file(path, args.text, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    sys.exit(main())
