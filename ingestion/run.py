"""CLI entrypoint for the AFM ingestion pipeline.

Usage examples::

    # Ingest a single file
    python -m ingestion.run --file Data/GV0130001.ibw --text "SrTiO3 thin film on STO substrate, tapping mode"

    # Batch-ingest a directory (all files share the same description)
    python -m ingestion.run --batch-dir Data/ --text "SrTiO3 on STO substrate"

    # Dry run — parse and preprocess only, no embedding or DB write
    python -m ingestion.run --file Data/GV0130001.ibw --text "test" --dry-run

Note on batch mode: all files in --batch-dir receive the same --text description.
For files with distinct descriptions, run --file once per file with its own --text.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from api.models.schemas import AFMMetadata
from ingestion.parsers.ibw import parse_ibw
from ingestion.preprocessing import preprocess
from ingestion.record import build_record


def _derive_sample_id(path: Path) -> str:
    """Derive a sample ID from the file stem (e.g. 'GV0130001' from 'GV0130001.ibw')."""
    return path.stem


def ingest_file(path: Path, text: str, *, dry_run: bool = False) -> None:
    """Ingest a single .ibw file through the full pipeline.

    Steps:
        1. Parse .ibw → height array + raw metadata dict
        2. Preprocess array → 224×224 RGB PIL image
        3. (dry_run stops here)
        4. Embed image + text via BiomedCLIP
        5. Fuse embeddings (60/40 image/text)
        6. Extract structured metadata via MatBERT NER
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

    # ── Step 1: Parse ────────────────────────────────────────────────────────
    print(f"  [1/6] Parsing {path.name} ...")
    array, ibw_meta = parse_ibw(path)
    print(f"        array shape={array.shape}, dtype={array.dtype}")
    print(f"        metadata fields: {len(ibw_meta)}")

    # ── Step 2: Preprocess ───────────────────────────────────────────────────
    print("  [2/6] Preprocessing height map ...")
    image = preprocess(array)
    print(f"        PIL image: {image.size}, mode={image.mode}")

    if dry_run:
        print("  [dry-run] Skipping embed + DB write. Done.")
        return

    # ── Step 3: Embed ────────────────────────────────────────────────────────
    print("  [3/6] Loading encoder and embedding ...")
    from services.encoder import get_encoder
    encoder = get_encoder()
    img_vec  = encoder.embed_image(image)
    txt_vec  = encoder.embed_text(text)
    fused    = encoder.fuse(img_vec, txt_vec)
    sim      = float(__import__("numpy").dot(img_vec, txt_vec))
    print(f"        image-text cosine similarity: {sim:.4f}")

    # ── Step 4: NER metadata extraction ─────────────────────────────────────
    print("  [4/6] Extracting NER metadata ...")
    try:
        from ingestion.ner import extract_metadata
        metadata = extract_metadata(text)
        print(f"        material={metadata.material}, substrate={metadata.substrate}, "
              f"technique={metadata.technique}, scan_size_um={metadata.scan_size_um}")
    except Exception as exc:
        print(f"        WARNING: NER failed ({exc.__class__.__name__}: {exc})")
        print("        Falling back to raw-text-only metadata.")
        metadata = AFMMetadata(raw_text=text)

    # ── Step 5: Build record ─────────────────────────────────────────────────
    print("  [5/6] Building record ...")
    sample_id = _derive_sample_id(path)
    record = build_record(
        sample_id=sample_id,
        embedding=fused,
        metadata=metadata,
        filename=path.name,
        model_version=settings.MODEL_NAME,
    )

    # ── Step 6: Upsert into pgvector ─────────────────────────────────────────
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

    total = len(paths)
    for i, path in enumerate(paths, 1):
        print(f"\n[{i}/{total}] {path}")
        try:
            ingest_file(path, args.text, dry_run=args.dry_run)
            print(f"  OK")
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
