"""
Dynamic Corpus Ingestion CLI

Usage:
    python scripts/ingest.py path/to/paper.pdf
    python scripts/ingest.py path/to/folder_of_pdfs/
    python scripts/ingest.py --list          # show all ingested PDFs
    python scripts/ingest.py --reset         # wipe index and registry
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._ingestion_engine import IngestionEngine


def main():
    parser = argparse.ArgumentParser(
        description="Ingest one or more PDFs into the Physics-RAG corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ingest.py paper.pdf
  python scripts/ingest.py data/raw_pdfs/
  python scripts/ingest.py --list
  python scripts/ingest.py --reset
        """,
    )
    parser.add_argument("path", nargs="?",
                        help="Path to a PDF file or a directory containing PDFs.")
    parser.add_argument("--list", action="store_true",
                        help="List all currently ingested PDFs and exit.")
    parser.add_argument("--reset", action="store_true",
                        help="Wipe all indexes and the ingestion registry. Irreversible.")
    args = parser.parse_args()

    engine = IngestionEngine()

    if args.list:
        registry = engine.load_registry()
        if not registry:
            print("No PDFs have been ingested yet.")
            return
        print(f"\n{'PDF':<50} {'Chunks':>6}  {'Ingested At'}")
        print("-" * 75)
        for filename, meta in sorted(registry.items()):
            print(f"{filename:<50} {meta['chunk_count']:>6}  {meta['ingested_at']}")
        total_chunks = sum(m["chunk_count"] for m in registry.values())
        print(f"\nTotal: {len(registry)} PDF(s), {total_chunks} chunks")
        return

    if args.reset:
        confirm = input("This will delete all indexes and the registry. Type YES to confirm: ")
        if confirm.strip() == "YES":
            engine.reset()
            print("All indexes and registry wiped.")
        else:
            print("Aborted.")
        return

    if not args.path:
        parser.print_help()
        sys.exit(1)

    target = Path(args.path)
    if not target.exists():
        print(f"Error: path not found — {target}")
        sys.exit(1)

    if target.is_dir():
        pdfs = sorted(target.glob("**/*.pdf"))
        if not pdfs:
            print(f"No PDFs found under {target}")
            sys.exit(1)
        print(f"Found {len(pdfs)} PDF(s) under {target}\n")
    else:
        if target.suffix.lower() != ".pdf":
            print(f"Error: expected a .pdf file, got {target.suffix}")
            sys.exit(1)
        pdfs = [target]

    t0 = time.time()
    results = engine.ingest_pdfs(pdfs)
    elapsed = time.time() - t0

    print("\n" + "-" * 55)
    skipped  = [r for r in results if r["status"] == "skipped"]
    ingested = [r for r in results if r["status"] == "ingested"]
    failed   = [r for r in results if r["status"] == "failed"]

    if ingested:
        total_chunks = sum(r["chunk_count"] for r in ingested)
        print(f"Ingested {len(ingested)} PDF(s) -> {total_chunks} new chunk(s) "
              f"added to index  [{elapsed:.1f}s]")
    if skipped:
        print(f"Skipped {len(skipped)} already-ingested PDF(s) "
              f"(use --reset to re-ingest everything)")
    if failed:
        print(f"{len(failed)} PDF(s) failed:")
        for r in failed:
            print(f"   {r['filename']}: {r['error']}")

    registry     = engine.load_registry()
    total_pdfs   = len(registry)
    total_chunks = sum(m["chunk_count"] for m in registry.values())
    print(f"\nIndex now contains {total_pdfs} PDF(s) - {total_chunks} chunk(s) total")


if __name__ == "__main__":
    main()
