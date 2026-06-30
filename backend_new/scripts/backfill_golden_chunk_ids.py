"""
backfill_golden_chunk_ids.py
----------------------------
One-time fix for the Stage-3 RAG evaluation.

PROBLEM (diagnosed 2026-06-07)
  The golden set (nvidia_golden_qa.jsonl) stored `source_chunk` as a 301-char
  TRUNCATED prefix of a blank-line PARAGRAPH (synthesize_data.py chunking),
  while the retrieval index (data/embeddings/docs.json) is chunked by 512-word
  SENTENCE WINDOWS. The two chunk spaces are disjoint: 0 of 1137 index chunks
  equal a generation paragraph, so the ground-truth chunk literally cannot be
  "hit" by the retriever. Combined with a length-mismatched 0.70 cosine
  threshold, this drove Hit@1 to 0.16 / Hit@3 to 0.27 while NDCG@3 sat at 0.96
  — a contradiction that exposed the metric, not the retriever, as broken.

FIX (this script)
  Map every golden question's truncated `source_chunk` to the index chunk that
  actually CONTAINS its source text, and record a stable `chunk_id` (the integer
  position in docs.json). Then Hit@k can be defined the standard, publishable
  way: does top-k contain the gold chunk_id?

  Matching is done two ways and cross-checked:
    1. Normalized-substring containment  — the stub's text appears inside an
       index chunk (whitespace/case-normalized). Authoritative when unique.
    2. Embedding nearest-neighbor (MiniLM, same model the index uses) — fallback
       when truncation/encoding noise breaks the substring match.

  Writes nvidia_golden_qa.jsonl back in place with added fields:
    chunk_id        : int   (position in docs.json; -1 if unmappable)
    chunk_id_method : "substring" | "embedding" | "unmapped"
    chunk_id_score  : float (cosine to chosen chunk; 1.0 for clean substring)
    source_chunk_full : str (the FULL index chunk text — replaces the 301-char stub for transparency)

Run from backend_new/:
  python scripts/backfill_golden_chunk_ids.py            # dry-run report
  python scripts/backfill_golden_chunk_ids.py --write    # write back in place
"""
import argparse
import io
import json
import re
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT   = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "data" / "evaluation" / "nvidia_golden_qa.jsonl"
DOCS   = ROOT / "data" / "embeddings" / "docs.json"

# A stub is unmappable below this cosine even after embedding fallback.
MIN_EMBED_SCORE = 0.45


def _norm(s: str) -> str:
    """Whitespace/case-normalize for robust substring matching."""
    return re.sub(r"\s+", " ", s).strip().lower()


def _clean_stub(stub: str) -> str:
    """Drop the trailing ellipsis / replacement char left by [:300] + '…'."""
    return stub.rstrip("…").rstrip("�").rstrip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="Write chunk_ids back into the golden file (default: dry-run).")
    args = ap.parse_args()

    docs = json.loads(DOCS.read_text(encoding="utf-8"))
    norm_docs = [_norm(d) for d in docs]
    golden = [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Loaded {len(golden)} golden QA, {len(docs)} index chunks.\n")

    # Embedding fallback — same model the index uses (DenseRetriever default).
    from sentence_transformers import SentenceTransformer
    import numpy as np
    st = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embs = st.encode(docs, batch_size=32, show_progress_bar=True,
                         normalize_embeddings=True)

    n_sub, n_emb, n_unmapped = 0, 0, 0

    for g in golden:
        stub = _clean_stub(g.get("source_chunk", ""))
        # use a confident interior slice (skip the noisy first/last chars)
        probe = _norm(stub)
        core = probe[20:200] if len(probe) > 60 else probe

        # 1) substring containment — unique match is authoritative
        matches = [i for i, nd in enumerate(norm_docs) if core and core in nd]
        if len(matches) == 1:
            g["chunk_id"] = matches[0]
            g["chunk_id_method"] = "substring"
            g["chunk_id_score"] = 1.0
            g["source_chunk_full"] = docs[matches[0]]
            n_sub += 1
            continue
        if len(matches) > 1:
            # multiple chunks contain the snippet (overlapping windows) — pick the
            # embedding-closest among them for determinism.
            stub_emb = st.encode([stub], normalize_embeddings=True)[0]
            best = max(matches, key=lambda i: float(np.dot(stub_emb, doc_embs[i])))
            g["chunk_id"] = best
            g["chunk_id_method"] = "substring"
            g["chunk_id_score"] = round(float(np.dot(stub_emb, doc_embs[best])), 4)
            g["source_chunk_full"] = docs[best]
            n_sub += 1
            continue

        # 2) embedding nearest-neighbor fallback
        stub_emb = st.encode([stub], normalize_embeddings=True)[0]
        sims = doc_embs @ stub_emb
        best = int(np.argmax(sims))
        score = float(sims[best])
        if score >= MIN_EMBED_SCORE:
            g["chunk_id"] = best
            g["chunk_id_method"] = "embedding"
            g["chunk_id_score"] = round(score, 4)
            g["source_chunk_full"] = docs[best]
            n_emb += 1
        else:
            g["chunk_id"] = -1
            g["chunk_id_method"] = "unmapped"
            g["chunk_id_score"] = round(score, 4)
            g["source_chunk_full"] = ""
            n_unmapped += 1
            print(f"  [UNMAPPED] {g['id']}: best cosine {score:.3f} < {MIN_EMBED_SCORE}")

    print(f"\nMapped: {n_sub} via substring, {n_emb} via embedding, "
          f"{n_unmapped} unmapped (chunk_id=-1).")

    if args.write:
        with open(GOLDEN, "w", encoding="utf-8") as f:
            for g in golden:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
        print(f"  Wrote chunk_ids back -> {GOLDEN}")
    else:
        print("  (dry-run — pass --write to persist)")


if __name__ == "__main__":
    main()
