"""
add_smallsignal_equations.py
----------------------------
Append curated small-signal / output-resistance equations to the retrieval
index so design/sweep queries like "plot ro vs Id" can ground on an equation
that actually contains those variables. The base corpus has prose about output
resistance but no *extractable* `ro = 1/(lam*Id)` equation.

Idempotent: skips chunks already present. Updates docs.json, docs.pkl,
dense.index (FAISS append), and bm25_docs.json — mirroring IngestionEngine.

Run from backend_new/:
    python scripts/add_smallsignal_equations.py
"""
import os
import re
import json
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # backend_new
os.chdir(ROOT)
HF = ROOT.parent / "hf_cache"
os.environ["HF_HOME"] = str(HF)
if (HF / "hub").exists():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMB         = ROOT / "data" / "embeddings"
DOCS_JSON   = EMB / "docs.json"
DOCS_PKL    = EMB / "docs.pkl"
DENSE_INDEX = EMB / "dense.index"
BM25_JSON   = EMB / "bm25_docs.json"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Each equation sits on its own line so EquationValidator extracts it cleanly.
# Symbols use the validator's vocabulary (ro, Id, lam, gm, mu, Cox, WL, Vov,
# Av, Vds). Prose lines (no "=") add retrieval keywords without being parsed.
NEW_CHUNKS = [
    (
        "MOSFET small-signal parameters in saturation: output resistance, "
        "transconductance, and intrinsic voltage gain. The small-signal output "
        "resistance ro due to channel-length modulation, plotted against the "
        "drain current Id, follows\n"
        "ro = 1 / (lam * Id)\n"
        "The transconductance gm relating drain current to gate overdrive "
        "voltage Vov is\n"
        "gm = mu * Cox * WL * Vov\n"
        "An equivalent transconductance expression in terms of the drain "
        "current Id and the overdrive voltage Vov is\n"
        "gm = 2 * Id / Vov\n"
        "The intrinsic voltage gain Av of a single common-source MOSFET stage "
        "is the product of transconductance and output resistance,\n"
        "Av = gm * ro\n"
        "The saturation drain current including channel-length modulation, as a "
        "function of the drain-source voltage Vds, is\n"
        "Id = 0.5 * mu * Cox * WL * Vov**2 * (1 + lam * Vds)"
    ),
]


def main():
    docs = json.loads(DOCS_JSON.read_text(encoding="utf-8"))
    print(f"Existing chunks: {len(docs)}")

    # Idempotency: skip any chunk whose signature equation already exists.
    existing_blob = "\n".join(docs)
    to_add = [c for c in NEW_CHUNKS if "ro = 1 / (lam * Id)" not in existing_blob
              and c not in docs]
    if not to_add:
        print("Small-signal equations already present — nothing to do.")
        return

    print(f"Adding {len(to_add)} chunk(s)...")
    model = SentenceTransformer(EMBED_MODEL)
    emb = model.encode(to_add, convert_to_numpy=True, show_progress_bar=False).astype("float32")

    index = faiss.read_index(str(DENSE_INDEX))
    assert index.d == emb.shape[1], f"dim mismatch {index.d} vs {emb.shape[1]}"
    before = index.ntotal
    index.add(emb)
    faiss.write_index(index, str(DENSE_INDEX))
    print(f"FAISS vectors: {before} -> {index.ntotal}")

    docs.extend(to_add)
    DOCS_JSON.write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
    with open(DOCS_PKL, "wb") as f:
        pickle.dump(docs, f)

    # Rebuild BM25 token store (matches IngestionEngine: \w+ lowercased)
    tokenised = [re.findall(r"\w+", d.lower()) for d in docs]
    BM25_JSON.write_text(json.dumps(tokenised, ensure_ascii=False), encoding="utf-8")

    print(f"docs.json / docs.pkl / bm25_docs.json updated. Total chunks: {len(docs)}")


if __name__ == "__main__":
    main()
