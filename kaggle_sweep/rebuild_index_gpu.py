"""
rebuild_index_gpu.py — rebuild the production retrieval index with the SOTA config.
==================================================================================
Winner of the Stage-3 sweep:  embedder = BAAI/bge-large-en-v1.5
                              chunk = 384 words, overlap = 64 words
                              Hit@3 = 0.65  (vs MiniLM@512 baseline 0.52, +25% rel)

Reads:  data/corpus/*.txt
Writes (into ./rebuilt_index/, drop these into backend_new/data/embeddings/):
  docs.json        list[str] of chunks (positional id = FAISS row)
  dense.index      FAISS IndexFlatIP over L2-normalized bge-large embeddings (cosine)
  bm25_docs.json   tokenized chunks for the sparse retriever
  registry.json    {single corpus entry, chunk_count}
  build_meta.json  embedder / chunk params / dim — so the backend knows what to load

RUN ON KAGGLE (GPU P100, Internet ON):
  !pip install -q -r requirements.txt
  !python rebuild_index_gpu.py
Then download the rebuilt_index/ folder and replace backend_new/data/embeddings/*.

NOTE: bge-large is 1024-dim (MiniLM was 384). The backend must load bge-large at
query time too — see the printed CONFIG PATCH at the end (apply to
utils/config.py + retrieval/dense_retriever.py + _ingestion_engine.py).
"""
import io, json, re, sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT   = Path(__file__).resolve().parent
CORPUS = ROOT / "data" / "corpus"
OUT    = ROOT / "rebuilt_index"

EMBED_MODEL  = "BAAI/bge-large-en-v1.5"
EMBED_DIM    = 1024
DOC_PREFIX   = ""     # bge-large: no prefix needed on documents
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CHUNK_WORDS   = 384
OVERLAP_WORDS = 64

_TOKEN_RE = re.compile(r'[A-Za-z][A-Za-z0-9_]*|[\d]+(?:\.\d+)?(?:[eE][+-]?\d+)?')


def chunk_text(text):
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) > 3]
    chunks, window, wc = [], [], 0
    for sent in sents:
        window.append(sent); wc += len(sent.split())
        if wc >= CHUNK_WORDS:
            c = " ".join(window)
            if len(c.split()) >= 30:
                chunks.append(c)
            while wc > OVERLAP_WORDS and window:
                wc -= len(window.pop(0).split())
    if window:
        c = " ".join(window)
        if len(c.split()) >= 30:
            chunks.append(c)
    return chunks


def main():
    import numpy as np, faiss, torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  embedder={EMBED_MODEL}  chunk={CHUNK_WORDS}/{OVERLAP_WORDS}")

    # 1) chunk the whole corpus
    all_chunks = []
    for f in sorted(CORPUS.glob("*.txt")):
        all_chunks.extend(chunk_text(f.read_text(encoding="utf-8", errors="replace")))
    print(f"chunked {len(list(CORPUS.glob('*.txt')))} files -> {len(all_chunks)} chunks")

    # 2) embed (cosine via normalized + inner-product index)
    model = SentenceTransformer(EMBED_MODEL, device=device)
    bs = 32 if device == "cuda" else 8
    embs = model.encode([DOC_PREFIX + c for c in all_chunks], batch_size=bs,
                        show_progress_bar=True, normalize_embeddings=True,
                        convert_to_numpy=True)
    embs = embs.astype("float32")
    assert embs.shape[1] == EMBED_DIM, f"dim {embs.shape[1]} != {EMBED_DIM}"

    index = faiss.IndexFlatIP(EMBED_DIM)   # IP on normalized vectors == cosine
    index.add(embs)

    # 3) write artifacts
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "docs.json").write_text(json.dumps(all_chunks, ensure_ascii=False),
                                   encoding="utf-8")
    faiss.write_index(index, str(OUT / "dense.index"))
    bm25 = [_TOKEN_RE.findall(c) for c in all_chunks]
    (OUT / "bm25_docs.json").write_text(json.dumps(bm25, ensure_ascii=False),
                                        encoding="utf-8")
    (OUT / "registry.json").write_text(json.dumps(
        {"corpus_txt": {"original_name": "data/corpus/*.txt",
                        "chunk_count": len(all_chunks), "chunk_start": 0}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "build_meta.json").write_text(json.dumps({
        "embed_model": EMBED_MODEL, "embed_dim": EMBED_DIM,
        "query_prefix": QUERY_PREFIX, "doc_prefix": DOC_PREFIX,
        "chunk_words": CHUNK_WORDS, "overlap_words": OVERLAP_WORDS,
        "metric": "cosine (IndexFlatIP on normalized)", "n_chunks": len(all_chunks),
    }, indent=2), encoding="utf-8")

    print(f"\nWrote -> {OUT}/  (docs.json, dense.index, bm25_docs.json, registry.json, build_meta.json)")
    print("=" * 72)
    print("CONFIG PATCH — apply to the backend so it LOADS this index correctly:")
    print("-" * 72)
    print(f"""utils/config.py:
    EMBED_MODEL_NAME = "{EMBED_MODEL}"        # was all-MiniLM-L6-v2
    EMBED_DIM        = {EMBED_DIM}                       # was 384
    QUERY_PREFIX     = "{QUERY_PREFIX}"

retrieval/dense_retriever.py:
    - load EMBED_MODEL_NAME from config (not hardcoded "all-MiniLM-L6-v2")
    - normalize_embeddings=True when encoding queries
    - prepend QUERY_PREFIX to the query before encoding
    - faiss.IndexFlatIP (cosine) — already baked into dense.index here

scripts/_ingestion_engine.py (for FUTURE PDF ingests to match):
    EMBED_MODEL_NAME = "{EMBED_MODEL}"
    EMBED_DIM        = {EMBED_DIM}
    CHUNK_WORDS      = {CHUNK_WORDS}
    OVERLAP_WORDS    = {OVERLAP_WORDS}
""")


if __name__ == "__main__":
    main()
