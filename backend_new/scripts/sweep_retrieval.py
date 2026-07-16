"""
sweep_retrieval.py
------------------
SOTA sweep for Stage-3 retrieval. Offline, no LLM, no API.

For each config (chunk_size, overlap, embedder, query_prefix):
  1. Re-chunk the corpus (.txt) with sentence-window splitter.
  2. Identify GOLD chunk(s) per question = chunks whose word-overlap with the
     question's anchor paragraph (gold_anchors.json) >= GOLD_OVERLAP. Anchor is
     chunking-independent → fair across configs, zero embedding circularity.
  3. Embed chunks + questions (cosine via normalized + inner product).
  4. Hit@1 / Hit@3 / MRR by gold-chunk containment in top-k.

Dense-only by default (fast, isolates the embedder/chunking effect). Pass
--rerank to add the cross-encoder on the best dense candidates.

Run from backend_new/:
  python scripts/sweep_retrieval.py
  python scripts/sweep_retrieval.py --rerank
"""
import argparse, io, json, re, sys, time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT    = Path(__file__).resolve().parent.parent
CORPUS  = ROOT / "data" / "corpus"
ANCHORS = ROOT / "data" / "evaluation" / "gold_anchors.json"
OUT     = ROOT / "data" / "evaluation" / "stage3_rag" / "sweep_results.json"

TOP_K        = 3
GOLD_OVERLAP = 0.50   # chunk is "gold" if it covers >=50% of anchor's content words

# (chunk_words, overlap_words) grid
CHUNK_GRID = [(128, 32), (192, 48), (256, 64), (384, 64), (512, 50)]

# embedder -> (model_id, query_prefix, doc_prefix)
EMBEDDERS = {
    "minilm-l6":  ("sentence-transformers/all-MiniLM-L6-v2", "", ""),
    "mpnet-base": ("sentence-transformers/all-mpnet-base-v2", "", ""),
    "bge-small":  ("BAAI/bge-small-en-v1.5",
                   "Represent this sentence for searching relevant passages: ", ""),
    "bge-base":   ("BAAI/bge-base-en-v1.5",
                   "Represent this sentence for searching relevant passages: ", ""),
}

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("the a an of to in for and or is are be on at by with as from this that "
            "it its which we can has have was were will".split())


def _norm(s): return re.sub(r"\s+", " ", s).strip().lower()
def _content_words(s): return {w for w in _WORD.findall(s.lower()) if w not in _STOP and len(w) > 2}


def chunk_corpus(chunk_words, overlap_words):
    """Sentence-window chunker (same algorithm as the production ingestion engine)."""
    chunks = []
    for f in sorted(CORPUS.glob("*.txt")):
        text = f.read_text(encoding="utf-8", errors="replace")
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) > 3]
        window, wc = [], 0
        for sent in sents:
            window.append(sent); wc += len(sent.split())
            if wc >= chunk_words:
                c = " ".join(window)
                if len(c.split()) >= 30:
                    chunks.append(c)
                while wc > overlap_words and window:
                    wc -= len(window.pop(0).split())
        if window:
            c = " ".join(window)
            if len(c.split()) >= 30:
                chunks.append(c)
    return chunks


def gold_ids_for(anchor_text, chunks_cw):
    """chunk indices whose content-word overlap with anchor >= GOLD_OVERLAP."""
    a = _content_words(anchor_text)
    if not a:
        return set()
    out = set()
    for i, cw in enumerate(chunks_cw):
        inter = len(a & cw)
        if inter / len(a) >= GOLD_OVERLAP:
            out.add(i)
    return out


def evaluate(chunks, anchors, model, qpref, dpref, rerank_model=None):
    import numpy as np
    chunks_cw = [_content_words(c) for c in chunks]
    gold = {a["id"]: gold_ids_for(a["anchor_text"], chunks_cw) for a in anchors}
    usable = [a for a in anchors if gold[a["id"]]]

    doc_emb = model.encode([dpref + c for c in chunks], batch_size=64,
                           show_progress_bar=False, normalize_embeddings=True)
    q_emb = model.encode([qpref + a["question"] for a in usable], batch_size=64,
                         show_progress_bar=False, normalize_embeddings=True)
    sims = q_emb @ doc_emb.T            # (nq, nchunks) cosine
    pool = max(TOP_K, TOP_K * 6 if rerank_model else TOP_K)
    topk_idx = np.argpartition(-sims, range(min(pool, sims.shape[1])), axis=1)[:, :pool]

    h1 = h3 = rr = 0
    by = {"easy": [0, 0, 0, 0], "medium": [0, 0, 0, 0], "hard": [0, 0, 0, 0]}
    for r, a in enumerate(usable):
        row = topk_idx[r]
        row = row[np.argsort(-sims[r, row])]               # sort pool by cosine
        cand = list(row)
        if rerank_model is not None:
            pairs = [(a["question"], chunks[i]) for i in cand]
            rs = rerank_model.predict(pairs)
            cand = [c for _, c in sorted(zip(rs, cand), key=lambda x: -x[0])]
        topids = cand[:TOP_K]
        g = gold[a["id"]]
        hit1 = int(topids[0] in g)
        hit3 = int(any(t in g for t in topids))
        rrv = 0.0
        for rank, t in enumerate(topids):
            if t in g:
                rrv = 1.0 / (rank + 1); break
        h1 += hit1; h3 += hit3; rr += rrv
        b = by[a["difficulty"]]; b[0] += 1; b[1] += hit1; b[2] += hit3; b[3] += rrv
    n = len(usable)
    res = {"n_gold": n, "Hit@1": round(h1/n, 4), "Hit@3": round(h3/n, 4), "MRR": round(rr/n, 4)}
    res["by_difficulty"] = {k: {"n": v[0], "Hit@1": round(v[1]/v[0], 4),
                                "Hit@3": round(v[2]/v[0], 4), "MRR": round(v[3]/v[0], 4)}
                            for k, v in by.items() if v[0]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--embedders", default="minilm-l6,bge-base",
                    help="comma list from: " + ",".join(EMBEDDERS))
    ap.add_argument("--full", action="store_true", help="all embedders x all chunk sizes")
    args = ap.parse_args()

    anchors = json.loads(ANCHORS.read_text(encoding="utf-8"))
    from sentence_transformers import SentenceTransformer, CrossEncoder
    rer = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") if args.rerank else None

    emb_keys = list(EMBEDDERS) if args.full else [e.strip() for e in args.embedders.split(",")]
    results = []
    t0 = time.time()

    # cache chunkings + models
    chunk_cache = {cfg: chunk_corpus(*cfg) for cfg in CHUNK_GRID}
    for cfg, ch in chunk_cache.items():
        print(f"chunk {cfg}: {len(ch)} chunks")

    for ek in emb_keys:
        mid, qp, dp = EMBEDDERS[ek]
        print(f"\n=== embedder {ek} ({mid}) ===")
        model = SentenceTransformer(mid)
        for cfg in CHUNK_GRID:
            r = evaluate(chunk_cache[cfg], anchors, model, qp, dp, rer)
            tag = f"{ek} cw={cfg[0]} ov={cfg[1]}{' +rerank' if rer else ''}"
            r.update({"embedder": ek, "chunk_words": cfg[0], "overlap": cfg[1],
                      "rerank": bool(rer), "tag": tag})
            results.append(r)
            print(f"  {tag:<34} n={r['n_gold']:<3} "
                  f"Hit@1={r['Hit@1']:.3f} Hit@3={r['Hit@3']:.3f} MRR={r['MRR']:.3f} "
                  f"({time.time()-t0:.0f}s)")

    results.sort(key=lambda x: (x["Hit@3"], x["Hit@1"], x["MRR"]), reverse=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + "=" * 70)
    print("TOP 8 CONFIGS (by Hit@3):")
    for r in results[:8]:
        print(f"  {r['tag']:<36} Hit@1={r['Hit@1']:.3f} Hit@3={r['Hit@3']:.3f} MRR={r['MRR']:.3f}")
    print(f"\nsaved -> {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
