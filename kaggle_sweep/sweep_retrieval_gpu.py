"""
sweep_retrieval_gpu.py  — Kaggle P100 GPU retrieval sweep (SOTA hunt)
=====================================================================
Goal: find the chunking + embedder config that maximizes Hit@1 / Hit@3 / MRR
for the Stage-3 RAG retrieval, then add a reranker on the winner.

Self-contained. No SLM, no API, no LLM. Reads only:
  data/corpus/*.txt              (the physics corpus)
  data/evaluation/gold_anchors.json  (chunking-independent ground truth: each
                                      question -> its true source paragraph)

GROUND TRUTH (fair across chunkings)
  Anchor = the raw source paragraph each question was authored from. For any
  chunking, a chunk is GOLD if its content-word overlap with the anchor >= 0.50.
  This removes embedding circularity (we don't use any embedder to pick gold) and
  lets us compare chunk sizes apples-to-apples.

OUTPUT
  results/sweep_results.json     full grid, sorted by Hit@3
  prints a leaderboard

RUN ON KAGGLE
  - Notebook settings: Accelerator = GPU P100, Internet = ON (to download embedders)
  - Upload this zip, unzip, then:
        !pip install -q -r requirements.txt
        !python sweep_retrieval_gpu.py --full
  - For just the strong/fast subset (default):
        !python sweep_retrieval_gpu.py
  - Add reranker on the winning dense config:
        !python sweep_retrieval_gpu.py --full --rerank

Send back results/sweep_results.json (or paste the leaderboard).
"""
import argparse, json, re, time
from pathlib import Path

ROOT    = Path(__file__).resolve().parent
CORPUS  = ROOT / "data" / "corpus"
ANCHORS = ROOT / "data" / "evaluation" / "gold_anchors.json"
OUT     = ROOT / "results" / "sweep_results.json"

TOP_K        = 3
# A chunk is GOLD if it covers >= GOLD_OVERLAP of the anchor's content words.
# We ALSO always include the single best-overlap chunk (if it clears GOLD_FLOOR),
# so every question has >=1 gold target regardless of chunk size — otherwise small
# chunks (which split an anchor across several chunks) would unfairly show 0 gold
# and depress Hit@k. This keeps n=100 comparable across all chunk granularities.
GOLD_OVERLAP = 0.50
GOLD_FLOOR   = 0.25

# (chunk_words, overlap_words)
CHUNK_GRID = [(96, 24), (128, 32), (160, 40), (192, 48), (256, 64), (384, 64), (512, 50)]

# key -> (hf_id, query_prefix, doc_prefix)
# E5 models REQUIRE "query:"/"passage:" prefixes. BGE uses an instruction on queries.
EMBEDDERS = {
    "minilm-l6":   ("sentence-transformers/all-MiniLM-L6-v2", "", ""),
    "mpnet-base":  ("sentence-transformers/all-mpnet-base-v2", "", ""),
    "bge-small":   ("BAAI/bge-small-en-v1.5",
                    "Represent this sentence for searching relevant passages: ", ""),
    "bge-base":    ("BAAI/bge-base-en-v1.5",
                    "Represent this sentence for searching relevant passages: ", ""),
    "bge-large":   ("BAAI/bge-large-en-v1.5",
                    "Represent this sentence for searching relevant passages: ", ""),
    "e5-base":     ("intfloat/e5-base-v2", "query: ", "passage: "),
    "e5-large":    ("intfloat/e5-large-v2", "query: ", "passage: "),
    "gte-large":   ("thenlper/gte-large", "", ""),
    "bge-m3":      ("BAAI/bge-m3", "", ""),
}

# Fast/strong default subset (skip the slowest unless --full)
DEFAULT_SUBSET = ["minilm-l6", "bge-base", "e5-base", "gte-large"]

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("the a an of to in for and or is are be on at by with as from this that "
            "it its which we can has have was were will".split())


def _content_words(s):
    return {w for w in _WORD.findall(s.lower()) if w not in _STOP and len(w) > 2}


def chunk_corpus(chunk_words, overlap_words):
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
    a = _content_words(anchor_text)
    if not a:
        return set()
    ov = [(len(a & cw) / len(a), i) for i, cw in enumerate(chunks_cw)]
    gold = {i for frac, i in ov if frac >= GOLD_OVERLAP}
    if not gold:                       # small-chunk case: take best chunk if usable
        frac, i = max(ov)
        if frac >= GOLD_FLOOR:
            gold.add(i)
    return gold


def _encode_safe(model, texts, device, batch_size=128):
    """Encode with OOM-retry: halve batch on CUDA OOM until it fits (or CPU)."""
    import torch
    bs = batch_size
    while True:
        try:
            return model.encode(texts, batch_size=bs, show_progress_bar=False,
                                 normalize_embeddings=True, convert_to_numpy=True,
                                 device=device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs <= 8:
                print(f"    [OOM] batch={bs} still too big — falling back to CPU")
                return model.encode(texts, batch_size=8, show_progress_bar=False,
                                    normalize_embeddings=True, convert_to_numpy=True,
                                    device="cpu")
            bs //= 2
            print(f"    [OOM] retrying with batch_size={bs}")


def evaluate(chunks, anchors, model, qp, dp, device, rerank_model=None, batch_size=128):
    import numpy as np, torch
    chunks_cw = [_content_words(c) for c in chunks]
    gold = {a["id"]: gold_ids_for(a["anchor_text"], chunks_cw) for a in anchors}
    usable = [a for a in anchors if gold[a["id"]]]

    doc_emb = _encode_safe(model, [dp + c for c in chunks], device, batch_size)
    q_emb = _encode_safe(model, [qp + a["question"] for a in usable], device, batch_size)
    sims = q_emb @ doc_emb.T
    pool = TOP_K * 8 if rerank_model else TOP_K
    pool = min(pool, sims.shape[1])

    h1 = h3 = rr = 0
    by = {"easy": [0, 0, 0, 0], "medium": [0, 0, 0, 0], "hard": [0, 0, 0, 0]}
    for r, a in enumerate(usable):
        order = np.argsort(-sims[r])[:pool]
        cand = list(order)
        if rerank_model is not None:
            pairs = [(a["question"], chunks[i]) for i in cand]
            rs = rerank_model.predict(pairs, batch_size=128)
            cand = [c for _, c in sorted(zip(rs, cand), key=lambda x: -x[0])]
        topids = cand[:TOP_K]
        g = gold[a["id"]]
        hit1 = int(topids[0] in g)
        hit3 = int(any(t in g for t in topids))
        rrv = next((1.0 / (k + 1) for k, t in enumerate(topids) if t in g), 0.0)
        h1 += hit1; h3 += hit3; rr += rrv
        b = by[a["difficulty"]]; b[0] += 1; b[1] += hit1; b[2] += hit3; b[3] += rrv
    n = len(usable)
    return {
        "n_gold": n, "n_chunks": len(chunks),
        "Hit@1": round(h1/n, 4), "Hit@3": round(h3/n, 4), "MRR": round(rr/n, 4),
        "by_difficulty": {k: {"n": v[0], "Hit@1": round(v[1]/v[0], 4),
                              "Hit@3": round(v[2]/v[0], 4), "MRR": round(v[3]/v[0], 4)}
                          for k, v in by.items() if v[0]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="all embedders (incl. large)")
    ap.add_argument("--rerank", action="store_true", help="add cross-encoder on winner")
    ap.add_argument("--embedders", default="", help="comma list override")
    args = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer, CrossEncoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}", torch.cuda.get_device_name(0) if device == "cuda" else "")

    anchors = json.loads(ANCHORS.read_text(encoding="utf-8"))
    print(f"{len(anchors)} gold anchors")

    if args.embedders:
        emb_keys = [e.strip() for e in args.embedders.split(",")]
    else:
        emb_keys = list(EMBEDDERS) if args.full else DEFAULT_SUBSET

    chunk_cache = {cfg: chunk_corpus(*cfg) for cfg in CHUNK_GRID}
    for cfg, ch in chunk_cache.items():
        print(f"  chunk {cfg}: {len(ch)} chunks")

    results = []
    t0 = time.time()
    for ek in emb_keys:
        mid, qp, dp = EMBEDDERS[ek]
        print(f"\n=== {ek} ({mid}) ===")
        try:
            model = SentenceTransformer(mid, device=device)
        except Exception as e:
            print(f"  [SKIP] load failed: {e}"); continue
        # Big models (large / m3) get a smaller batch up front to avoid OOM.
        bs = 32 if any(t in ek for t in ("large", "m3")) else 128
        for cfg in CHUNK_GRID:
            r = evaluate(chunk_cache[cfg], anchors, model, qp, dp, device, batch_size=bs)
            r.update({"embedder": ek, "chunk_words": cfg[0], "overlap": cfg[1],
                      "rerank": False,
                      "tag": f"{ek} cw={cfg[0]} ov={cfg[1]}"})
            results.append(r)
            print(f"  cw={cfg[0]:<3} ov={cfg[1]:<3}  Hit@1={r['Hit@1']:.3f} "
                  f"Hit@3={r['Hit@3']:.3f} MRR={r['MRR']:.3f}  ({time.time()-t0:.0f}s)")
        del model
        torch.cuda.empty_cache() if device == "cuda" else None

    # Rerank pass on the single best dense config
    if args.rerank and results:
        best = max(results, key=lambda x: (x["Hit@3"], x["Hit@1"], x["MRR"]))
        print(f"\n=== RERANK on winner: {best['tag']} ===")
        mid, qp, dp = EMBEDDERS[best["embedder"]]
        model = SentenceTransformer(mid, device=device)
        rer = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
        cfg = (best["chunk_words"], best["overlap"])
        bs = 32 if any(t in best["embedder"] for t in ("large", "m3")) else 128
        r = evaluate(chunk_cache[cfg], anchors, model, qp, dp, device,
                     rerank_model=rer, batch_size=bs)
        r.update({"embedder": best["embedder"], "chunk_words": cfg[0], "overlap": cfg[1],
                  "rerank": True, "tag": f"{best['embedder']} cw={cfg[0]} ov={cfg[1]} +rerank"})
        results.append(r)
        print(f"  +rerank  Hit@1={r['Hit@1']:.3f} Hit@3={r['Hit@3']:.3f} MRR={r['MRR']:.3f}")

    results.sort(key=lambda x: (x["Hit@3"], x["Hit@1"], x["MRR"]), reverse=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 72)
    print("LEADERBOARD (by Hit@3):")
    print(f"  {'config':<34}{'Hit@1':>8}{'Hit@3':>8}{'MRR':>8}")
    for r in results[:15]:
        print(f"  {r['tag']:<34}{r['Hit@1']:>8.3f}{r['Hit@3']:>8.3f}{r['MRR']:>8.3f}")
    print(f"\nsaved -> {OUT}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
