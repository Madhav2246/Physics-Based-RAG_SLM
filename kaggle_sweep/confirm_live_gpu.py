"""
confirm_live_gpu.py — confirm Hit@k on the REBUILT index, full hybrid + rerank.
==============================================================================
Runs the exact deployed retrieval path against the rebuilt bge-large index:
  dense (bge-large / cosine IndexFlatIP) + BM25 sparse + RRF fusion + cross-encoder
  rerank, top-3 — scored on the chunking-independent gold anchors.

This is the strongest Stage-3 number for the paper: live pipeline, not dense-only.

PREREQ: run rebuild_index_gpu.py first (creates ./rebuilt_index/).
Reads:
  rebuilt_index/docs.json, dense.index, bm25_docs.json, build_meta.json
  data/evaluation/gold_anchors.json

RUN ON KAGGLE (GPU P100, Internet ON):
  !pip install -q -r requirements.txt
  !python rebuild_index_gpu.py        # if not already done
  !python confirm_live_gpu.py

Prints Hit@1 / Hit@3 / MRR overall + by difficulty. Send me the numbers.
"""
import io, json, re, sys, time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT    = Path(__file__).resolve().parent
IDX     = ROOT / "rebuilt_index"
ANCHORS = ROOT / "data" / "evaluation" / "gold_anchors.json"
OUT     = ROOT / "rebuilt_index" / "stage3_confirm.json"

TOP_K        = 3
GOLD_OVERLAP = 0.50
GOLD_FLOOR   = 0.25
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("the a an of to in for and or is are be on at by with as from this that "
            "it its which we can has have was were will".split())
_TOKEN_RE = re.compile(r'[A-Za-z][A-Za-z0-9_]*|[\d]+(?:\.\d+)?(?:[eE][+-]?\d+)?')


def _cw(s):
    return {w for w in _WORD.findall(s.lower()) if w not in _STOP and len(w) > 2}


def _norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def _avg(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    import numpy as np, torch, faiss
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from rank_bm25 import BM25Okapi

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    meta = json.loads((IDX / "build_meta.json").read_text(encoding="utf-8"))
    docs = json.loads((IDX / "docs.json").read_text(encoding="utf-8"))
    bm25_docs = json.loads((IDX / "bm25_docs.json").read_text(encoding="utf-8"))
    index = faiss.read_index(str(IDX / "dense.index"))
    anchors = json.loads(ANCHORS.read_text(encoding="utf-8"))
    print(f"device={device} | {len(docs)} chunks | embedder={meta['embed_model']} "
          f"| metric={meta['metric']}")

    model = SentenceTransformer(meta["embed_model"], device=device)
    bm25 = BM25Okapi(bm25_docs)
    reranker = CrossEncoder(RERANK_MODEL, device=device)
    qprefix = meta.get("query_prefix", "")

    anchor_cw = {a["id"]: _cw(a["anchor_text"]) for a in anchors}

    # Pre-embed all queries once (cosine -> normalized).
    q_texts = [qprefix + a["question"] for a in anchors]
    q_embs = model.encode(q_texts, batch_size=64, normalize_embeddings=True,
                          convert_to_numpy=True).astype("float32")

    per_q = []
    for i, a in enumerate(anchors):
        acw = anchor_cw[a["id"]]
        if not acw:
            continue
        # dense top (cosine via IP)
        D, I = index.search(q_embs[i:i+1], TOP_K * 4)
        dense_hits = [docs[j] for j in I[0] if 0 <= j < len(docs)]
        # sparse top
        qtok = _TOKEN_RE.findall(a["question"])
        sscores = bm25.get_scores(qtok)
        sidx = np.argsort(-sscores)[:TOP_K * 4]
        sparse_hits = [docs[j] for j in sidx]
        # RRF fusion (content-hashed keys)
        scores = {}
        for hits in (dense_hits, sparse_hits):
            for rank, d in enumerate(hits):
                key = _norm(d)
                scores.setdefault(key, {"doc": d, "s": 0.0})
                scores[key]["s"] += 1.0 / (60 + rank + 1)
        fused = sorted(scores.values(), key=lambda x: x["s"], reverse=True)
        cand = [f["doc"] for f in fused[:TOP_K * 4]]
        # rerank
        rs = reranker.predict([(a["question"], c) for c in cand], batch_size=64)
        reranked = [c for _, c in sorted(zip(rs, cand), key=lambda x: -x[0])][:TOP_K]

        ovs = [len(acw & _cw(c)) / len(acw) for c in reranked]
        hits = [o >= GOLD_OVERLAP for o in ovs]
        if not any(hits) and ovs and max(ovs) >= GOLD_FLOOR:
            hits[ovs.index(max(ovs))] = True
        hit1 = int(hits[0]) if hits else 0
        hit3 = int(any(hits))
        rr = next((1.0/(r+1) for r, h in enumerate(hits) if h), 0.0)
        per_q.append({"id": a["id"], "difficulty": a["difficulty"],
                      "hit1": hit1, "hit3": hit3, "rr": round(rr, 4)})

    def agg(rows):
        return {"n": len(rows), "Hit@1": round(_avg([p["hit1"] for p in rows]), 4),
                "Hit@3": round(_avg([p["hit3"] for p in rows]), 4),
                "MRR": round(_avg([p["rr"] for p in rows]), 4)}

    overall = agg(per_q)
    by_diff = {d: agg([p for p in per_q if p["difficulty"] == d])
               for d in ["easy", "medium", "hard"]
               if any(p["difficulty"] == d for p in per_q)}

    OUT.write_text(json.dumps({"embedder": meta["embed_model"], "n_chunks": len(docs),
                               "pipeline": "dense+sparse+RRF+rerank",
                               "overall": overall, "by_difficulty": by_diff,
                               "per_question": per_q}, indent=2), encoding="utf-8")

    SEP = "-" * 60
    print(f"\n{SEP}\n  STAGE 3 CONFIRM — LIVE HYBRID+RERANK ({meta['embed_model']})\n{SEP}")
    print(f"  Hit@1 : {overall['Hit@1']:.4f}  ({overall['Hit@1']*100:.1f}%)")
    print(f"  Hit@3 : {overall['Hit@3']:.4f}  ({overall['Hit@3']*100:.1f}%)")
    print(f"  MRR   : {overall['MRR']:.4f}")
    print(SEP)
    for d, v in by_diff.items():
        print(f"  {d:<8} n={v['n']:<3} Hit@1={v['Hit@1']:.3f} Hit@3={v['Hit@3']:.3f} MRR={v['MRR']:.3f}")
    print(f"{SEP}\n  saved -> {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
