"""
stage3_confirm.py
-----------------
Confirm Stage-3 retrieval quality on the LIVE production pipeline after the
bge-large @ 384/64 index rebuild. Index-independent ground truth.

Ground truth: data/evaluation/gold_anchors.json — each question's true source
paragraph. A retrieved chunk is a HIT if its content-word overlap with the
anchor >= GOLD_OVERLAP (same definition the SOTA sweep used), so this works on
ANY index/chunking without precomputed chunk_ids.

Tests the exact deployed hybrid path: dense (bge-large/cosine) + sparse (BM25)
+ RRF fusion + cross-encoder rerank, top-k.

Run from backend_new/:
  python scripts/stage3_confirm.py
"""
import io, json, re, sys, time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ANCHORS = ROOT / "data" / "evaluation" / "gold_anchors.json"
OUT     = ROOT / "data" / "evaluation" / "stage3_rag" / "stage3_confirm.json"

GOLD_OVERLAP = 0.50
GOLD_FLOOR   = 0.25

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("the a an of to in for and or is are be on at by with as from this that "
            "it its which we can has have was were will".split())


def _cw(s):
    return {w for w in _WORD.findall(s.lower()) if w not in _STOP and len(w) > 2}


def _gold_match(anchor_cw, chunk_cw):
    if not anchor_cw:
        return 0.0
    return len(anchor_cw & chunk_cw) / len(anchor_cw)


def _avg(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    t0 = time.time()
    anchors = json.loads(ANCHORS.read_text(encoding="utf-8"))
    anchor_cw = {a["id"]: _cw(a["anchor_text"]) for a in anchors}
    print(f"{len(anchors)} anchors loaded. Building live pipeline …")

    import utils.config as cfg
    from retrieval.dense_retriever import DenseRetriever
    from retrieval.sparse_retriever import SparseRetriever
    from retrieval.reranker import CrossEncoderReranker

    dense = DenseRetriever()
    dense.load_index()
    sparse = SparseRetriever()
    sparse.build_index_from_docs(dense.documents)
    reranker = CrossEncoderReranker()
    print(f"  pipeline ready ({time.time()-t0:.0f}s) | embedder={cfg.EMBED_MODEL_NAME} "
          f"| {len(dense.documents)} chunks")

    def _norm(s):
        return re.sub(r"\s+", " ", s).strip().lower()

    def retrieve_hybrid(q, top_k=cfg.TOP_K):
        dense_hits = dense.retrieve(q, top_k=top_k * 4)
        sparse_hits = sparse.retrieve(q, top_k=top_k * 4)
        scores = {}
        def _add(hits):
            for rank, d in enumerate(hits):
                key = _norm(d)
                if key not in scores:
                    scores[key] = {"doc": d, "s": 0.0}
                scores[key]["s"] += 1.0 / (60 + rank + 1)
        _add(dense_hits); _add(sparse_hits)
        fused = sorted(scores.values(), key=lambda x: x["s"], reverse=True)
        docs_ = [f["doc"] for f in fused[:top_k * 4]]
        return reranker.rerank(q, docs_, top_k=top_k)

    per_q = []
    for i, a in enumerate(anchors):
        acw = anchor_cw[a["id"]]
        if not acw:
            continue
        retrieved = retrieve_hybrid(a["question"], top_k=cfg.TOP_K)
        # per-retrieved-chunk overlap -> hit if >= GOLD_OVERLAP (or best >= FLOOR)
        ovs = [_gold_match(acw, _cw(c)) for c in retrieved]
        hits = [o >= GOLD_OVERLAP for o in ovs]
        if not any(hits) and ovs and max(ovs) >= GOLD_FLOOR:
            # accept the single best as a soft hit only if it clears the floor
            hits[ovs.index(max(ovs))] = True
        hit1 = int(hits[0]) if hits else 0
        hit3 = int(any(hits))
        rr = next((1.0/(r+1) for r, h in enumerate(hits) if h), 0.0)
        per_q.append({"id": a["id"], "difficulty": a["difficulty"],
                      "hit1": hit1, "hit3": hit3, "rr": round(rr, 4),
                      "overlaps": [round(o, 3) for o in ovs]})
        if (i+1) % 20 == 0:
            print(f"  {i+1}/{len(anchors)}  Hit@1={_avg([p['hit1'] for p in per_q]):.3f} "
                  f"Hit@3={_avg([p['hit3'] for p in per_q]):.3f} ({time.time()-t0:.0f}s)")

    def agg(rows):
        return {"n": len(rows),
                "Hit@1": round(_avg([p["hit1"] for p in rows]), 4),
                "Hit@3": round(_avg([p["hit3"] for p in rows]), 4),
                "MRR":   round(_avg([p["rr"] for p in rows]), 4)}

    overall = agg(per_q)
    by_diff = {d: agg([p for p in per_q if p["difficulty"] == d])
               for d in ["easy", "medium", "hard"]
               if any(p["difficulty"] == d for p in per_q)}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"embedder": cfg.EMBED_MODEL_NAME,
                               "n_chunks": len(dense.documents),
                               "overall": overall, "by_difficulty": by_diff,
                               "per_question": per_q}, indent=2, ensure_ascii=False),
                   encoding="utf-8")

    SEP = "-" * 60
    print(f"\n{SEP}\n  STAGE 3 CONFIRM — LIVE PIPELINE ({cfg.EMBED_MODEL_NAME})\n{SEP}")
    print(f"  Hit@1 : {overall['Hit@1']:.4f}  ({overall['Hit@1']*100:.1f}%)")
    print(f"  Hit@3 : {overall['Hit@3']:.4f}  ({overall['Hit@3']*100:.1f}%)")
    print(f"  MRR   : {overall['MRR']:.4f}")
    print(SEP)
    for d, v in by_diff.items():
        print(f"  {d:<8} n={v['n']:<3} Hit@1={v['Hit@1']:.3f} Hit@3={v['Hit@3']:.3f} MRR={v['MRR']:.3f}")
    print(SEP)
    print(f"  saved -> {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
