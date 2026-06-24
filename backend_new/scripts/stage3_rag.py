"""
stage3_rag.py
-------------
Stage 3: RAG Retrieval Quality.

Metrics (publishable, chunk-ID based):
  Hit@1, Hit@3   — does top-k contain the GOLD chunk_id? (exact ID containment)
  MRR            — mean reciprocal rank of the gold chunk_id within top-k
  NDCG@3         — graded gain; gold chunk = rel 1.0, others graded by cosine to gold
  Context Relevancy   — cosine_sim(question, top-1 retrieved chunk)
  Faithfulness (full) — cosine_sim(SYS_answer, concatenated retrieved evidence)

GROUND TRUTH
  nvidia_golden_qa.jsonl `chunk_id` field — the integer position in
  data/embeddings/docs.json of the index chunk the QA was generated from.
  (Backfilled by scripts/backfill_golden_chunk_ids.py.)

WHY THIS CHANGED (2026-06-07)
  The previous version defined a "hit" as cosine_sim(source_chunk, retrieved) >= 0.70,
  where source_chunk was a 301-char TRUNCATED prefix of a blank-line paragraph that
  did not exist in the 512-word-window retrieval index. That metric measured a
  length/threshold artifact, not retrieval — it reported Hit@1=0.16 while NDCG@3=0.96,
  a self-contradiction. Hit@k is now exact gold-chunk-ID containment, the standard
  definition used in retrieval papers (DPR, BEIR, etc.).

  We report two cohorts:
    full      — all questions with a mapped chunk_id (chunk_id >= 0)
    hiconf    — questions whose gold chunk_id was assigned by exact substring match
                OR embedding match with score >= HICONF_EMBED. This subset removes
                the mild circularity of using MiniLM to BOTH pick the gold chunk and
                drive the retriever, so reviewers can trust it independently.

Runs on CPU — no model generation needed.
Run from backend_new/:
  python scripts/stage3_rag.py
"""
import io, json, math, sys, time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GOLDEN   = ROOT / "data" / "evaluation" / "nvidia_golden_qa.jsonl"
DUMP     = ROOT / "data" / "evaluation" / "answers_dump.jsonl"
DOCS     = ROOT / "data" / "embeddings" / "docs.json"
OUT_DIR  = ROOT / "data" / "evaluation" / "stage3_rag"
OUT_JSON = OUT_DIR / "stage3_rag.json"

HICONF_EMBED = 0.70   # embedding-mapped gold chunks at/above this are "high-confidence"

# ---------------------------------------------------------------------------
def _avg(xs):
    return sum(xs) / len(xs) if xs else 0.0

def _dcg(rels):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))

def _ndcg(rels, k=3):
    ideal = sorted(rels, reverse=True)[:k]
    actual = rels[:k]
    idcg = _dcg(ideal)
    return _dcg(actual) / idcg if idcg > 0 else 0.0

def _norm(s):
    import re
    return re.sub(r"\s+", " ", s).strip().lower()

# ---------------------------------------------------------------------------
def main():
    t0 = time.time()

    # -- Load golden QA (chunk_id ground truth) ------------------------------
    golden = {}
    for l in GOLDEN.read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            golden[r["id"]] = r
    if not any("chunk_id" in g for g in golden.values()):
        sys.exit("[FATAL] golden_qa has no chunk_id field. "
                 "Run: python scripts/backfill_golden_chunk_ids.py --write")

    dump = [json.loads(l) for l in DUMP.read_text(encoding="utf-8").splitlines() if l.strip()]
    n    = len(dump)

    # -- Index docs + text->id lookup (so retrieved text maps back to chunk_id)
    docs = json.loads(DOCS.read_text(encoding="utf-8"))
    text2id = {}
    for i, d in enumerate(docs):
        text2id.setdefault(_norm(d), i)   # first occurrence wins (dups are rare)
    print(f"Loaded {n} questions, {len(docs)} index chunks. Building retrieval pipeline …")

    # -- Build retrieval pipeline (CPU, no SLM) ------------------------------
    from retrieval.dense_retriever import DenseRetriever
    from retrieval.sparse_retriever import SparseRetriever
    from retrieval.reranker import CrossEncoderReranker
    import utils.config as cfg

    dense  = DenseRetriever()
    dense.load_index()
    sparse = SparseRetriever()
    sparse.build_index_from_docs(dense.documents)
    reranker = CrossEncoderReranker()
    print(f"  Pipeline ready ({time.time()-t0:.0f}s). Loading embedder …")

    from sentence_transformers import SentenceTransformer
    import numpy as np
    st = SentenceTransformer("all-MiniLM-L6-v2")

    def cos(a, b):
        return float(np.dot(a, b))   # inputs are L2-normalized

    # Pre-embed gold chunks (for graded NDCG: how related is each retrieved chunk
    # to the gold chunk, beyond the binary ID hit).
    gold_ids   = {qid: g["chunk_id"] for qid, g in golden.items()}
    gold_texts = {qid: docs[g["chunk_id"]] for qid, g in golden.items()
                  if 0 <= g.get("chunk_id", -1) < len(docs)}
    g_keys = list(gold_texts.keys())
    g_embs = st.encode([gold_texts[k] for k in g_keys], batch_size=32,
                       show_progress_bar=False, normalize_embeddings=True)
    gold_emb = {k: e for k, e in zip(g_keys, g_embs)}
    print(f"  Gold chunks embedded ({time.time()-t0:.0f}s)")

    # -- Hybrid retrieval (RRF fusion, content-hashed keys) ------------------
    def retrieve_hybrid(question, top_k=cfg.TOP_K):
        dense_hits  = dense.retrieve(question, top_k=top_k * 4)
        sparse_hits = sparse.retrieve(question, top_k=top_k * 4)
        scores = {}
        def _add(hits):
            for rank, d in enumerate(hits):
                key = _norm(d)                       # FIX: full-content key, not d[:80]
                if key not in scores:
                    scores[key] = {"doc": d, "s": 0.0}
                scores[key]["s"] += 1.0 / (60 + rank + 1)
        _add(dense_hits)
        _add(sparse_hits)
        fused = sorted(scores.values(), key=lambda x: x["s"], reverse=True)
        docs_ = [f["doc"] for f in fused[:top_k * 4]]
        return reranker.rerank(question, docs_, top_k=top_k)

    # -- Per-question evaluation ---------------------------------------------
    per_q = []

    for i, r in enumerate(dump):
        qid = r["id"]
        if qid not in golden:
            continue
        g = golden[qid]
        gold_id = g.get("chunk_id", -1)
        if gold_id < 0:
            continue   # unmappable gold — excluded from Hit/MRR (reported as n_unmapped)

        q    = r["question"]
        diff = r.get("difficulty", "easy")
        sys_text = (f"Equation: {r['corpus_eq']}\n\n{r['raw_samples'][0]}"
                    if r.get("corpus_eq") else r["raw_samples"][0])
        hiconf = (g.get("chunk_id_method") == "substring"
                  or g.get("chunk_id_score", 0) >= HICONF_EMBED)

        try:
            retrieved = retrieve_hybrid(q, top_k=cfg.TOP_K)
        except Exception as e:
            print(f"  [WARN] Q{i+1} retrieval failed: {e}")
            continue

        # Map retrieved text -> chunk_id (exact, via normalized lookup)
        retrieved_ids = [text2id.get(_norm(t), -1) for t in retrieved]

        # Hit@k = gold_id present in top-k ids  (STANDARD definition)
        hit1 = int(len(retrieved_ids) >= 1 and retrieved_ids[0] == gold_id)
        hit3 = int(gold_id in retrieved_ids[:cfg.TOP_K])
        rr = 0.0
        for rank, rid in enumerate(retrieved_ids[:cfg.TOP_K]):
            if rid == gold_id:
                rr = 1.0 / (rank + 1)
                break

        # Graded NDCG: gold chunk -> 1.0; others -> cosine to gold (capped <1)
        chunk_embs = st.encode(retrieved, show_progress_bar=False, normalize_embeddings=True)
        rels = []
        for rid, ce in zip(retrieved_ids, chunk_embs):
            if rid == gold_id:
                rels.append(1.0)
            else:
                rels.append(min(0.99, max(0.0, cos(gold_emb[qid], ce))))
        ndcg = _ndcg(rels, k=cfg.TOP_K)

        q_emb   = st.encode([q], normalize_embeddings=True)[0]
        sys_emb = st.encode([sys_text], normalize_embeddings=True)[0]
        ctx_rel = cos(q_emb, chunk_embs[0]) if len(chunk_embs) else 0.0
        ev_emb  = st.encode([" ".join(retrieved)], normalize_embeddings=True)[0]
        faith_full = cos(sys_emb, ev_emb)

        per_q.append({
            "id": qid, "difficulty": diff, "question": q,
            "gold_chunk_id": gold_id, "retrieved_ids": retrieved_ids,
            "hiconf": hiconf, "gold_method": g.get("chunk_id_method"),
            "hit1": hit1, "hit3": hit3, "rr": round(rr, 4),
            "ndcg3": round(ndcg, 4),
            "ctx_relevancy": round(ctx_rel, 4),
            "faithfulness_full": round(faith_full, 4),
            "rel_grades": [round(x, 4) for x in rels],
        })

        if (i + 1) % 10 == 0 or i == n - 1:
            h1 = _avg([p["hit1"] for p in per_q]); h3 = _avg([p["hit3"] for p in per_q])
            print(f"  {i+1}/{n}  | Hit@1={h1:.2f} Hit@3={h3:.2f} "
                  f"MRR={_avg([p['rr'] for p in per_q]):.3f} ({time.time()-t0:.0f}s)")

    # -- Cohort aggregation --------------------------------------------------
    def agg(rows):
        if not rows:
            return None
        return {
            "n": len(rows),
            "Hit@1":  round(_avg([p["hit1"] for p in rows]), 4),
            "Hit@3":  round(_avg([p["hit3"] for p in rows]), 4),
            "MRR":    round(_avg([p["rr"] for p in rows]), 4),
            "NDCG@3": round(_avg([p["ndcg3"] for p in rows]), 4),
            "Context_Relevancy": round(_avg([p["ctx_relevancy"] for p in rows]), 4),
            "Faithfulness_full_evidence": round(_avg([p["faithfulness_full"] for p in rows]), 4),
        }

    full   = per_q
    hiconf = [p for p in per_q if p["hiconf"]]
    n_unmapped = sum(1 for g in golden.values() if g.get("chunk_id", -1) < 0)

    by_diff = {}
    for d in ["easy", "medium", "hard"]:
        sub = [p for p in full if p["difficulty"] == d]
        if sub:
            by_diff[d] = agg(sub)

    summary = {
        "n_evaluated_full": len(full),
        "n_evaluated_hiconf": len(hiconf),
        "n_unmapped_excluded": n_unmapped,
        "hit_definition": "gold chunk_id present in top-k retrieved chunk ids (exact ID containment)",
        "hiconf_definition": f"gold chunk assigned by substring match OR embedding score >= {HICONF_EMBED}",
        "overall_full":   agg(full),
        "overall_hiconf": agg(hiconf),
        "by_difficulty_full": by_diff,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"summary": summary, "per_question": per_q},
                                   indent=2, ensure_ascii=False), encoding="utf-8")

    # -- Print ---------------------------------------------------------------
    SEP = "-" * 64
    of, oh = summary["overall_full"], summary["overall_hiconf"]
    print(f"\n{SEP}")
    print(f"  STAGE 3 — RAG RETRIEVAL QUALITY")
    print(f"  Hit definition: gold chunk_id ∈ top-{cfg.TOP_K} (exact ID containment)")
    print(SEP)
    print(f"  {'metric':<22}{'FULL (n='+str(of['n'])+')':>16}{'HICONF (n='+str(oh['n'])+')':>18}")
    for m in ["Hit@1", "Hit@3", "MRR", "NDCG@3", "Context_Relevancy",
              "Faithfulness_full_evidence"]:
        print(f"  {m:<22}{of[m]:>16.4f}{oh[m]:>18.4f}")
    print(SEP)
    for d in ["easy", "medium", "hard"]:
        if d in by_diff:
            bd = by_diff[d]
            print(f"  {d:<8} (n={bd['n']:<3})  Hit@1={bd['Hit@1']:.2f}  "
                  f"Hit@3={bd['Hit@3']:.2f}  MRR={bd['MRR']:.3f}")
    print(SEP)
    print(f"  unmapped gold (excluded): {n_unmapped}")
    print(f"  saved -> {OUT_JSON}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
