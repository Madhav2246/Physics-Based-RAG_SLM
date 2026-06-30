"""
stage4_ablation.py
------------------
Stage 4: Ablation Study — contribution of each architectural component.

Ablation configs (all use same 100 QA, seed=42 stored samples):
  full        Full system (retrieval: dense+sparse+reranker, best-of-N physics selection)
  -reranker   Dense+sparse RRF only, no CrossEncoder
  -sparse     Dense (FAISS) only retrieval
  -dense      Sparse (BM25) only retrieval
  -bestofN    Full retrieval, but take sample[0] (no best-of-N selection)
  -validator  Full retrieval, but random sample (no physics-score selection)
  raw_0.5b    No retrieval at all (already in Stage 1; re-scored here for consistency)

Generation-side ablations use stored texts from answers_dump.jsonl (no GPU).
Retrieval-side ablations re-run retrieval on CPU; use stored raw samples.

Run from backend_new/:
  python scripts/stage4_ablation.py
"""
import io, json, random, sys, time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DUMP    = ROOT / "data" / "evaluation" / "answers_dump.jsonl"
# Fall back to stage1's subfolder if the promoted copy isn't present.
if not DUMP.exists():
    _alt = ROOT / "data" / "evaluation" / "stage1_new_separate_eval" / "answers_dump.jsonl"
    if _alt.exists():
        DUMP = _alt
OUT_DIR = ROOT / "data" / "evaluation" / "stage4_ablation"
OUT_JSON = OUT_DIR / "stage4_ablation.json"

random.seed(42)

def _avg(xs): return sum(xs)/len(xs) if xs else 0.0


def _agg(score_dicts):
    n = len(score_dicts)
    if n == 0:
        return {"n": 0}
    # Conditional NVR and DCR
    nvr_ok = nvr_eval = 0
    dcr_ok = dcr_check = 0
    for s in score_dicts:
        if s["parseable"]:
            nm = s["num_msg"]
            if "[OK]" in nm:
                nvr_ok += 1; nvr_eval += 1
            elif "Unresolved" in nm or "skipped" in nm.lower():
                pass
            else:
                nvr_eval += 1
            if s["coverage_frac"] >= 0.999:
                dcr_check += 1
                if s["dimensional"]:
                    dcr_ok += 1
    return {
        "n": n,
        "avg_score":   round(_avg([s["total"] for s in score_dicts]), 3),
        "parseable":   round(100*_avg([1 if s["parseable"] else 0 for s in score_dicts]), 1),
        "coverage":    round(100*_avg([s["coverage_frac"] for s in score_dicts]), 1),
        "nvr_cond":    round(100*nvr_ok/nvr_eval, 1) if nvr_eval else None,
        "dcr_cond":    round(100*dcr_ok/dcr_check, 1) if dcr_check else None,
    }


def _make_sys_text(corpus_eq, raw_sample):
    return (f"Equation: {corpus_eq}\n\n{raw_sample}" if corpus_eq
            else f"Equation: NOT FOUND IN CORPUS\n\n{raw_sample}")


def _extract_corpus_eq(evidence_chunks, validator):
    """Find first parseable physics equation in evidence chunks."""
    import sympy as sp
    for chunk in evidence_chunks:
        c_lhs, c_rhs, c_msg = validator.validate(chunk)
        if c_lhs is None or "[OK]" not in c_msg:
            continue
        if not (isinstance(c_lhs, sp.Basic) and isinstance(c_rhs, sp.Basic)):
            continue
        if c_lhs.is_number and c_rhs.is_number:
            continue
        if len(c_lhs.free_symbols | c_rhs.free_symbols) < 2:
            continue
        return f"{c_lhs} = {c_rhs}"
    return None


def main():
    t0 = time.time()
    records = [json.loads(l) for l in DUMP.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(records)
    print(f"Loaded {n} records. Building retrieval pipeline …")

    import utils.config as cfg
    from physics.new_checker import score_text
    from physics.equation_validator import EquationValidator
    from retrieval.dense_retriever import DenseRetriever
    from retrieval.sparse_retriever import SparseRetriever
    from retrieval.reranker import CrossEncoderReranker

    dense    = DenseRetriever()
    dense.load_index()
    sparse   = SparseRetriever()
    sparse.build_index_from_docs(dense.documents)
    reranker = CrossEncoderReranker()
    validator = EquationValidator()
    top_k = cfg.TOP_K
    print(f"  Pipeline ready ({time.time()-t0:.0f}s)\n")

    # ── Retrieval helpers ───────────────────────────────────────────────────
    def rrf_fuse(dense_hits, sparse_hits, top_n):
        scores = {}
        for rank, d in enumerate(dense_hits):
            key = d[:80]
            scores[key] = scores.get(key, {"doc": d, "s": 0.0})
            scores[key]["s"] += 1.0 / (60 + rank + 1)
        for rank, d in enumerate(sparse_hits):
            key = d[:80]
            if key not in scores:
                scores[key] = {"doc": d, "s": 0.0}
            scores[key]["s"] += 1.0 / (60 + rank + 1)
        return [v["doc"] for v in sorted(scores.values(), key=lambda x: x["s"], reverse=True)[:top_n]]

    def retrieve_no_reranker(q):
        d = dense.retrieve(q, top_k=top_k*2)
        s = sparse.retrieve(q, top_k=top_k*2)
        return rrf_fuse(d, s, top_k)

    def retrieve_dense_only(q):
        return dense.retrieve(q, top_k=top_k)

    def retrieve_sparse_only(q):
        return sparse.retrieve(q, top_k=top_k)

    def retrieve_full(q):
        d = dense.retrieve(q, top_k=top_k*2)
        s = sparse.retrieve(q, top_k=top_k*2)
        fused = rrf_fuse(d, s, top_k*2)
        return reranker.rerank(q, fused, top_k)

    # ── Run each ablation ───────────────────────────────────────────────────
    CONFIGS = [
        # (name, retrieval_fn, selection)
        # selection: "best" | "first" | "random"
        ("full",         retrieve_full,          "best"),
        ("-reranker",    retrieve_no_reranker,   "best"),
        ("-sparse",      retrieve_dense_only,    "best"),
        ("-dense",       retrieve_sparse_only,   "best"),
        ("-bestofN",     retrieve_full,          "first"),
        ("-validator",   retrieve_full,          "random"),
        ("raw_0.5b",     None,                   "best"),    # no retrieval
    ]

    results = {}

    for cfg_name, ret_fn, selection in CONFIGS:
        print(f"  Running ablation: {cfg_name} …")
        per_q_scores = []

        for i, r in enumerate(records):
            q       = r["question"]
            samples = r["raw_samples"]

            if ret_fn is not None:
                evidence = ret_fn(q)
                corpus_eq = _extract_corpus_eq(evidence, validator)
            else:
                corpus_eq = None   # raw — no retrieval

            if cfg_name == "raw_0.5b":
                # Raw: score each sample without corpus equation
                raw_scores = [score_text(s, "RAW") for s in samples]
                if selection == "best":
                    chosen = max(raw_scores, key=lambda d: d["total"])
                else:
                    chosen = raw_scores[0]
                per_q_scores.append(chosen)
            else:
                # SYS-style: prepend corpus equation to each sample, then select
                sys_texts  = [_make_sys_text(corpus_eq, s) for s in samples]
                sys_scores = [score_text(t, cfg_name) for t in sys_texts]
                if selection == "best":
                    chosen = max(sys_scores, key=lambda d: d["total"])
                elif selection == "first":
                    chosen = sys_scores[0]
                else:   # random
                    chosen = random.choice(sys_scores)
                per_q_scores.append(chosen)

        agg = _agg(per_q_scores)
        results[cfg_name] = {**agg, "selection": selection,
                             "retrieval": cfg_name if ret_fn is None else "full" if cfg_name=="full" else cfg_name}
        nv = agg["nvr_cond"] if agg["nvr_cond"] is not None else "-"
        dc = agg["dcr_cond"] if agg["dcr_cond"] is not None else "-"
        print(f"    {cfg_name:<16}  score={agg['avg_score']:.3f}  parse={agg['parseable']:.1f}%"
              f"  DCRcond={str(dc):<5}  NVRcond={str(nv):<5}  cover={agg['coverage']:.1f}%"
              f"  ({time.time()-t0:.0f}s)")

    # ── Delta vs full system ─────────────────────────────────────────────────
    full_score = results["full"]["avg_score"]
    for k, v in results.items():
        v["delta_vs_full"] = round(v["avg_score"] - full_score, 3)

    # ── Save ─────────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"full_score": full_score, "ablations": results},
                                   indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Print summary table ──────────────────────────────────────────────────
    SEP = "-" * 76
    print(f"\n{SEP}")
    print(f"  STAGE 4 — ABLATION STUDY  (n={n}, new_checker v2)")
    print(SEP)
    print(f"  {'Config':<18}{'Score/4':>8}{'Δ vs Full':>10}{'Parse%':>8}{'DCRcond%':>10}{'NVRcond%':>10}{'Cover%':>8}")
    for cfg_name, v in results.items():
        nv = v["nvr_cond"] if v["nvr_cond"] is not None else "-"
        dc = v["dcr_cond"] if v["dcr_cond"] is not None else "-"
        delta = v["delta_vs_full"]
        delta_str = f"{delta:+.3f}"
        marker = " ←" if cfg_name == "full" else ""
        print(f"  {cfg_name:<18}{v['avg_score']:>8.3f}{delta_str:>10}{v['parseable']:>8.1f}"
              f"{str(dc):>10}{str(nv):>10}{v['coverage']:>8.1f}{marker}")
    print(SEP)
    print(f"  Δ < 0 = component removal hurts.  Saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
