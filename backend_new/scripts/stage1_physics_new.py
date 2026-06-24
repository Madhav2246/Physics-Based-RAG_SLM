"""
stage1_physics_new.py
---------------------
Stage-1 RE-EVALUATION with the FIXED checker (physics/new_checker.py).

Differences vs stage1_physics.py:
  1. Scores with physics.new_checker.score_text  (letter-soup rejection +
     expanded device-physics vocabulary). Original physics_scorer untouched.
  2. STORES every raw answer text (70B golden, raw 0.5B samples, corpus eq).
     => any future checker change can re-score straight from the JSON with NO
        re-generation. (Addresses "store the answers".)
  3. Writes to a SEPARATE folder: data/evaluation/stage1_new_separate_eval/
     so the original Stage-1 results are never overwritten. If results are
     worse, just delete this folder and keep the original.

Run on Kaggle (P100) from backend_new/:
  python scripts/stage1_physics_new.py --n 100 --samples 3
Set cfg.SEED and MAX_NEW_TOKENS in utils/config.py (use 512 on GPU).

Re-score later WITHOUT re-running the model:
  python scripts/stage1_physics_new.py --rescore   # reads stored texts, re-applies checker
"""
import argparse, io, json, sys, time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import utils.config as cfg
from physics.new_checker import score_text   # <-- FIXED checker

OUT_DIR  = ROOT / "data" / "evaluation" / "stage1_new_separate_eval"
OUT_JSON = OUT_DIR / "stage1_new.json"
ANSWERS  = OUT_DIR / "answers_dump.jsonl"     # portable text store for re-scoring


def _avg(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _agg(score_dicts):
    n = len(score_dicts)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "avg_score":   round(_avg([s["total"] for s in score_dicts]), 3),
        "parseable":   round(100 * _avg([1 if s["parseable"]   else 0 for s in score_dicts]), 1),
        "dimensional": round(100 * _avg([1 if s["dimensional"] else 0 for s in score_dicts]), 1),
        "numerical":   round(100 * _avg([1 if s["numerical"]   else 0 for s in score_dicts]), 1),
        # coverage = vocabulary-overlap / APPLICABILITY (not a correctness metric)
        "coverage":    round(100 * _avg([s["coverage_frac"] for s in score_dicts]), 1),
        "applicability": _applicability(score_dicts),   # % parsed whose symbols are all mapped
        # conditional gates: pass-rate over the APPLICABLE subset only
        "nvr_conditional": _nvr_cond(score_dicts),
        "dcr_conditional": _dcr_cond(score_dicts),
    }


def _nvr_cond(score_dicts):
    """Numerical pass over numerically-evaluable equations (N/A symbolic excluded)."""
    ok = evaluable = 0
    for s in score_dicts:
        if s["parseable"]:
            nm = s["num_msg"]
            if "[OK]" in nm:
                ok += 1; evaluable += 1
            elif "Unresolved" in nm or "skipped" in nm.lower():
                pass                                   # N/A — not evaluable
            else:
                evaluable += 1                         # real numeric failure
    return round(100 * ok / evaluable, 1) if evaluable else None


def _dcr_cond(score_dicts):
    """Dimensional pass over dim-CHECKABLE equations (all RHS symbols mapped)."""
    ok = checkable = 0
    for s in score_dicts:
        if s["parseable"] and s["coverage_frac"] >= 0.999:
            checkable += 1
            if s["dimensional"]:
                ok += 1
    return round(100 * ok / checkable, 1) if checkable else None


def _applicability(score_dicts):
    """% of parsed equations whose symbols are fully in the validator vocabulary."""
    parsed = [s for s in score_dicts if s["parseable"]]
    if not parsed:
        return None
    full = sum(1 for s in parsed if s["coverage_frac"] >= 0.999)
    return round(100 * full / len(parsed), 1)


def _summarize(per_q, meta):
    summary = {**meta, "n": len(per_q), "overall": {
        "70B": _agg([r["b70"] for r in per_q]),
        "SYS": _agg([r["sys"] for r in per_q]),
        "RAW_bestofN": _agg([r["raw_repr"] for r in per_q]),
        "RAW_mean_total": round(_avg([r["raw_mean_total"] for r in per_q]), 3),
    }, "by_difficulty": {}}
    for d in ["easy", "medium", "hard"]:
        sub = [r for r in per_q if r["difficulty"] == d]
        if sub:
            summary["by_difficulty"][d] = {
                "70B": _agg([r["b70"] for r in sub]),
                "SYS": _agg([r["sys"] for r in sub]),
                "RAW_bestofN": _agg([r["raw_repr"] for r in sub]),
            }
    return summary


def _print(summary):
    SEP = "-" * 72
    o = summary["overall"]
    print("\n" + SEP)
    print(f"  STAGE 1 (NEW CHECKER) — PHYSICS VALIDATION  (n={summary['n']})")
    print(SEP)
    print(f"  {'side':<16}{'score/4':>8}{'parse%':>8}{'DCRcond%':>9}{'NVRcond%':>9}{'applic%':>8}{'cover%':>8}")
    for name, key in [("70B baseline", "70B"), ("Complete System", "SYS"), ("raw 0.5B", "RAW_bestofN")]:
        a = o[key]
        nv = a["nvr_conditional"] if a["nvr_conditional"] is not None else "-"
        dc = a["dcr_conditional"] if a["dcr_conditional"] is not None else "-"
        ap = a["applicability"] if a["applicability"] is not None else "-"
        print(f"  {name:<16}{a['avg_score']:>8}{a['parseable']:>8}{str(dc):>9}{str(nv):>9}{str(ap):>8}{a['coverage']:>8}")
    print("  (DCRcond/NVRcond = pass over checkable subset; cover/applic = vocabulary overlap, not correctness)")
    print(SEP)
    for d in ["easy", "medium", "hard"]:
        if d in summary["by_difficulty"]:
            bd = summary["by_difficulty"][d]
            print(f"  {d:<8} 70B {bd['70B']['avg_score']:.2f} | SYS {bd['SYS']['avg_score']:.2f} | RAW {bd['RAW_bestofN']['avg_score']:.2f}")
    print(SEP)


# ---------------------------------------------------------------------------
def rescore():
    """Re-apply the current checker to stored texts — NO model needed."""
    if not ANSWERS.exists():
        print(f"No stored answers at {ANSWERS}. Run a full pass first."); return
    per_q = []
    for line in ANSWERS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        t = json.loads(line)
        raw_scores = [score_text(s, "RAW") for s in t["raw_samples"]]
        sys_scores = [score_text((f"Equation: {t['corpus_eq']}\n\n{s}" if t["corpus_eq"]
                                  else f"Equation: NOT FOUND IN CORPUS\n\n{s}"), "SYS")
                      for s in t["raw_samples"]]
        per_q.append({
            "id": t["id"], "difficulty": t["difficulty"], "question": t["question"],
            "raw_mean_total": round(_avg([d["total"] for d in raw_scores]), 3),
            "raw_repr": max(raw_scores, key=lambda d: d["total"]),
            "sys": max(sys_scores, key=lambda d: d["total"]),
            "b70": score_text(t["b70_text"], "70B"),
        })
    summary = _summarize(per_q, {"mode": "rescore", "checker": "new_checker"})
    OUT_JSON.write_text(json.dumps({"summary": summary, "per_question": per_q},
                                   indent=2, ensure_ascii=False), encoding="utf-8")
    _print(summary)
    print(f"  re-scored from stored texts -> {OUT_JSON}")


def full_run(args):
    base_seed = cfg.SEED if cfg.SEED is not None else 0
    rows = []
    with open(args.golden, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.n:
                break
            if line.strip():
                rows.append(json.loads(line))
    print(f"Loaded {len(rows)} QA (max_new_tokens={cfg.MAX_NEW_TOKENS}, samples={args.samples}, seed={base_seed})")

    from pipeline.rag_pipeline import RAGPipeline
    from reasoning.prompt_builder import build_prompt
    pipe = RAGPipeline()
    pipe.retriever.dense.load_index()
    pipe.retriever.sparse.build_index_from_docs(pipe.retriever.dense.documents)
    print("Pipeline ready.\n" + "=" * 72)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ANSWERS.write_text("", encoding="utf-8")   # reset store

    per_q = []
    t0 = time.time()
    for i, item in enumerate(rows):
        q = item["question"].strip()
        diff = item.get("difficulty", "easy")
        evidence = pipe.retriever.retrieve(q, top_k=cfg.TOP_K)
        _, _, corpus_eq = pipe._find_corpus_equation(evidence)
        prompt = build_prompt(q, evidence, corpus_equation=corpus_eq)
        samples = pipe.slm.generate_multiple(prompt, n_samples=args.samples, seed=base_seed + i)
        b70_text = item.get("answer", "")

        # ---- store raw texts for future re-scoring ----
        with open(ANSWERS, "a", encoding="utf-8") as af:
            af.write(json.dumps({
                "id": item.get("id", f"Q{i+1}"), "difficulty": diff, "question": q,
                "corpus_eq": corpus_eq, "raw_samples": samples, "b70_text": b70_text,
            }, ensure_ascii=False) + "\n")

        raw_scores = [score_text(s, "RAW") for s in samples]
        sys_scores = [score_text((f"Equation: {corpus_eq}\n\n{s}" if corpus_eq
                                  else f"Equation: NOT FOUND IN CORPUS\n\n{s}"), "SYS")
                      for s in samples]
        per_q.append({
            "id": item.get("id", f"Q{i+1}"), "difficulty": diff, "question": q,
            "raw_mean_total": round(_avg([d["total"] for d in raw_scores]), 3),
            "raw_repr": max(raw_scores, key=lambda d: d["total"]),
            "sys": max(sys_scores, key=lambda d: d["total"]),
            "b70": score_text(b70_text, "70B"),
        })
        if (i + 1) % 5 == 0 or i == len(rows) - 1:
            print(f"  {i+1}/{len(rows)} | 70B {per_q[-1]['b70']['total']:.1f} "
                  f"| SYS {per_q[-1]['sys']['total']:.1f} | RAW {per_q[-1]['raw_repr']['total']:.1f} "
                  f"| {time.time()-t0:.0f}s")

    summary = _summarize(per_q, {
        "checker": "new_checker", "samples_per_q": args.samples,
        "seed": base_seed, "max_new_tokens": cfg.MAX_NEW_TOKENS,
    })
    OUT_JSON.write_text(json.dumps({"summary": summary, "per_question": per_q},
                                   indent=2, ensure_ascii=False), encoding="utf-8")
    _print(summary)
    print(f"  saved -> {OUT_JSON}")
    print(f"  texts -> {ANSWERS}  (re-score anytime with --rescore, no GPU)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--golden", type=Path, default=ROOT / "data/evaluation/nvidia_golden_qa.jsonl")
    ap.add_argument("--rescore", action="store_true", help="re-score stored texts, no model")
    args = ap.parse_args()
    if args.rescore:
        rescore()
    else:
        full_run(args)


if __name__ == "__main__":
    main()
