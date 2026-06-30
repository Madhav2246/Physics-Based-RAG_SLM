"""
stage4b_validator_test.py
--------------------------
Kaggle script: Does the physics-score validator matter with n>=3 samples?

Ablation configs (all use same retrieval — full hybrid):
  sys_best3   Corpus-grounded, best-of-3 by physics score  (deployed system)
  sys_first3  Corpus-grounded, always take sample[0]       (-bestofN)
  sys_rand3   Corpus-grounded, random sample               (-validator)
  raw_best3   No corpus_eq, best-of-3 by physics score     (validator on raw)
  raw_first3  No corpus_eq, sample[0]                      (baseline raw)

Expected: validator gap (best3 - rand3) is larger at n=3 than n=2.
Expected: raw_best3 - raw_first3 gap shows true discriminatory power without corpus_eq clamping.

Stores all texts in answers_dump_n3.jsonl for re-scoring.

Run from backend_new/ on Kaggle P100:
  python scripts/stage4b_validator_test.py --n 100 --samples 3
"""
import argparse, io, json, random, sys, time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import utils.config as cfg
from physics.new_checker import score_text

OUT_DIR  = ROOT / "data" / "evaluation" / "stage4b_validator"
OUT_JSON = OUT_DIR / "stage4b_validator.json"
DUMP     = OUT_DIR / "answers_dump_n3.jsonl"

random.seed(42)

def _avg(xs): return sum(xs)/len(xs) if xs else 0.0

def _agg(score_dicts):
    nvr_ok = nvr_eval = dcr_ok = dcr_check = 0
    for s in score_dicts:
        if s["parseable"]:
            nm = s["num_msg"]
            if "[OK]" in nm:
                nvr_ok += 1; nvr_eval += 1
            elif "Unresolved" not in nm and "skipped" not in nm.lower():
                nvr_eval += 1
            if s["coverage_frac"] >= 0.999:
                dcr_check += 1
                if s["dimensional"]: dcr_ok += 1
    return {
        "avg_score":   round(_avg([s["total"] for s in score_dicts]), 3),
        "parseable":   round(100*_avg([1 if s["parseable"] else 0 for s in score_dicts]), 1),
        "coverage":    round(100*_avg([s["coverage_frac"] for s in score_dicts]), 1),
        "nvr_cond":    round(100*nvr_ok/nvr_eval, 1) if nvr_eval else None,
        "dcr_cond":    round(100*dcr_ok/dcr_check, 1) if dcr_check else None,
    }

def _make_sys(corpus_eq, sample):
    return (f"Equation: {corpus_eq}\n\n{sample}" if corpus_eq
            else f"Equation: NOT FOUND IN CORPUS\n\n{sample}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",       type=int, default=100)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--golden",  type=Path,
                    default=ROOT/"data/evaluation/nvidia_golden_qa.jsonl")
    ap.add_argument("--rescore", action="store_true",
                    help="re-score stored texts, no model needed")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.rescore:
        _rescore()
        return

    base_seed = cfg.SEED or 0
    rows = [json.loads(l) for l in open(args.golden, encoding="utf-8") if l.strip()][:args.n]
    print(f"Loaded {len(rows)} QA | samples={args.samples} | seed={base_seed} | max_tokens={cfg.MAX_NEW_TOKENS}")

    from pipeline.rag_pipeline import RAGPipeline
    from reasoning.prompt_builder import build_prompt
    pipe = RAGPipeline()
    pipe.retriever.dense.load_index()
    pipe.retriever.sparse.build_index_from_docs(pipe.retriever.dense.documents)
    print("Pipeline ready.\n" + "="*70)

    DUMP.write_text("", encoding="utf-8")

    # ── Per-question: score under all 5 configs ──────────────────────────────
    agg_buckets = {k: [] for k in ["sys_best","sys_first","sys_rand","raw_best","raw_first"]}
    t0 = time.time()

    for i, item in enumerate(rows):
        q        = item["question"].strip()
        diff     = item.get("difficulty", "easy")
        evidence = pipe.retriever.retrieve(q, top_k=cfg.TOP_K)
        _, _, corpus_eq = pipe._find_corpus_equation(evidence)
        prompt   = build_prompt(q, evidence, corpus_equation=corpus_eq)
        samples  = pipe.slm.generate_multiple(prompt, n_samples=args.samples,
                                              seed=base_seed + i)
        # Store
        with open(DUMP, "a", encoding="utf-8") as af:
            af.write(json.dumps({
                "id": item.get("id", f"Q{i+1}"), "difficulty": diff,
                "question": q, "corpus_eq": corpus_eq, "raw_samples": samples,
            }, ensure_ascii=False) + "\n")

        sys_scores = [score_text(_make_sys(corpus_eq, s), "SYS") for s in samples]
        raw_scores = [score_text(s, "RAW") for s in samples]

        agg_buckets["sys_best"].append(max(sys_scores,  key=lambda d: d["total"]))
        agg_buckets["sys_first"].append(sys_scores[0])
        agg_buckets["sys_rand"].append(random.choice(sys_scores))
        agg_buckets["raw_best"].append(max(raw_scores,  key=lambda d: d["total"]))
        agg_buckets["raw_first"].append(raw_scores[0])

        if (i+1) % 5 == 0 or i == len(rows)-1:
            sb = agg_buckets["sys_best"][-1]["total"]
            sf = agg_buckets["sys_first"][-1]["total"]
            rb = agg_buckets["raw_best"][-1]["total"]
            print(f"  {i+1}/{len(rows)} | sys_best={sb:.1f} sys_first={sf:.1f} raw_best={rb:.1f} | {time.time()-t0:.0f}s")

    # ── Aggregate ────────────────────────────────────────────────────────────
    summary = {k: _agg(v) for k, v in agg_buckets.items()}

    # Deltas that answer the question
    n_samples = args.samples
    sys_gap   = round(summary["sys_best"]["avg_score"] - summary["sys_rand"]["avg_score"], 3)
    raw_gap   = round(summary["raw_best"]["avg_score"] - summary["raw_first"]["avg_score"], 3)
    bonN_gain = round(summary["sys_best"]["avg_score"] - summary["sys_first"]["avg_score"], 3)
    summary["_analysis"] = {
        f"validator_gap_SYS_n{n_samples}": sys_gap,
        f"validator_gap_RAW_n{n_samples}": raw_gap,
        f"bestofN_gain_n{n_samples}":      bonN_gain,
        "note": ("sys_gap = physics-sel vs random on corpus-grounded answers. "
                 "raw_gap = physics-sel on raw (no corpus_eq) — true discriminatory power.")
    }

    OUT_JSON.write_text(json.dumps({"n_samples": n_samples, "summary": summary},
                                   indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Print ────────────────────────────────────────────────────────────────
    SEP = "-"*68
    print(f"\n{SEP}")
    print(f"  STAGE 4b — VALIDATOR MATTERS? (n={len(rows)}, samples={n_samples})")
    print(SEP)
    print(f"  {'Config':<16}{'Score':>7}{'Parse%':>8}{'DCRcond%':>10}{'NVRcond%':>10}")
    for name, label in [
        ("sys_best",  f"SYS best-of-{n_samples} (physics sel)"),
        ("sys_first", f"SYS first sample (-bestofN)"),
        ("sys_rand",  f"SYS random sel  (-validator)"),
        ("raw_best",  f"RAW best-of-{n_samples} (physics sel)"),
        ("raw_first", f"RAW first sample"),
    ]:
        a = summary[name]
        dc = a["dcr_cond"] if a["dcr_cond"] is not None else "-"
        nv = a["nvr_cond"] if a["nvr_cond"] is not None else "-"
        print(f"  {label:<40}{a['avg_score']:>7.3f}{a['parseable']:>8.1f}{str(dc):>10}{str(nv):>10}")
    print(SEP)
    an = summary["_analysis"]
    print(f"  Validator gap  SYS (n={n_samples}): {an[f'validator_gap_SYS_n{n_samples}']:+.3f}  "
          f"(was +0.013 at n=2 stored ablation)")
    print(f"  Validator gap  RAW (n={n_samples}): {an[f'validator_gap_RAW_n{n_samples}']:+.3f}  "
          f"(no corpus_eq clamping — true discriminatory power)")
    print(f"  Best-of-N gain (n={n_samples}):     {an[f'bestofN_gain_n{n_samples}']:+.3f}")
    print(SEP)
    print(f"  texts -> {DUMP}")
    print(f"  saved -> {OUT_JSON}  ({time.time()-t0:.0f}s)")


def _rescore():
    if not DUMP.exists():
        print(f"No stored answers at {DUMP}. Run full pass first."); return
    records = [json.loads(l) for l in DUMP.read_text(encoding="utf-8").splitlines() if l.strip()]
    agg_buckets = {k: [] for k in ["sys_best","sys_first","sys_rand","raw_best","raw_first"]}
    for r in records:
        samples   = r["raw_samples"]
        corpus_eq = r.get("corpus_eq", "")
        sys_scores = [score_text(_make_sys(corpus_eq, s), "SYS") for s in samples]
        raw_scores = [score_text(s, "RAW") for s in samples]
        agg_buckets["sys_best"].append(max(sys_scores,  key=lambda d: d["total"]))
        agg_buckets["sys_first"].append(sys_scores[0])
        agg_buckets["sys_rand"].append(random.choice(sys_scores))
        agg_buckets["raw_best"].append(max(raw_scores,  key=lambda d: d["total"]))
        agg_buckets["raw_first"].append(raw_scores[0])
    summary = {k: _agg(v) for k, v in agg_buckets.items()}
    n = len(records); ns = len(records[0]["raw_samples"])
    sys_gap = round(summary["sys_best"]["avg_score"] - summary["sys_rand"]["avg_score"], 3)
    raw_gap = round(summary["raw_best"]["avg_score"] - summary["raw_first"]["avg_score"], 3)
    summary["_analysis"] = {
        f"validator_gap_SYS_n{ns}": sys_gap,
        f"validator_gap_RAW_n{ns}": raw_gap,
    }
    OUT_JSON.write_text(json.dumps({"n_samples": ns, "summary": summary},
                                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Re-scored {n} records (n_samples={ns})")
    print(f"  Validator gap SYS: {sys_gap:+.3f}  RAW: {raw_gap:+.3f}")
    print(f"  -> {OUT_JSON}")


if __name__ == "__main__":
    main()
