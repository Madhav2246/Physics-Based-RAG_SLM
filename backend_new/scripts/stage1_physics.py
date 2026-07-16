"""
stage1_physics.py
-----------------
Stage 1 of the report evaluation: Physics Validation (the differentiator).

Compares THREE sides on the golden QA set, scored by the model-agnostic
neuro-symbolic physics scorer (parse + dimensional + numerical + coverage, 0-4):

  70B   : Llama-3.1-70B golden `answer` text (ungrounded baseline)
  SYS   : Complete System = corpus-grounded + best-of-N re-ranked (deployed-equivalent)
  RAW   : raw Qwen-0.5B output, no grounding (mean over samples)

Seeded (cfg.SEED) → reproducible. Reports overall + per-difficulty, with full
sub-metrics for every side. Writes JSON + a human summary.

Run from backend_new/:
  python scripts/stage1_physics.py --n 100 --samples 3
"""
import argparse, io, json, sys, time
from pathlib import Path
from collections import defaultdict

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import utils.config as cfg
from physics.physics_scorer import score_text


def _avg(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _agg(score_dicts):
    """Aggregate a list of physics_scorer dicts into rates + mean total."""
    n = len(score_dicts)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "avg_score":   round(_avg([s["total"] for s in score_dicts]), 3),
        "parseable":   round(100 * _avg([1 if s["parseable"]   else 0 for s in score_dicts]), 1),
        "dimensional": round(100 * _avg([1 if s["dimensional"] else 0 for s in score_dicts]), 1),
        "numerical":   round(100 * _avg([1 if s["numerical"]   else 0 for s in score_dicts]), 1),
        "coverage":    round(100 * _avg([s["coverage_frac"] for s in score_dicts]), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--golden", type=Path, default=ROOT / "data/evaluation/nvidia_golden_qa.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "data/evaluation/stage1_physics.json")
    args = ap.parse_args()
    base_seed = cfg.SEED if cfg.SEED is not None else 0

    rows = []
    with open(args.golden, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.n:
                break
            if line.strip():
                rows.append(json.loads(line))
    print(f"Loaded {len(rows)} QA  (max_new_tokens={cfg.MAX_NEW_TOKENS}, samples={args.samples}, seed={base_seed})")

    from pipeline.rag_pipeline import RAGPipeline
    from reasoning.prompt_builder import build_prompt
    pipe = RAGPipeline()
    pipe.retriever.dense.load_index()
    pipe.retriever.sparse.build_index_from_docs(pipe.retriever.dense.documents)
    print("Pipeline ready.\n" + "=" * 70)

    per_q = []
    t_start = time.time()
    for i, item in enumerate(rows):
        q   = item["question"].strip()
        diff = item.get("difficulty", "easy")

        evidence = pipe.retriever.retrieve(q, top_k=cfg.TOP_K)
        _, _, corpus_eq = pipe._find_corpus_equation(evidence)
        prompt = build_prompt(q, evidence, corpus_equation=corpus_eq)

        samples = pipe.slm.generate_multiple(prompt, n_samples=args.samples, seed=base_seed + i)

        # RAW: model's own output (mean over samples)
        raw_scores = [score_text(s, "RAW 0.5B") for s in samples]
        # SYS: corpus-grounded + best-of-N (deployed system picks best by physics score)
        sys_scores = []
        for s in samples:
            composed = (f"Equation: {corpus_eq}\n\n{s}" if corpus_eq
                        else f"Equation: NOT FOUND IN CORPUS\n\n{s}")
            sys_scores.append(score_text(composed, "SYS"))
        sys_best = max(sys_scores, key=lambda d: d["total"])   # best-of-N
        # 70B golden answer
        b70 = score_text(item.get("answer", ""), "70B")

        per_q.append({
            "id": item.get("id", f"Q{i+1}"), "difficulty": diff, "question": q,
            "raw_mean_total": round(_avg([d["total"] for d in raw_scores]), 3),
            "raw_repr": max(raw_scores, key=lambda d: d["total"]),   # best raw for sub-flags
            "sys": sys_best, "b70": b70,
        })
        if (i + 1) % 5 == 0 or i == len(rows) - 1:
            el = time.time() - t_start
            print(f"  {i+1}/{len(rows)} | 70B {b70['total']:.1f} | SYS {sys_best['total']:.1f} "
                  f"| RAWbest {per_q[-1]['raw_repr']['total']:.1f} | {el:.0f}s")

    # ---- Aggregate overall + per difficulty ----
    def collect(side):
        if side == "raw":
            return [r["raw_repr"] for r in per_q]      # best-of-N raw (capability)
        return [r[side] for r in per_q]

    summary = {
        "n": len(per_q), "samples_per_q": args.samples, "seed": base_seed,
        "max_new_tokens": cfg.MAX_NEW_TOKENS,
        "overall": {
            "70B": _agg([r["b70"] for r in per_q]),
            "SYS": _agg([r["sys"] for r in per_q]),
            "RAW_bestofN": _agg([r["raw_repr"] for r in per_q]),
            "RAW_mean_total": round(_avg([r["raw_mean_total"] for r in per_q]), 3),
        },
        "by_difficulty": {},
    }
    for d in ["easy", "medium", "hard"]:
        sub = [r for r in per_q if r["difficulty"] == d]
        if not sub:
            continue
        summary["by_difficulty"][d] = {
            "70B": _agg([r["b70"] for r in sub]),
            "SYS": _agg([r["sys"] for r in sub]),
            "RAW_bestofN": _agg([r["raw_repr"] for r in sub]),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "per_question": per_q},
                                   indent=2, ensure_ascii=False), encoding="utf-8")

    SEP = "-" * 70
    o = summary["overall"]
    print("\n" + SEP)
    print(f"  STAGE 1 — PHYSICS VALIDATION  (n={summary['n']}, seed={base_seed})")
    print(SEP)
    hdr = f"  {'side':<14}{'score/4':>8}{'parse%':>8}{'dim%':>7}{'num%':>7}{'cover%':>8}"
    print(hdr)
    for name, key in [("70B (baseline)", "70B"), ("Complete System", "SYS"), ("raw 0.5B", "RAW_bestofN")]:
        a = o[key]
        print(f"  {name:<14}{a['avg_score']:>8}{a['parseable']:>8}{a['dimensional']:>7}{a['numerical']:>7}{a['coverage']:>8}")
    print(f"  (raw 0.5B mean-over-samples score: {o['RAW_mean_total']}/4)")
    print(SEP)
    for d in ["easy", "medium", "hard"]:
        if d in summary["by_difficulty"]:
            bd = summary["by_difficulty"][d]
            print(f"  {d:<8} 70B {bd['70B']['avg_score']:.2f} | SYS {bd['SYS']['avg_score']:.2f} | RAW {bd['RAW_bestofN']['avg_score']:.2f}")
    print(SEP)
    print(f"  saved -> {args.out}")


if __name__ == "__main__":
    main()
