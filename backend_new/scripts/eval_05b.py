"""
eval_05b.py
-----------
Focused, REPRODUCIBLE evaluation of the 0.5B RAG model (Tier 1 + Tier 2).

Why this script exists
======================
The old headline number bounced run-to-run (e.g. 1.4 -> 0.7 with no code
change) because generation was unseeded and only ONE sample per question was
scored — a single stochastic draw of a knife-edge metric. This script fixes
both problems for the 0.5B specifically:

  Tier 1 (reproducibility):
    * Generation is seeded (cfg.SEED), so re-running gives identical outputs.
    * Each question is sampled N times (cfg.EVAL_SAMPLES_PER_Q) and scored as a
      DISTRIBUTION — we report mean +/- std, best-of-N and worst-of-N instead of
      one coin flip.

  Tier 2 (fair scoring):
    * Uses the shared physics_scorer with PARTIAL coverage credit and
      DECOUPLED sub-checks, so a near-correct equation no longer scores 0.

Two views of the 0.5B are reported:
    RAW  = what the 0.5B itself generates (shows the model's true variance)
    SYS  = the deployed RAG answer (corpus equation prepended, as in production)
The static 70B golden answers are scored once as a fixed reference.

Run from the backend/ directory:
    python scripts/eval_05b.py
    python scripts/eval_05b.py --n 20 --samples 5
"""
import argparse
import io
import json
import statistics
import sys
import time
from pathlib import Path

# UTF-8 safe stdout on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import utils.config as cfg
from physics.physics_scorer import score_text


def _mean(xs):
    return statistics.mean(xs) if xs else 0.0


def _std(xs):
    # population std: defined (0.0) even for a single sample
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Reproducible 0.5B RAG evaluation")
    parser.add_argument("--n", type=int, default=20,
                        help="number of questions to evaluate")
    parser.add_argument("--samples", type=int, default=cfg.EVAL_SAMPLES_PER_Q,
                        help="samples drawn per question (for mean +/- std)")
    parser.add_argument("--golden", type=Path,
                        default=PROJECT_ROOT / "data/evaluation/nvidia_golden_qa.jsonl")
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "data/evaluation/eval_05b_distribution.json")
    args = parser.parse_args()

    base_seed = cfg.SEED if cfg.SEED is not None else 0

    # -- Load questions --------------------------------------------------------
    print(f"Loading first {args.n} questions from {args.golden.name} ...")
    dataset = []
    with open(args.golden, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.n:
                break
            if line.strip():
                dataset.append(json.loads(line))
    print(f"  Loaded {len(dataset)} questions\n")

    # -- Init pipeline (loads the fine-tuned 0.5B + retriever) -----------------
    print("Initialising RAG pipeline (0.5B + LoRA + retriever) ...")
    from pipeline.rag_pipeline import RAGPipeline
    from reasoning.prompt_builder import build_prompt
    pipeline = RAGPipeline()
    pipeline.retriever.dense.load_index()
    pipeline.retriever.sparse.build_index_from_docs(pipeline.retriever.dense.documents)
    print(f"Pipeline ready. Sampling {args.samples}x per question, seed={base_seed}.\n"
          + "=" * 70)

    records = []
    for i, item in enumerate(dataset):
        q = item["question"].strip()
        evidence = pipeline.retriever.retrieve(q, top_k=cfg.TOP_K)

        # Mirror the deployed pipeline: ground on a corpus equation if present.
        _, _, corpus_equation = pipeline._find_corpus_equation(evidence)
        prompt = build_prompt(q, evidence, corpus_equation=corpus_equation)

        # Deterministic, reproducible multi-sample draw for THIS question.
        t0 = time.time()
        responses = pipeline.slm.generate_multiple(
            prompt, n_samples=args.samples, seed=base_seed + i
        )
        latency = (time.time() - t0) / max(args.samples, 1)

        raw_scores, sys_scores = [], []
        for model_response in responses:
            # RAW: the model's own output
            raw_scores.append(score_text(model_response, "RAG 0.5B"))
            # SYS: deployed answer (corpus equation prepended, as in production)
            if corpus_equation:
                composed = f"Equation: {corpus_equation}\n\n{model_response}"
            else:
                composed = f"Equation: NOT FOUND IN CORPUS\n\n{model_response}"
            sys_scores.append(score_text(composed, "RAG 0.5B"))

        raw_totals = [s["total"] for s in raw_scores]
        sys_totals = [s["total"] for s in sys_scores]
        score_70b = score_text(item.get("answer", ""), "NVIDIA 70B")

        rec = {
            "id":          item.get("id", f"Q{i+1}"),
            "difficulty":  item.get("difficulty", "easy"),
            "question":    q,
            "raw_samples": raw_scores,
            "sys_samples": sys_scores,
            "raw_mean":    round(_mean(raw_totals), 4),
            "raw_std":     round(_std(raw_totals), 4),
            "raw_best":    max(raw_totals),
            "raw_worst":   min(raw_totals),
            "sys_mean":    round(_mean(sys_totals), 4),
            "score_70b":   score_70b["total"],
            "latency":     round(latency, 2),
        }
        records.append(rec)
        print(f"Q{i+1:<2} | RAW {rec['raw_mean']:.2f}±{rec['raw_std']:.2f} "
              f"(best {rec['raw_best']:.2f}/worst {rec['raw_worst']:.2f}) | "
              f"SYS {rec['sys_mean']:.2f} | 70B {rec['score_70b']:.2f} | "
              f"{rec['latency']:.1f}s/sample")

    # -- Aggregate -------------------------------------------------------------
    def rate(key):
        """Pass rate of a boolean sub-check across ALL question x sample raw scores."""
        flags = [s[key] for r in records for s in r["raw_samples"]]
        return _mean([1.0 if f else 0.0 for f in flags])

    def cov_rate():
        fracs = [s["coverage_frac"] for r in records for s in r["raw_samples"]]
        return _mean(fracs)

    raw_means = [r["raw_mean"] for r in records]
    raw_stds  = [r["raw_std"] for r in records]
    sys_means = [r["sys_mean"] for r in records]
    b70       = [r["score_70b"] for r in records]

    summary = {
        "n_questions":        len(records),
        "samples_per_q":      args.samples,
        "seed":               base_seed,
        "raw_0p5b": {
            "avg_score":        round(_mean(raw_means), 4),
            "within_q_std":     round(_mean(raw_stds), 4),   # single-run wobble
            "across_q_std":     round(_std(raw_means), 4),
            "best_of_n_avg":    round(_mean([r["raw_best"] for r in records]), 4),
            "worst_of_n_avg":   round(_mean([r["raw_worst"] for r in records]), 4),
            "parseable_rate":   round(rate("parseable"), 4),
            "dimensional_rate": round(rate("dimensional"), 4),
            "numerical_rate":   round(rate("numerical"), 4),
            "avg_coverage":     round(cov_rate(), 4),
        },
        "sys_0p5b_deployed":  {"avg_score": round(_mean(sys_means), 4)},
        "ref_70b":            {"avg_score": round(_mean(b70), 4)},
    }

    SEP = "-" * 70
    print(f"\n{SEP}")
    print("  0.5B REPRODUCIBLE EVALUATION  (mean +/- std over "
          f"{args.samples} samples/question, seed={base_seed})")
    print(SEP)
    r = summary["raw_0p5b"]
    print(f"  RAW 0.5B  (model's own output)")
    print(f"    Avg physics score   : {r['avg_score']:.2f} / 4.00   "
          f"(+/- {r['within_q_std']:.2f} within a single run)")
    print(f"    Best-of-{args.samples} avg      : {r['best_of_n_avg']:.2f} / 4.00")
    print(f"    Worst-of-{args.samples} avg     : {r['worst_of_n_avg']:.2f} / 4.00")
    print(f"    Equation parseable  : {r['parseable_rate']*100:.0f}%")
    print(f"    Dimensional pass    : {r['dimensional_rate']*100:.0f}%")
    print(f"    Numerical pass      : {r['numerical_rate']*100:.0f}%")
    print(f"    Avg symbol coverage : {r['avg_coverage']*100:.0f}%  (partial credit)")
    print(f"\n  SYS 0.5B  (deployed RAG, corpus eq grounded): "
          f"{summary['sys_0p5b_deployed']['avg_score']:.2f} / 4.00")
    print(f"  70B reference (static golden answers)        : "
          f"{summary['ref_70b']['avg_score']:.2f} / 4.00")
    print(SEP)
    print("  NOTE: re-running this script gives the SAME numbers (seeded).")
    print("        The +/- is sampling spread, not run-to-run noise.")
    print(SEP)

    # -- Save ------------------------------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"summary": summary, "per_question": records},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  Full distribution saved -> {args.out}\n")


if __name__ == "__main__":
    main()
