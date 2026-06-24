"""
compare_physics_validators.py
-----------------------------
Runs the same deterministic physics validator against TWO sets of answers:
  A) NVIDIA Llama-3.1-70B answers  (from nvidia_golden_qa.jsonl  → "answer" field)
  B) RAG 0.5B answers              (from eval_results_*.jsonl    → "actual_response" field)

Physics Score per answer (0–4 points):
  +1  Equation found and SymPy-parseable       (symbolic check)
  +1  Dimensionally consistent LHS = RHS       (dimensional check)
  +1  Numerically realistic after substitution (numerical check)
  +1  All symbols resolved (no unknowns left)  (coverage check)

Output:
  data/evaluation/physics_comparison.json   ← per-question breakdown
  Printed table to stdout                   ← paste into paper

Run:
  python scripts/compare_physics_validators.py
  python scripts/compare_physics_validators.py --golden data/evaluation/nvidia_golden_qa.jsonl
                                               --results data/evaluation/eval_results_nvidia_golden_qa.jsonl
"""

import argparse
import io
import json
import sys
from pathlib import Path

# Force UTF-8 stdout/stderr on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pathlib
_orig_read_text = pathlib.Path.read_text
def _utf8_read_text(self, encoding=None, errors=None):
    return _orig_read_text(self, encoding=encoding or "utf-8", errors=errors)
pathlib.Path.read_text = _utf8_read_text

import builtins
_orig_open = builtins.open
def _utf8_open(*args, **kwargs):
    mode = kwargs.get("mode", args[1] if len(args) > 1 else "r")
    if "b" not in mode and "encoding" not in kwargs:
        kwargs["encoding"] = "utf-8"
    return _orig_open(*args, **kwargs)
builtins.open = _utf8_open

import os
os.environ["HF_HOME"] = "d:/S6/NLP/Physics_Based_RAG_SLM/hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    del os.environ["SSL_CERT_FILE"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from physics.physics_scorer import score_text
from physics.physics_explainer import explain_physics_score


# -----------------------------------------------------------------------------
# Core scorer
# -----------------------------------------------------------------------------

def physics_score(text: str, model_label: str = "") -> dict:
    """
    Run the shared physics scorer (Tier-2: partial coverage + decoupled checks)
    and attach plain-English explanations for the per-question breakdown.
    """
    score = score_text(text, model_label)
    score["explanation"] = explain_physics_score(
        score["sym_msg"], score["dim_msg"], score["num_msg"],
        score["equation"], model_label,
    )
    return score


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _load_json_or_jsonl(path: Path) -> list:
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    return _load_jsonl(path)


def _pct(n, total):
    return f"{n}/{total} ({100*n/total:.1f}%)" if total else "0/0 (0%)"


def _bar(score, max_score=4, width=20):
    filled = int(round(score / max_score * width))
    return "#" * filled + "." * (width - filled)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Physics validator: 70B vs RAG 0.5B")
    parser.add_argument(
        "--golden",
        type=Path,
        default=PROJECT_ROOT / "data/evaluation/nvidia_golden_qa.jsonl",
        help="NVIDIA golden QA JSONL (has 'answer' field = 70B answer)",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=PROJECT_ROOT / "data/evaluation/eval_results_nvidia_golden_qa.jsonl",
        help="RAG eval results JSONL (has 'actual_response' field = RAG answer)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data/evaluation/physics_comparison_nvidia.json",
    )
    args = parser.parse_args()

    if not args.golden.exists():
        print(f"Golden QA not found: {args.golden}")
        sys.exit(1)

    golden = _load_json_or_jsonl(args.golden)
    print(f"Loaded {len(golden)} golden QA entries (70B answers)")

    # Build lookup: question → RAG answer
    rag_lookup: dict[str, str] = {}
    if args.results.exists():
        results = _load_jsonl(args.results)
        for r in results:
            q = r.get("question", "").strip()
            rag_lookup[q] = r.get("actual_response", "")
        print(f"Loaded {len(rag_lookup)} RAG answers from {args.results.name}")
    else:
        print(f"RAG results not found at {args.results}.")
        print("Run evaluate_pipeline.py first, then re-run this script.")
        print("Continuing with 70B-only analysis...")

    # -- Score each question ---------------------------------------------------
    records = []
    diff_buckets: dict[str, list] = {"easy": [], "medium": [], "hard": []}

    for item in golden:
        q          = item["question"].strip()
        difficulty = item.get("difficulty", "easy")
        ref_answer = item.get("answer", "")
        rag_answer = rag_lookup.get(q, "")

        score_70b = physics_score(ref_answer, model_label="NVIDIA 70B")
        score_rag = physics_score(rag_answer, model_label="RAG 0.5B") if rag_answer else None

        record = {
            "id":         item.get("id", ""),
            "difficulty": difficulty,
            "question":   q,
            "score_70b":  score_70b,
            "score_rag":  score_rag,
        }
        records.append(record)
        diff_buckets.setdefault(difficulty, []).append(record)

    # -- Aggregate -------------------------------------------------------------
    def aggregate(recs, key):
        subset = [r for r in recs if r[key] is not None]
        if not subset:
            return None
        n = len(subset)
        return {
            "n":             n,
            "parseable":     sum(1 for r in subset if r[key]["parseable"]),
            "dimensional":   sum(1 for r in subset if r[key]["dimensional"]),
            "numerical":     sum(1 for r in subset if r[key]["numerical"]),
            "coverage":      sum(1 for r in subset if r[key]["coverage"]),
            "avg_coverage":  sum(r[key].get("coverage_frac", 0.0) for r in subset) / n,
            "avg_score":     sum(r[key]["total"] for r in subset) / n,
        }

    total = len(records)
    agg_70b = aggregate(records, "score_70b")
    agg_rag  = aggregate(records, "score_rag") if rag_lookup else None

    # -- Per-question verbose breakdown ----------------------------------------
    print(f"\n{'='*72}")
    print("  PER-QUESTION BREAKDOWN  (physics checker reasoning)")
    print(f"{'='*72}")

    for rec in records:
        s70  = rec["score_70b"]
        srag = rec["score_rag"]
        print(f"\n[{rec['id']}] [{rec['difficulty'].upper()}]  {rec['question'][:75]}")
        print(f"  {'-'*68}")

        for label, score in [("NVIDIA 70B", s70), ("RAG 0.5B", srag)]:
            if score is None:
                print(f"  {label}: no answer available")
                continue
            exp = score["explanation"]
            icons = {
                "parseable":   "[OK]" if score["parseable"]   else "[X]",
                "dimensional": "[OK]" if score["dimensional"] else "[X]",
                "numerical":   "[OK]" if score["numerical"]   else "[X]",
                "coverage":    "[OK]" if score["coverage"]    else "[X]",
            }
            print(f"  {label}  [{score['total']}/4]  eq: {score['equation'][:55] or '—'}")
            print(f"    {icons['parseable']} Symbolic   : {exp['symbolic']['reason'][:90]}")
            print(f"    {icons['dimensional']} Dimensional: {exp['dimensional']['reason'][:90]}")
            print(f"    {icons['numerical']} Numerical  : {exp['numerical']['reason'][:90]}")
            print(f"    {icons['coverage']} Coverage   : {exp['coverage']['reason'][:90]}")
            if exp["feedback_hint"] and any(not score[k] for k in ("parseable","dimensional","numerical","coverage")):
                hint_lines = exp["feedback_hint"].replace("\n", "\n      ")
                print(f"    → Feedback hint: {hint_lines[:200]}")

    # -- Aggregate summary -----------------------------------------------------
    SEP = "-" * 72

    print(f"\n{SEP}")
    print("  PHYSICS VALIDATOR COMPARISON: NVIDIA 70B  vs  RAG 0.5B")
    print(SEP)

    def print_col(label, agg, total):
        if agg is None:
            print(f"  {label:<28}  (no data)")
            return
        n = agg["n"]
        print(f"  {label:<28}")
        print(f"    Equation parseable :  {_pct(agg['parseable'],   n)}")
        print(f"    Dimensional pass   :  {_pct(agg['dimensional'], n)}")
        print(f"    Numerical pass     :  {_pct(agg['numerical'],   n)}")
        print(f"    Full symbol coverage: {_pct(agg['coverage'],    n)}")
        print(f"    Avg symbol coverage:  {agg.get('avg_coverage', 0.0)*100:.1f}%  (partial credit)")
        print(f"    Avg physics score  :  {agg['avg_score']:.2f} / 4.00  "
              f"  {_bar(agg['avg_score'])}")

    print()
    print_col("NVIDIA Llama-3.1-70B", agg_70b, total)
    print()
    print_col("RAG Qwen-0.5B", agg_rag, total)

    # -- By difficulty ---------------------------------------------------------
    print(f"\n{SEP}")
    print("  BREAKDOWN BY DIFFICULTY")
    print(SEP)
    header = f"  {'Difficulty':<10}  {'N':>4}  {'70B avg':>9}  {'RAG avg':>9}  {'Δ (RAG-70B)':>12}  {'Winner'}"
    print(header)
    print("  " + "-" * 68)

    for diff in ["easy", "medium", "hard"]:
        recs = diff_buckets.get(diff, [])
        if not recs:
            continue
        a70 = aggregate(recs, "score_70b")
        ar  = aggregate(recs, "score_rag") if rag_lookup else None

        avg70 = a70["avg_score"] if a70 else 0.0
        avgr  = ar["avg_score"]  if ar  else None

        if avgr is not None:
            delta  = avgr - avg70
            winner = "RAG [OK]" if delta > 0.1 else ("70B [OK]" if delta < -0.1 else "Tie ~")
            print(f"  {diff:<10}  {len(recs):>4}  {avg70:>9.2f}  {avgr:>9.2f}  "
                  f"{delta:>+12.2f}  {winner}")
        else:
            print(f"  {diff:<10}  {len(recs):>4}  {avg70:>9.2f}  {'—':>9}")

    # -- Key finding -----------------------------------------------------------
    print(f"\n{SEP}")
    if agg_70b and agg_rag:
        delta_overall = agg_rag["avg_score"] - agg_70b["avg_score"]
        print(f"  Overall RAG Δ vs 70B: {delta_overall:+.2f} / 4.00")
        if delta_overall > 0:
            print("  → RAG 0.5B produces more physics-consistent equations")
            print("    despite being 140× smaller than the 70B model.")
        elif delta_overall < -0.1:
            print("  → 70B scores higher on physics consistency.")
            print("    Consider expanding symbol dictionary further.")
        else:
            print("  → Physics consistency is comparable between models.")
            print("    RAG advantage lies in grounding (retrieval quality).")
    print(SEP)

    # -- Save output -----------------------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_data = {
        "total_questions": total,
        "aggregate_70b":   agg_70b,
        "aggregate_rag":   agg_rag,
        "by_difficulty": {
            diff: {
                "70b": aggregate(diff_buckets.get(diff, []), "score_70b"),
                "rag": aggregate(diff_buckets.get(diff, []), "score_rag") if rag_lookup else None,
            }
            for diff in ["easy", "medium", "hard"]
        },
        "per_question": records,
    }
    args.out.write_text(json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Full breakdown saved → {args.out}")


if __name__ == "__main__":
    main()
