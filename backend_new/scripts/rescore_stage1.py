"""
scripts/rescore_stage1.py
--------------------------
Re-score Stage-1 using the fixed new_checker (v2) against the raw model
outputs stored in answers_dump.jsonl.

Fixes applied in new_checker v2 (vs the original run):
  (1) Stricter letter-soup: known_multis==0 + singles>=3  OR  unknown_singles>=4
      Catches corpus garbage like D*V=0, C*E*h=3, E*I*n=E*I*J*Y
  (2) Coverage uses LHS+RHS combined free symbols (was RHS-only)
      Fixes `F*O = 2` getting coverage=1.0 when LHS symbols are unknown
  (3) Trivial-dimensionless guard: both sides {} + unknowns → UNRESOLVABLE
      Stops unknown symbols defaulting to {} and faking a dimensional pass

Input:
  backend_new/data/evaluation_new/stage1_new_separate_eval/answers_dump.jsonl

Output:
  backend_new/data/evaluation_new/stage1_rescored.json
"""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path
from collections import defaultdict

# Make sure backend_new is on the path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physics.new_checker import score_text, score_equation

# ── paths ────────────────────────────────────────────────────────────────────
DUMP    = ROOT / "data/evaluation_new/stage1_new_separate_eval/answers_dump.jsonl"
OUT     = ROOT / "data/evaluation_new/stage1_rescored.json"

# ── helpers ──────────────────────────────────────────────────────────────────

def best_of(scores: list[dict]) -> dict:
    """Return the highest-scoring dict; tie-break on coverage_frac."""
    return max(scores, key=lambda s: (s["total"], s["coverage_frac"]))


def score_sys(corpus_eq: str | None, raw_samples: list[str]) -> dict:
    """
    SYS pipeline best-of-N:
      candidates = score each raw_sample text + score corpus_eq (if any)
    """
    candidates = [score_text(t) for t in raw_samples if t and t.strip() not in ("-", "")]
    if corpus_eq and corpus_eq.strip():
        candidates.append(score_equation(corpus_eq))
    if not candidates:
        return score_text("")
    return best_of(candidates)


def score_raw_bon(raw_samples: list[str]) -> dict:
    """RAW best-of-N: score each sample text, pick best."""
    candidates = [score_text(t) for t in raw_samples if t and t.strip() not in ("-", "")]
    if not candidates:
        return score_text("")
    return best_of(candidates)


def score_raw_mean(raw_samples: list[str]) -> float:
    """RAW mean total across all samples (including blanks)."""
    scores = [score_text(t)["total"] for t in raw_samples]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


# ── stats helpers ─────────────────────────────────────────────────────────────

def _dcr_cond(score_dicts: list[dict]) -> float | None:
    ok = checkable = 0
    for s in score_dicts:
        if s["parseable"] and s["coverage_frac"] >= 0.999:
            checkable += 1
            if s["dimensional"]:
                ok += 1
    return round(100 * ok / checkable, 1) if checkable else None


def _nvr_cond(score_dicts: list[dict]) -> float | None:
    ok = evaluable = 0
    for s in score_dicts:
        if s["parseable"]:
            nm = s["num_msg"]
            if "[OK]" in nm:
                ok += 1; evaluable += 1
            elif "Unresolved" in nm or "skipped" in nm.lower() or "Unresolvable" in nm:
                pass
            else:
                evaluable += 1
    return round(100 * ok / evaluable, 1) if evaluable else None


def summarise(score_dicts: list[dict]) -> dict:
    n = len(score_dicts)
    if n == 0:
        return {}
    avg   = round(sum(s["total"]      for s in score_dicts) / n, 4)
    parse = round(100 * sum(s["parseable"]   for s in score_dicts) / n, 1)
    dim   = round(100 * sum(s["dimensional"] for s in score_dicts) / n, 1)
    num   = round(100 * sum(s["numerical"]   for s in score_dicts) / n, 1)
    cov   = round(100 * sum(s["coverage"]    for s in score_dicts) / n, 1)
    cov_f = round(100 * sum(s["coverage_frac"] for s in score_dicts) / n, 1)
    return {
        "n": n,
        "avg_score":   avg,
        "parseable":   parse,
        "dimensional": dim,
        "numerical":   num,
        "coverage":    cov,
        "coverage_frac_mean": cov_f,
        "nvr_conditional": _nvr_cond(score_dicts),
        "dcr_conditional": _dcr_cond(score_dicts),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    records = []
    with open(DUMP, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    per_question = []
    # Group by difficulty for summary
    by_diff: dict[str, dict[str, list]] = defaultdict(lambda: {"70B": [], "SYS": [], "RAW": [], "RAW_mean": []})

    for rec in records:
        qid        = rec["id"]
        diff       = rec["difficulty"]
        question   = rec["question"]
        corpus_eq  = rec.get("corpus_eq")
        raw_samples= rec.get("raw_samples", [])
        b70_text   = rec.get("b70_text", "")

        s_70b  = score_text(b70_text)
        s_sys  = score_sys(corpus_eq, raw_samples)
        s_raw  = score_raw_bon(raw_samples)
        raw_mean = score_raw_mean(raw_samples)

        per_question.append({
            "id": qid,
            "difficulty": diff,
            "question": question,
            "corpus_eq": corpus_eq,
            "raw_mean_total": raw_mean,
            "raw_repr": s_raw,
            "sys":  s_sys,
            "b70":  s_70b,
        })

        by_diff[diff]["70B"].append(s_70b)
        by_diff[diff]["SYS"].append(s_sys)
        by_diff[diff]["RAW"].append(s_raw)
        by_diff[diff]["RAW_mean"].append(raw_mean)

    # Overall lists
    all_70b = [q["b70"]  for q in per_question]
    all_sys = [q["sys"]  for q in per_question]
    all_raw = [q["raw_repr"] for q in per_question]
    all_raw_mean = [q["raw_mean_total"] for q in per_question]

    by_diff_summary = {}
    for d, grp in by_diff.items():
        by_diff_summary[d] = {
            "70B":         summarise(grp["70B"]),
            "SYS":         summarise(grp["SYS"]),
            "RAW_bestofN": summarise(grp["RAW"]),
            "RAW_mean_total": round(sum(grp["RAW_mean"]) / len(grp["RAW_mean"]), 4),
        }

    result = {
        "summary": {
            "checker":       "new_checker_v2",
            "fixes":         ["letter_soup_stricter", "coverage_lhs_rhs", "dim_trivial_guard"],
            "samples_per_q": 5,
            "n":             len(per_question),
            "overall": {
                "70B":         summarise(all_70b),
                "SYS":         summarise(all_sys),
                "RAW_bestofN": summarise(all_raw),
                "RAW_mean_total": round(sum(all_raw_mean) / len(all_raw_mean), 4),
            },
            "by_difficulty": by_diff_summary,
        },
        "per_question": per_question,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"Wrote {OUT}")
    print("\n=== OVERALL COMPARISON ===")
    print(f"{'System':<14} {'old avg':>8} {'new avg':>8}  {'parse%':>7}  {'dim%':>6}  {'DCRcond':>8}")
    old = {"70B": 1.292, "SYS": 1.227, "RAW": 0.555}
    for key in ("70B", "SYS", "RAW_bestofN"):
        label = key.replace("_bestofN", "")
        s = result["summary"]["overall"][key]
        print(f"  {label:<12} {old.get(label, '?'):>8}  {s['avg_score']:>8.3f}  "
              f"{s['parseable']:>6.1f}%  {s['dimensional']:>5.1f}%  "
              f"{str(s['dcr_conditional']):>8}")

    print("\n=== BY DIFFICULTY ===")
    for d in ("easy", "medium", "hard"):
        if d not in by_diff_summary:
            continue
        print(f"\n  [{d.upper()}]")
        for key in ("70B", "SYS", "RAW_bestofN"):
            s = by_diff_summary[d][key]
            print(f"    {key:<14} avg={s['avg_score']:.3f}  parse={s['parseable']:.1f}%  "
                  f"dim={s['dimensional']:.1f}%  DCR={s['dcr_conditional']}")


if __name__ == "__main__":
    main()
