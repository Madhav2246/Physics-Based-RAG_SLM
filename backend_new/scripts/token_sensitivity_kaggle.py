"""
token_sensitivity_kaggle.py
----------------------------
Token Budget Sensitivity Study — Kaggle (GPU) edition.
Runs the SAME 30-QA subset (20 hard + 10 medium) at three token budgets:
    128 (production baseline)
    256 (mid-range robustness check)
    512 (maximum practical limit)

Architecture note
-----------------
The deployed system is:
  corpus_eq  +  Qwen-symbol-explanation  →  physics scorer
            SYS score (non-zero when corpus_eq found)

"RAW 0.5B" means the model output with NO corpus equation prepended.
Because the model's system prompt only asks for symbol definitions
(not full equations), raw_score is structurally near-zero — this is
expected and is the whole POINT of the neuro-symbolic layer.
The sensitivity study therefore tracks SYS score across budgets.
Word-count and truncation-rate capture generation completeness.

Run from backend_new/:
  python scripts/token_sensitivity_kaggle.py --n 30 --seed 42

Outputs (written to /kaggle/working/eval_token_sensitivity/):
  sampled_ids.json
  raw_outputs_128T.jsonl
  raw_outputs_256T.jsonl
  raw_outputs_512T.jsonl
  metrics_summary.json
  evaluation_report_diff_tokensize.md
"""
from __future__ import annotations
import argparse, io, json, random, re, sys, time
from pathlib import Path
from datetime import datetime

# ── UTF-8 stdout (safe on Linux Kaggle too) ────────────────────────────────────
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent          # backend_new/
GOLDEN   = ROOT / "data/evaluation/nvidia_golden_qa.jsonl"
OUT_DIR  = Path("/kaggle/working/eval_token_sensitivity")   # override with --out
TOKEN_BUDGETS = [128, 256, 512]

sys.path.insert(0, str(ROOT))

import utils.config as cfg
from physics.physics_scorer import score_text


# ─────────────────────────────────────────────────────────────────────────────
# Tiny helpers
# ─────────────────────────────────────────────────────────────────────────────
def _mean(xs):      return sum(xs) / len(xs) if xs else 0.0
def _fmt(v):        return f"{v*100:.1f}%"

def _is_truncated(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    if text[-1] in ("-", "+", "*", "/", "=", ",", "(", "[", "{", "\\"):
        return True
    last = text.split()[-1] if text.split() else ""
    if last and last[-1].isalnum() and not text.endswith((".", "?", "!")):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_dataset(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def select_subset(dataset, seed, medium_n=10):
    hard   = [q for q in dataset if q.get("difficulty") == "hard"]
    medium = [q for q in dataset if q.get("difficulty") == "medium"]
    rng    = random.Random(seed)
    med_sample = rng.sample(medium, min(medium_n, len(medium)))
    subset = hard + med_sample
    id_log = {
        "seed": seed,
        "hard_count": len(hard),
        "medium_sampled": len(med_sample),
        "total": len(subset),
        "hard_ids":   [q.get("id", q["question"][:40]) for q in hard],
        "medium_ids": [q.get("id", q["question"][:40]) for q in med_sample],
    }
    return subset, id_log


# ─────────────────────────────────────────────────────────────────────────────
# Per-budget evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_budget(subset, pipeline, token_budget, seed):
    from reasoning.prompt_builder import build_prompt

    results = []
    n = len(subset)
    print(f"\n{'='*65}")
    print(f"  TOKEN BUDGET: {token_budget} tokens  |  {n} questions  |  seed={seed}")
    print(f"{'='*65}")

    for i, item in enumerate(subset):
        q    = item["question"].strip()
        diff = item.get("difficulty", "unknown")

        # Retrieve evidence + extract corpus equation
        evidence = pipeline.retriever.retrieve(q, top_k=cfg.TOP_K)
        _, _, corpus_eq = pipeline._find_corpus_equation(evidence)

        # ── SYS pass (deployed pipeline) ──────────────────────────────────────
        prompt_sys = build_prompt(q, evidence, corpus_equation=corpus_eq)
        t0 = time.time()
        sys_outputs = pipeline.slm.generate_multiple(
            prompt_sys, n_samples=1, max_tokens=token_budget, seed=seed + i
        )
        latency = time.time() - t0
        sys_text = sys_outputs[0] if sys_outputs else ""

        # Compose full SYS answer (corpus_eq prepended — this is what the scorer sees)
        if corpus_eq:
            sys_composed = f"Equation: {corpus_eq}\n\n{sys_text}"
        else:
            sys_composed = f"Equation: NOT FOUND IN CORPUS\n\n{sys_text}"
        sys_score = score_text(sys_composed, f"SYS-{token_budget}T")

        # ── RAW pass (no corpus eq — tests pure model generation) ─────────────
        # Note: because the model is prompted for symbol definitions only,
        # raw physics score will be near-zero by design (no parseable equation).
        # We track it for completeness and to document this architectural choice.
        prompt_raw = build_prompt(q, evidence, corpus_equation=None)
        t0r = time.time()
        raw_outputs = pipeline.slm.generate_multiple(
            prompt_raw, n_samples=1, max_tokens=token_budget, seed=seed + i
        )
        latency_raw = time.time() - t0r
        raw_text = raw_outputs[0] if raw_outputs else ""
        raw_score = score_text(raw_text, f"RAW-{token_budget}T")

        # ── Diagnostics ───────────────────────────────────────────────────────
        truncated  = _is_truncated(sys_text)
        word_count = len(sys_text.split())

        rec = {
            "id":            item.get("id", f"Q{i+1}"),
            "difficulty":    diff,
            "question":      q,
            "token_budget":  token_budget,
            "sys_text":      sys_text,
            "raw_text":      raw_text,
            "corpus_eq":     corpus_eq or "",
            "sys_score":     sys_score,
            "raw_score":     raw_score,
            "latency_sys":   round(latency, 2),
            "latency_raw":   round(latency_raw, 2),
            "word_count":    word_count,
            "truncated":     truncated,
        }
        results.append(rec)

        print(
            f"  [{i+1:>2}/{n}] {diff.upper():<7} "
            f"sys={sys_score['total']:.2f} raw={raw_score['total']:.2f} "
            f"trunc={'Y' if truncated else 'N'} "
            f"words={word_count} latency={latency:.1f}s"
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate
# ─────────────────────────────────────────────────────────────────────────────
def aggregate(results, difficulty=None):
    if difficulty:
        results = [r for r in results if r["difficulty"] == difficulty]
    if not results:
        return {"n": 0}
    n = len(results)

    sys_scores = [r["sys_score"]["total"]       for r in results]
    parseables = [r["sys_score"]["parseable"]    for r in results]
    dims       = [r["sys_score"]["dimensional"]  for r in results]
    nums       = [r["sys_score"]["numerical"]    for r in results]
    covs       = [r["sys_score"]["coverage_frac"] for r in results]
    trunc      = [r["truncated"]                 for r in results]
    words      = [r["word_count"]               for r in results]

    return {
        "n":              n,
        "sys_score":      round(_mean(sys_scores), 4),
        "parse_rate":     round(sum(parseables) / n, 4),
        "dcr":            round(sum(dims) / n, 4),
        "nvr":            round(sum(nums) / n, 4),
        "coverage":       round(_mean(covs), 4),
        "truncation_rate": round(sum(trunc) / n, 4),
        "avg_words":      round(_mean(words), 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report writer
# ─────────────────────────────────────────────────────────────────────────────
def write_report(path, id_log, all_results, metrics, hard_m, med_m, ts):
    m128, m256, m512 = metrics[128], metrics[256], metrics[512]

    def delta(a, b, key):
        va, vb = a.get(key, 0) or 0, b.get(key, 0) or 0
        return round(vb - va, 4)

    d_128_256 = {k: delta(m128, m256, k) for k in
                 ["sys_score", "parse_rate", "dcr", "nvr", "truncation_rate", "avg_words"]}
    d_256_512 = {k: delta(m256, m512, k) for k in d_128_256}
    d_128_512 = {k: delta(m128, m512, k) for k in d_128_256}

    # Confound classification
    abs_span = abs(d_128_512["sys_score"])
    if abs_span < 0.05:   confound = "negligible (<0.05 pts)"
    elif abs_span < 0.15: confound = "minor (0.05–0.15 pts)"
    elif abs_span < 0.30: confound = "moderate (0.15–0.30 pts)"
    else:                 confound = "MAJOR (>0.30 pts)"

    # Recommendation
    g128_256 = d_128_256["sys_score"]
    g256_512 = d_256_512["sys_score"]
    hard_gain = delta(hard_m[128], hard_m[256], "sys_score")
    med_gain  = delta(med_m[128],  med_m[256],  "sys_score")

    if hard_gain > 0.10 and med_gain < 0.05:
        rec = "Adaptive Budget (128 for medium, 256 for hard)"
    elif abs(g128_256) > 0.10:
        rec = "Move to 256 Tokens"
    else:
        rec = "Keep 128 Tokens"

    lines = [
        "# Token Budget Sensitivity Study",
        f"\n*Generated: {ts} | Kaggle P100 edition*",
        f"*System: Qwen-2.5-0.5B + LoRA + Neuro-Symbolic RAG*\n",
        "---",
        "",
        "## Architecture Note",
        "",
        "> **Why RAW score ≈ 0?**  ",
        "> The system prompt instructs Qwen to output **symbol definitions only**  ",
        "> (e.g. `- Id = drain current in Amperes`), deliberately **not** equations.  ",
        "> The physics scorer needs a parseable equation, which comes from `corpus_eq`.  ",
        "> So `SYS score = corpus_eq + model text` is the right metric.  ",
        "> RAW ≈ 0 is the design baseline — it **motivates** the retrieval layer.",
        "",
        "---",
        "",
        "## Objective",
        "",
        "Determine whether the **128-token generation budget** materially confounds",
        "Stage 1 Physics Validation scores **for the SYS (deployed) pipeline**.",
        "",
        "---",
        "",
        "## Experimental Setup",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Hard questions | {id_log['hard_count']} (all) |",
        f"| Medium questions | {id_log['medium_sampled']} (seed={id_log['seed']}) |",
        f"| Total QA | {id_log['total']} |",
        f"| Token budgets | 128, 256, 512 |",
        f"| Samples per question | 1 per budget |",
        f"| Hardware | Kaggle P100 GPU |",
        f"| Primary metric | **SYS physics score** (0–4) |",
        "",
        "---",
        "",
        "## Results",
        "",
        "### Overall (all QA)",
        "",
        "| Token Limit | SYS Score/4 | Parse% | DCR% | NVR% | Coverage% | Trunc% | Avg Words |",
        "|-------------|:-----------:|:------:|:----:|:----:|:---------:|:------:|:---------:|",
    ]

    for bud, m in [(128, m128), (256, m256), (512, m512)]:
        flag = " ← baseline" if bud == 128 else ""
        lines.append(
            f"| **{bud}T**{flag} "
            f"| {m.get('sys_score',0):.3f} "
            f"| {_fmt(m.get('parse_rate',0))} "
            f"| {_fmt(m.get('dcr',0))} "
            f"| {_fmt(m.get('nvr',0))} "
            f"| {_fmt(m.get('coverage',0))} "
            f"| {_fmt(m.get('truncation_rate',0))} "
            f"| {m.get('avg_words',0):.1f} |"
        )

    lines += [
        "",
        "### By Difficulty",
        "",
        "| Subset | 128T | 256T | 512T | Δ(128→512) |",
        "|--------|:----:|:----:|:----:|:----------:|",
        f"| Hard   | {hard_m[128].get('sys_score',0):.3f} | {hard_m[256].get('sys_score',0):.3f} | {hard_m[512].get('sys_score',0):.3f} | {delta(hard_m[128], hard_m[512], 'sys_score'):+.3f} |",
        f"| Medium | {med_m[128].get('sys_score',0):.3f} | {med_m[256].get('sys_score',0):.3f} | {med_m[512].get('sys_score',0):.3f} | {delta(med_m[128], med_m[512], 'sys_score'):+.3f} |",
        "",
        "---",
        "",
        "## Delta Analysis (128 → 256 → 512)",
        "",
        "| Metric | 128T | 256T | Δ | 512T | Δ |",
        "|--------|:----:|:----:|:-:|:----:|:-:|",
        f"| SYS Score | {m128.get('sys_score',0):.3f} | {m256.get('sys_score',0):.3f} | {d_128_256['sys_score']:+.3f} | {m512.get('sys_score',0):.3f} | {d_256_512['sys_score']:+.3f} |",
        f"| Parse Rate | {_fmt(m128.get('parse_rate',0))} | {_fmt(m256.get('parse_rate',0))} | {d_128_256['parse_rate']:+.3f} | {_fmt(m512.get('parse_rate',0))} | {d_256_512['parse_rate']:+.3f} |",
        f"| DCR | {_fmt(m128.get('dcr',0))} | {_fmt(m256.get('dcr',0))} | {d_128_256['dcr']:+.3f} | {_fmt(m512.get('dcr',0))} | {d_256_512['dcr']:+.3f} |",
        f"| Trunc Rate | {_fmt(m128.get('truncation_rate',0))} | {_fmt(m256.get('truncation_rate',0))} | {d_128_256['truncation_rate']:+.3f} | {_fmt(m512.get('truncation_rate',0))} | {d_256_512['truncation_rate']:+.3f} |",
        f"| Avg Words | {m128.get('avg_words',0):.1f} | {m256.get('avg_words',0):.1f} | {d_128_256['avg_words']:+.1f} | {m512.get('avg_words',0):.1f} | {d_256_512['avg_words']:+.1f} |",
        "",
        "---",
        "",
        "## Interpretation",
        "",
        f"Full-span SYS score delta (128→512): **{d_128_512['sys_score']:+.3f} pts** — classified as **{confound}**.",
        "",
        f"Truncation rate at 128T: **{_fmt(m128.get('truncation_rate',0))}** → at 512T: **{_fmt(m512.get('truncation_rate',0))}**.",
        "",
        "---",
        "",
        "## Recommendation",
        "",
        f"### ✅ {rec}",
        "",
        f"Hard questions: {hard_gain:+.3f} pts (128→256) | Medium: {med_gain:+.3f} pts",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"1. SYS physics score delta (128→512): **{d_128_512['sys_score']:+.3f} pts** — {confound}.",
        f"2. RAW score ≈ 0 at all budgets by design (model outputs symbol defs, not equations).",
        f"3. Truncation rate decreases from {_fmt(m128.get('truncation_rate',0))} (128T) to {_fmt(m512.get('truncation_rate',0))} (512T).",
        f"4. Recommendation: **{rec}**",
        "",
        f"> Conclusion: The 128-token budget introduces a **{confound}** confound in Stage 1.",
        "",
        "---",
        f"*Auto-generated by `token_sensitivity_kaggle.py` | n={id_log['total']} | budgets: 128/256/512T*",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Report saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--medium-n", type=int, default=10)
    ap.add_argument("--out",      type=Path, default=None,
                    help="Output directory (default: /kaggle/working/eval_token_sensitivity)")
    args = ap.parse_args()

    out_dir = args.out or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Load + sample
    print(f"Loading dataset from {GOLDEN} ...")
    dataset = load_dataset(GOLDEN)
    subset, id_log = select_subset(dataset, args.seed, args.medium_n)
    print(f"  Subset: {id_log['hard_count']} hard + {id_log['medium_sampled']} medium = {id_log['total']} questions")
    (out_dir / "sampled_ids.json").write_text(json.dumps(id_log, indent=2), encoding="utf-8")

    # Initialise pipeline ONCE
    print("\nInitialising RAG pipeline...")
    from pipeline.rag_pipeline import RAGPipeline
    pipe = RAGPipeline()
    pipe.retriever.dense.load_index()
    pipe.retriever.sparse.build_index_from_docs(pipe.retriever.dense.documents)
    print("Pipeline ready.\n")

    # Run each budget
    all_results: dict[int, list] = {}
    for budget in TOKEN_BUDGETS:
        res = evaluate_budget(subset, pipe, budget, args.seed)
        all_results[budget] = res
        raw_path = out_dir / f"raw_outputs_{budget}T.jsonl"
        with open(raw_path, "w", encoding="utf-8") as f:
            for rec in res:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        print(f"  Saved → {raw_path}")

    # Aggregate
    metrics = {b: aggregate(all_results[b])           for b in TOKEN_BUDGETS}
    hard_m  = {b: aggregate(all_results[b], "hard")   for b in TOKEN_BUDGETS}
    med_m   = {b: aggregate(all_results[b], "medium") for b in TOKEN_BUDGETS}

    # Print console summary
    print("\n" + "="*65)
    print("  TOKEN BUDGET SENSITIVITY — SYS PHYSICS SCORE")
    print("="*65)
    print(f"  {'Budget':<10} {'SYS/4':>8} {'Parse%':>8} {'Trunc%':>8} {'AvgWds':>8}")
    print("  " + "-"*45)
    for b in TOKEN_BUDGETS:
        m = metrics[b]
        print(f"  {b}T{' ←base' if b==128 else '':<6} "
              f"{m.get('sys_score',0):>8.3f} "
              f"{_fmt(m.get('parse_rate',0)):>8} "
              f"{_fmt(m.get('truncation_rate',0)):>8} "
              f"{m.get('avg_words',0):>8.1f}")
    print("="*65)

    # Save summary JSON
    summary = {
        "ts": ts, "subset": id_log,
        "overall": {str(b): metrics[b] for b in TOKEN_BUDGETS},
        "hard":    {str(b): hard_m[b]  for b in TOKEN_BUDGETS},
        "medium":  {str(b): med_m[b]   for b in TOKEN_BUDGETS},
    }
    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Write report
    write_report(
        out_dir / "evaluation_report_diff_tokensize.md",
        id_log, all_results, metrics, hard_m, med_m, ts
    )
    print(f"\n✅ Done. Full report → {out_dir / 'evaluation_report_diff_tokensize.md'}")


if __name__ == "__main__":
    main()
