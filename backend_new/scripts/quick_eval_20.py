"""
quick_eval_20.py
-----------------
Runs the RAG pipeline on the first 20 questions from nvidia_golden_qa.jsonl,
saves results to eval_results_quick20.jsonl, then immediately runs the
physics validator comparison (70B vs RAG) on those 20 questions and prints
the summary table.

Run from the backend/ directory:
  python scripts/quick_eval_20.py
"""

import io
import json
import sys
import time
import datetime
from pathlib import Path

# Force UTF-8 on Windows (avoids cp1252 crashes on unicode symbols)
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

GOLDEN_PATH  = PROJECT_ROOT / "data/evaluation/nvidia_golden_qa.jsonl"
RESULTS_PATH = PROJECT_ROOT / "data/evaluation/eval_results_quick20.jsonl"
COMPARE_OUT  = PROJECT_ROOT / "data/evaluation/physics_comparison_quick20.json"
N = 20  # questions to evaluate

# ── 1. Load first N questions ─────────────────────────────────────────────────
print(f"Loading first {N} questions from {GOLDEN_PATH.name}...")
dataset = []
with open(GOLDEN_PATH, encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= N:
            break
        if line.strip():
            dataset.append(json.loads(line))
print(f"  Loaded {len(dataset)} questions  ({dataset[0]['difficulty']}…{dataset[-1]['difficulty']})\n")

# ── 2. Init pipeline ──────────────────────────────────────────────────────────
print("Initialising RAG Pipeline (loading model weights)...")
from pipeline.rag_pipeline import RAGPipeline
pipeline = RAGPipeline()
print("Loading vector indexes...")
pipeline.retriever.dense.load_index()
pipeline.retriever.sparse.build_index_from_docs(pipeline.retriever.dense.documents)
print("Pipeline ready.\n" + "="*60)

# ── 3. Run evaluation ─────────────────────────────────────────────────────────
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, "w", encoding="utf-8") as out_f:
    for i, item in enumerate(dataset):
        question = item["question"]
        expected = item.get("answer", "")
        print(f"\n[{i+1}/{N}] ({item['difficulty'].upper()}) {question[:80]}")

        t0 = time.time()
        result = pipeline.answer(question)
        latency = time.time() - t0

        record = {
            "question":             question,
            "expected_answer":      expected,
            "actual_response":      result["response"],
            "semantic_similarity":  result["semantic_similarity"],
            "confidence_score":     result["confidence_score"],
            "confidence_label":     result["confidence_label"],
            "symbolic_validation":  result["symbolic_validation"],
            "dimension_validation": result["dimension_validation"],
            "numerical_validation": result["numerical_validation"],
            "latency_sec":          latency,
        }
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_f.flush()

        sym_short = result["symbolic_validation"][:60]
        print(f"  Conf   : {result['confidence_label']} ({result['confidence_score']:.3f})")
        print(f"  Symbolic: {sym_short}")
        print(f"  Latency : {latency:.1f}s")

print(f"\n{'='*60}")
print(f"Evaluation done! Results → {RESULTS_PATH}\n")

# ── 4. Physics-validator comparison (shared Tier-2 scorer) ───────────────────
from physics.physics_scorer import score_text as physics_score

# Build RAG lookup
rag_lookup = {}
with open(RESULTS_PATH, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            rag_lookup[r["question"].strip()] = r["actual_response"]

def pct(n, total):
    return f"{n}/{total} ({100*n/total:.0f}%)" if total else "0/0"

def aggregate(recs, key):
    subset = [r for r in recs if r[key] is not None]
    if not subset: return None
    n = len(subset)
    return {
        "n": n,
        "parseable":    sum(1 for r in subset if r[key]["parseable"]),
        "dimensional":  sum(1 for r in subset if r[key]["dimensional"]),
        "numerical":    sum(1 for r in subset if r[key]["numerical"]),
        "coverage":     sum(1 for r in subset if r[key]["coverage"]),
        "avg_coverage": sum(r[key].get("coverage_frac", 0.0) for r in subset) / n,
        "avg_score":    sum(r[key]["total"] for r in subset) / n,
    }

records = []
diff_buckets: dict[str, list] = {}
for item in dataset:
    q   = item["question"].strip()
    s70 = physics_score(item.get("answer", ""), "70B")
    sr  = physics_score(rag_lookup.get(q, ""), "RAG") if q in rag_lookup else None
    rec = {"difficulty": item.get("difficulty","easy"), "score_70b": s70, "score_rag": sr}
    records.append(rec)
    diff_buckets.setdefault(item.get("difficulty","easy"), []).append(rec)

agg70  = aggregate(records, "score_70b")
agg_rag = aggregate(records, "score_rag")

SEP = "-"*60
print(f"\n{SEP}")
print("  PHYSICS VALIDATOR COMPARISON  (20-question sample)")
print(f"  NVIDIA Llama-3.1-70B  vs  RAG Qwen-0.5B")
print(SEP)

def print_col(label, agg):
    if agg is None: print(f"  {label}: (no data)"); return
    n = agg["n"]
    print(f"\n  {label}")
    print(f"    Equation parseable  : {pct(agg['parseable'],   n)}")
    print(f"    Dimensional pass    : {pct(agg['dimensional'], n)}")
    print(f"    Numerical pass      : {pct(agg['numerical'],   n)}")
    print(f"    Full symbol coverage: {pct(agg['coverage'],    n)}")
    print(f"    Avg symbol coverage : {agg.get('avg_coverage', 0.0)*100:.1f}%  (partial credit)")
    print(f"    Avg physics score   : {agg['avg_score']:.2f} / 4.00")

print_col("NVIDIA Llama-3.1-70B", agg70)
print_col("RAG Qwen-0.5B       ", agg_rag)

print(f"\n{SEP}")
print("  BY DIFFICULTY")
print(f"{'  Difficulty':<14} {'70B avg':>9} {'RAG avg':>9} {'Delta':>8}  Winner")
print("  " + "-"*52)
for diff in ["easy", "medium", "hard"]:
    recs = diff_buckets.get(diff, [])
    if not recs: continue
    a70 = aggregate(recs, "score_70b")
    ar  = aggregate(recs, "score_rag")
    avg70 = a70["avg_score"] if a70 else 0.0
    avgr  = ar["avg_score"]  if ar  else None
    if avgr is not None:
        delta  = avgr - avg70
        winner = "RAG ✓" if delta > 0.05 else ("70B ✓" if delta < -0.05 else "Tie")
        print(f"  {diff:<12} {avg70:>9.2f} {avgr:>9.2f} {delta:>+8.2f}  {winner}")
    else:
        print(f"  {diff:<12} {avg70:>9.2f} {'—':>9}")

if agg70 and agg_rag:
    delta = agg_rag["avg_score"] - agg70["avg_score"]
    print(f"\n  Overall Δ : {delta:+.2f}  ({'RAG wins ✓' if delta>0 else '70B wins'})")
print(SEP)
print(f"\nFull comparison JSON → {COMPARE_OUT}\n")
COMPARE_OUT.write_text(json.dumps({
    "n": N, "aggregate_70b": agg70, "aggregate_rag": agg_rag,
    "by_difficulty": {d: {"70b": aggregate(diff_buckets.get(d,[]),"score_70b"),
                          "rag": aggregate(diff_buckets.get(d,[]),"score_rag")}
                      for d in ["easy","medium","hard"]}
}, indent=2, ensure_ascii=False), encoding="utf-8")
