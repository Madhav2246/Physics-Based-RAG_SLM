"""
eval_3model_compare.py
-----------------------
Runs the first 20 questions from nvidia_golden_qa.jsonl through:
  - Qwen 0.5B (Local)
  - Qwen 1.5B (Local)
  - Qwen 3B   (Local)

All models run LOCALLY sequentially.
All three models see IDENTICAL retrieved evidence.
Results saved to: data/evaluation/eval_compare_3models.jsonl
"""
import io, json, sys, time, gc
from pathlib import Path
import torch

# ── UTF-8 safe output on Windows ──────────────────────────────────────────────
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

GOLDEN_PATH  = PROJECT_ROOT / "data/evaluation/nvidia_golden_qa.jsonl"
OUT_PATH     = PROJECT_ROOT / "data/evaluation/eval_compare_3models.jsonl"
N = 20

# Local Models (No adapters since fine-tuning was skipped)
MODELS = [
    {"id": "0.5B", "base": "Qwen/Qwen2.5-0.5B-Instruct", "adapter": "models/finetuned_slm"},
    {"id": "1.5B", "base": "Qwen/Qwen2.5-1.5B-Instruct", "adapter": None},
    {"id": "3B",   "base": "Qwen/Qwen2.5-3B-Instruct",   "adapter": None},
]

def free_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ── Physics scoring ────────────────────────────────────────────────────────────
def physics_score_response(response: str, validator, dim_checker, num_val) -> dict:
    import re
    def _strip_latex(text: str) -> str:
        text = re.sub(r'\\\[.*?\\\]', ' ', text, flags=re.DOTALL)
        text = re.sub(r'\\\(.*?\\\)', ' ', text, flags=re.DOTALL)
        text = text.replace('\\frac', '/').replace('\\sqrt', 'sqrt')
        text = text.replace('\\cdot', '*').replace('\\times', '*')
        text = re.sub(r'\\[a-zA-Z]+', ' ', text)
        return text

    response = _strip_latex(response)
    lhs, rhs, sym_msg = validator.validate(response)
    dim_msg = "[WARN] Dimension check skipped."
    num_msg = "[WARN] Numerical check skipped."
    
    if lhs is not None:
        dim_msg = dim_checker.check_equation(lhs, rhs)
        num_msg = num_val.evaluate(lhs, rhs)
        
    sym_ok  = lhs is not None and "[OK]" in sym_msg
    dim_ok  = "[OK]" in dim_msg
    num_ok  = "[OK]" in num_msg
    cov_ok  = num_ok and "Unresolved" not in num_msg
    total   = (1 if sym_ok else 0) + (1 if dim_ok else 0) + (1 if num_ok else 0) + (1 if cov_ok else 0)
    
    return {
        "symbolic": sym_msg, "dimensional": dim_msg, "numerical": num_msg,
        "sym_ok": sym_ok, "dim_ok": dim_ok, "num_ok": num_ok, "cov_ok": cov_ok,
        "score": total,
    }

# ── Load questions ─────────────────────────────────────────────────────────────
print(f"Loading first {N} questions…")
dataset = []
with open(GOLDEN_PATH, encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= N: break
        if line.strip(): dataset.append(json.loads(line))
print(f"  Loaded {len(dataset)} questions\n")

# ── Init pipeline and pre-retrieve evidence ────────────────────────────────────
print("Initialising Retriever…")
from pipeline.rag_pipeline import RAGPipeline
from reasoning.slm_model import TinySLM
pipeline = RAGPipeline()
pipeline.retriever.dense.load_index()
pipeline.retriever.sparse.build_index_from_docs(pipeline.retriever.dense.documents)

print("Pre-retrieving evidence for all questions…")
for item in dataset:
    # Retrieve top 3
    item["evidence"] = pipeline.retriever.retrieve(item["question"], top_k=3)
    item["results"] = {}

# Unload the RAG pipeline models to free VRAM for the SLMs
del pipeline.retriever
del pipeline.slm
free_vram()

# ── Init validators ───────────────────────────────────────────────────────────
from physics.equation_validator  import EquationValidator
from physics.dimension_checker   import DimensionChecker
from physics.numerical_validator import NumericalValidator
validator   = EquationValidator()
dim_checker = DimensionChecker()
num_val     = NumericalValidator()

from reasoning.prompt_builder import build_prompt

# ── Main loop: Sequential execution per model ─────────────────────────────────
print("=" * 70)
print("Starting Sequential Local Evaluation")
print("=" * 70)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

for m_cfg in MODELS:
    print(f"\n[+] Loading {m_cfg['id']} model locally...")
    slm = TinySLM(model_name=m_cfg['base'], adapter_path=m_cfg['adapter'])
    
    print(f"    Running 20 questions for {m_cfg['id']}...")
    for i, item in enumerate(dataset):
        q = item["question"]
        evidence = item["evidence"]
        
        prompt = build_prompt(q, evidence, corpus_equation=None) 
        
        t0 = time.time()
        resp = slm.generate_multiple(prompt, n_samples=1)[0]
        lat = time.time() - t0
        
        sc = physics_score_response(resp, validator, dim_checker, num_val)
        
        item["results"][m_cfg['id']] = {
            "response": resp,
            "latency": lat,
            **sc
        }
        print(f"    Q{i+1:<2} | {m_cfg['id']}: {sc['score']}/4 | {lat:.1f}s")
        
    print(f"[-] Unloading {m_cfg['id']} model...")
    del slm
    free_vram()

# ── Write and print results ───────────────────────────────────────────────────
records = []
for item in dataset:
    q = item["question"]
    diff = item.get("difficulty", "easy")
    
    record = {
        "question": q,
        "difficulty": diff,
        "model_05b": item["results"]["0.5B"],
        "model_15b": item["results"]["1.5B"],
        "model_3b":  item["results"]["3B"],
    }
    records.append(record)
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("\n" + "=" * 70)
print(f"{'Q':>3}  {'0.5B':>6}  {'1.5B':>6}  {'3B':>6}  Question (truncated)")
print("=" * 70)
for i, item in enumerate(dataset):
    sc05 = item["results"]["0.5B"]["score"]
    sc15 = item["results"]["1.5B"]["score"]
    sc3  = item["results"]["3B"]["score"]
    print(f"{i+1:>3}  {sc05}/4  {sc15:>4}/4  {sc3:>2}/4  {item['question'][:50]}")

# ── Summary table ─────────────────────────────────────────────────────────────
def agg(records, key):
    n = len(records)
    scores = [r[key]["score"] for r in records]
    sym  = sum(1 for r in records if r[key]["sym_ok"])
    dim  = sum(1 for r in records if r[key]["dim_ok"])
    num  = sum(1 for r in records if r[key]["num_ok"])
    cov  = sum(1 for r in records if r[key]["cov_ok"])
    lats = [r[key]["latency"] for r in records if r[key]["latency"]]
    return {
        "n": n, "avg": sum(scores)/n,
        "sym": sym, "dim": dim, "num": num, "cov": cov,
        "avg_lat": sum(lats)/len(lats) if lats else 0,
    }

a05 = agg(records, "model_05b")
a15 = agg(records, "model_15b")
a3  = agg(records, "model_3b")

SEP = "-" * 70
print(f"\n{SEP}")
print("  3-MODEL RAG COMPARISON  (Local Execution)")
print(f"  {'Metric':<28} {'0.5B':>10} {'1.5B':>10} {'3B':>8}")
print(f"  {SEP}")

def pct(n, t): return f"{n}/{t} ({100*n//t}%)" if t else "—"
print(f"  {'Equation parseable':<28} {pct(a05['sym'],N):>10} {pct(a15['sym'],N):>10} {pct(a3['sym'],N):>8}")
print(f"  {'Dimensional pass':<28} {pct(a05['dim'],N):>10} {pct(a15['dim'],N):>10} {pct(a3['dim'],N):>8}")
print(f"  {'Numerical pass':<28} {pct(a05['num'],N):>10} {pct(a15['num'],N):>10} {pct(a3['num'],N):>8}")
print(f"  {'Full coverage':<28} {pct(a05['cov'],N):>10} {pct(a15['cov'],N):>10} {pct(a3['cov'],N):>8}")
print(f"  {'Avg physics score /4':<28} {a05['avg']:>10.2f} {a15['avg']:>10.2f} {a3['avg']:>8.2f}")
print(f"  {'Avg latency (s)':<28} {a05['avg_lat']:>10.1f} {a15['avg_lat']:>10.1f} {a3['avg_lat']:>8.1f}")
print(SEP)

def winner(vals):
    best = max(vals)
    labels = ["0.5B", "1.5B", "3B"]
    return labels[vals.index(best)] + " ✓"

print(f"\n  Winner (physics score): {winner([a05['avg'], a15['avg'], a3['avg']])}")
print(f"\n  Results -> {OUT_PATH}")
print(SEP)
