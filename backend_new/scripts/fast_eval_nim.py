"""
fast_eval_nim.py
----------------
Fast version of the 3-model comparison.
- Reads 0.5B responses directly from `eval_results_quick20.jsonl` (already computed)
- Calls NVIDIA NIM API for 1.5B and 3B
- Applies physics validation to all three
- Takes ~5 seconds total.
"""
import json, sys, time, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import os
if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    del os.environ["SSL_CERT_FILE"]
os.environ["HF_HOME"] = "d:/S6/NLP/Physics_Based_RAG_SLM/hf_cache"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

QUICK20_PATH = PROJECT_ROOT / "data/evaluation/eval_results_quick20.jsonl"
OUT_PATH     = PROJECT_ROOT / "data/evaluation/eval_compare_3models.jsonl"
N = 20

# NVIDIA NIM
NVIDIA_API_KEY  = "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"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_1_5B = "qwen/qwen2.5-1.5b-instruct"
NIM_3B   = "qwen/qwen2.5-3b-instruct"

NIM_SYSTEM = (
    "You are a semiconductor device physics assistant. "
    "The following corpus evidence was retrieved from a physics textbook. "
    "Use it to answer the question. "
    "Write the key equation in plain text Python notation (e.g. Id = 0.5*mu*Cox*(W/L)*(Vgs-Vth)**2). "
    "Then explain each symbol in a plain bulleted list. "
    "NO LaTeX, NO \\[, NO \\(, NO dollar signs, NO markdown math."
)

def _strip_latex(text: str) -> str:
    text = re.sub(r'\\\[.*?\\\]', ' ', text, flags=re.DOTALL)
    text = re.sub(r'\\\(.*?\\\)', ' ', text, flags=re.DOTALL)
    text = text.replace('\\frac', '/').replace('\\sqrt', 'sqrt')
    text = text.replace('\\cdot', '*').replace('\\times', '*')
    text = re.sub(r'\\[a-zA-Z]+', ' ', text)
    return text

def physics_score_response(response: str, validator, dim_checker, num_val) -> dict:
    response = _strip_latex(response)
    lhs, rhs, sym_msg = validator.validate(response)
    dim_msg, num_msg = "[WARN] Dimension check skipped.", "[WARN] Numerical check skipped."
    if lhs is not None:
        dim_msg = dim_checker.check_equation(lhs, rhs)
        num_msg = num_val.evaluate(lhs, rhs)
    sym_ok, dim_ok, num_ok = (lhs is not None and "[OK]" in sym_msg), ("[OK]" in dim_msg), ("[OK]" in num_msg)
    cov_ok = num_ok and "Unresolved" not in num_msg
    return {
        "score": (1 if sym_ok else 0) + (1 if dim_ok else 0) + (1 if num_ok else 0) + (1 if cov_ok else 0),
        "sym_ok": sym_ok, "dim_ok": dim_ok, "num_ok": num_ok, "cov_ok": cov_ok
    }

def call_nim(client, model_id: str, question: str, evidence: list) -> tuple[str, float]:
    ev_block = "\n".join(f"- {e[:300]}" for e in evidence[:3])
    user_msg = f"Evidence:\n{ev_block}\n\nQuestion: {question}"
    t0 = time.time()
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": NIM_SYSTEM}, {"role": "user", "content": user_msg}],
            temperature=0.2, top_p=0.7, max_tokens=512,
        )
        return completion.choices[0].message.content, time.time() - t0
    except Exception as e:
        return f"[NIM ERROR] {e}", time.time() - t0

# Load Data from quick20
dataset = []
with open(QUICK20_PATH, encoding="utf-8") as f:
    for line in f:
        if line.strip(): dataset.append(json.loads(line))
        if len(dataset) >= N: break

# Init only the retriever to get evidence (no TinySLM loading!)
from retrieval.hybrid_retriever import HybridRetriever
retriever = HybridRetriever()
retriever.dense.load_index()
retriever.sparse.build_index_from_docs(retriever.dense.documents)

def retrieve(q, top_k=3):
    return retriever.retrieve(q, top_k=top_k)

from physics.equation_validator import EquationValidator
from physics.dimension_checker import DimensionChecker
from physics.numerical_validator import NumericalValidator
validator, dim_checker, num_val = EquationValidator(), DimensionChecker(), NumericalValidator()

from openai import OpenAI
nim_client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)

# Execute
records = []
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
print("="*60)
with ThreadPoolExecutor(max_workers=5) as pool:
    for i, item in enumerate(dataset):
        q = item["question"]
        evidence = retrieve(q, top_k=3)
        
        # 0.5B is already computed in the file
        resp_05b = item["actual_response"]
        lat_05b = item.get("latency_sec", 0)
        
        future_15b = pool.submit(call_nim, nim_client, NIM_1_5B, q, evidence)
        future_3b  = pool.submit(call_nim, nim_client, NIM_3B,  q, evidence)

        resp_15b, lat_15b = future_15b.result()
        resp_3b,  lat_3b  = future_3b.result()

        sc05 = physics_score_response(resp_05b, validator, dim_checker, num_val)
        sc15 = physics_score_response(resp_15b, validator, dim_checker, num_val)
        sc3  = physics_score_response(resp_3b,  validator, dim_checker, num_val)
        print(f"Q{i+1:<2} | 0.5B: {sc05['score']}/4 | 1.5B: {sc15['score']}/4 | 3B: {sc3['score']}/4")

        records.append({
            "model_05b": {"score": sc05["score"], "sym_ok": sc05["sym_ok"], "dim_ok": sc05["dim_ok"], "num_ok": sc05["num_ok"], "cov_ok": sc05["cov_ok"], "latency": lat_05b},
            "model_15b": {"score": sc15["score"], "sym_ok": sc15["sym_ok"], "dim_ok": sc15["dim_ok"], "num_ok": sc15["num_ok"], "cov_ok": sc15["cov_ok"], "latency": lat_15b},
            "model_3b":  {"score": sc3["score"],  "sym_ok": sc3["sym_ok"],  "dim_ok": sc3["dim_ok"],  "num_ok": sc3["num_ok"],  "cov_ok": sc3["cov_ok"],  "latency": lat_3b},
        })

def agg(k): return sum(r[k]["score"] for r in records)/N
print(f"\nFINAL AVG SCORES => 0.5B: {agg('model_05b'):.2f} | 1.5B: {agg('model_15b'):.2f} | 3B: {agg('model_3b'):.2f}")
