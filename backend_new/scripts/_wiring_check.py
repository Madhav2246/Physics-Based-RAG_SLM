"""
Wiring check — run before evaluation to confirm all components are connected.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def check(label, condition, detail=""):
    status = "[PASS]" if condition else "[FAIL]"
    print(f"  {status}  {label}")
    if detail:
        print(f"         {detail}")
    if not condition:
        sys.exit(1)

print("=" * 60)
print("WIRING CHECK")
print("=" * 60)

# STEP 1 - Physics validators emit [OK]
print("\n--- Step 1: Physics validators emit [OK] tokens ---")
from physics.equation_validator import EquationValidator
from physics.dimension_checker import DimensionChecker
from physics.numerical_validator import NumericalValidator

v = EquationValidator()
d = DimensionChecker()
n = NumericalValidator()

test_eq = "Id = 0.5 * mu * Cox * (W/L) * (Vgs - Vth)**2"
lhs, rhs, sym = v.validate(test_eq)
dim = d.check_equation(lhs, rhs) if lhs else "[WARN] skipped"
num = n.evaluate(lhs, rhs) if lhs else "[WARN] skipped"

print(f"  Symbolic : {sym}")
print(f"  Dimension: {dim}")
print(f"  Numerical: {num}")
check("equation_validator emits [OK]", "[OK]" in sym)
check("dimension_checker emits [OK]",  "[OK]" in dim)
check("numerical_validator emits [OK]", "[OK]" in num)

# STEP 2 - Confidence engine reads [OK] and scores HIGH
print("\n--- Step 2: Confidence engine reads [OK] ---")
from utils.confidence_engine import ConfidenceEngine
e = ConfidenceEngine()
score = e.score(["c1", "c2"], sym, dim, test_eq, num, semantic_similarity=0.82)
label = e.interpret(score)
print(f"  Score: {score} -> {label}")
check("confidence engine scores HIGH for valid equation", score >= 0.80, f"got {score}")

# STEP 3 - Confidence engine LOW for bad answer
bad_sym = "[WARN] No equation detected."
bad_dim = "[WARN] Dimension check skipped."
bad_num = "[WARN] Numerical check skipped."
low_score = e.score(["c1", "c2"], bad_sym, bad_dim, "Some prose answer.", bad_num, semantic_similarity=0.45)
check("confidence engine scores LOW for prose answer", low_score < 0.55, f"got {low_score}")

# STEP 4 - nvidia_golden_qa.jsonl is readable with correct schema
print("\n--- Step 3: Dataset schema ---")
dataset_path = Path("data/evaluation/nvidia_golden_qa.jsonl")
check("nvidia_golden_qa.jsonl exists", dataset_path.exists())
with open(dataset_path, encoding="utf-8") as f:
    first = json.loads(f.readline())
print(f"  Keys: {list(first.keys())}")
print(f"  Sample Q: {first['question'][:70]}...")
check("dataset has 'question' field", "question" in first)
check("dataset has 'answer' field",   "answer" in first)

# STEP 5 - Output paths are named correctly
print("\n--- Step 4: Output path naming ---")
stem = dataset_path.stem
results_path = dataset_path.parent / f"eval_results_{stem}.jsonl"
live_path    = dataset_path.parent / f"live_evaluation_{stem}.json"
print(f"  Results  -> {results_path}")
print(f"  Live JSON-> {live_path}")
check("results path contains 'nvidia'", "nvidia" in str(results_path))
check("live path contains 'nvidia'",    "nvidia" in str(live_path))
check("results file does NOT exist yet (clean slate)", not results_path.exists())

# STEP 6 - evaluate_pipeline.py has flush fix
print("\n--- Step 5: evaluate_pipeline.py flush fix ---")
eval_script = Path("scripts/evaluate_pipeline.py").read_text(encoding="utf-8")
check("evaluate_pipeline.py has out_f.flush()", "out_f.flush()" in eval_script)
check("evaluate_pipeline.py has UTF-8 wrapper",  "TextIOWrapper" in eval_script)

print("\n" + "=" * 60)
print("ALL CHECKS PASSED. Safe to run evaluation.")
print("=" * 60)
