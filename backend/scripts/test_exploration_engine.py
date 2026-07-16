"""
Standalone test for ExplorationEngine.
Run from backend/ directory:
    python -X utf8 scripts/test_exploration_engine.py

Expected outputs are printed so you can verify before integrating.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from physics.equation_validator import EquationValidator
from physics.exploration_engine import ExplorationEngine, detect_mode, extract_numeric_values

validator = EquationValidator()
engine    = ExplorationEngine(validator.symbols)

SEP = "-" * 60

def run_test(title: str, corpus_eq_str: str, query: str, expect_approx: float = None):
    print(f"\n{SEP}")
    print(f"TEST: {title}")
    print(f"  Corpus eq : {corpus_eq_str}")
    print(f"  Query     : {query}")

    # Parse corpus equation
    lhs, rhs, msg = validator.validate(corpus_eq_str)
    if lhs is None:
        print(f"  FAIL — could not parse corpus eq: {msg}")
        return

    # Mode detection
    mode = detect_mode(query)
    print(f"  Mode      : {mode}")

    if mode != "EXPLORE":
        print("  [SKIPPED — mode is LOOKUP, not EXPLORE]")
        return

    result = engine.solve(lhs, rhs, query)

    if result["error"]:
        print(f"  ERROR     : {result['error']}")
        return

    print(f"  Target    : {result['target']} (strategy: {result['strategy']})")
    print(f"  Derived   : {result['derived_sympy']}")
    print(f"  LaTeX     : {result['derived_eq']}")

    if result["numeric_inputs"]:
        print(f"  Num inputs: {result['numeric_inputs']}")

    if result["numeric_result"] is not None:
        val = result["numeric_result"]
        print(f"  Answer    : {result['numeric_latex']}")
        if result["bounds_warning"]:
            print(f"  [WARNING] : {result['bounds_warning']}")
        if expect_approx is not None:
            tol = 0.05  # 5% tolerance
            ok = abs(val - expect_approx) / max(abs(expect_approx), 1e-30) < tol
            status = "PASS" if ok else f"FAIL (expected ~{expect_approx}, got {val:.4g})"
            print(f"  Numeric   : {status}")
    else:
        print(f"  Numeric   : not computed (missing inputs or error)")
        if result["error"]:
            print(f"  Note      : {result['error']}")

    print(f"  Confidence: {result['confidence_flags']}")


# ── Test 1: Classic W/L design problem ────────────────────────────────────────
run_test(
    title="W/L for target Id",
    corpus_eq_str="Id = 0.5 * mu * Cox * (W/L) * (Vgs - Vth)^2",
    query="What W/L do I need for Id = 1mA with Vov = 0.5V?",
        expect_approx=8.0,    # 2*1e-3 / (0.05 * 0.02 * 0.5^2) = 8 using standard test_values
)

# ── Test 2: Solve for Vth from same equation ──────────────────────────────────
run_test(
    title="Vth from drain current equation",
    corpus_eq_str="Id = 0.5 * mu * Cox * (W/L) * (Vgs - Vth)^2",
    query="What Vth is required to achieve Id = 0.5mA with Vgs = 1.2V?",
        expect_approx=None,   # symbolic — skip numeric check (WL not provided)
)

# ── Test 3: Transconductance from drain current ───────────────────────────────
run_test(
    title="gm from small-signal expression",
    corpus_eq_str="gm = mu * Cox * (W/L) * (Vgs - Vth)",
    query="Calculate the transconductance gm for Vov = 0.3V",
        expect_approx=None,
)

# ── Test 4: Mode detection on LOOKUP queries ──────────────────────────────────
print(f"\n{SEP}")
print("TEST: Mode detection")
lookup_queries = [
    "What is the drain current equation for a MOSFET in saturation?",
    "Define threshold voltage.",
    "Give me the body effect equation.",
]
explore_queries = [
    "What W/L do I need for Id = 1mA?",
    "How do I choose Cox to achieve a target gm?",
    "Calculate the required tox for Cox = 5 mF/m2.",
]
for q in lookup_queries:
    print(f"  LOOKUP  [{'OK' if detect_mode(q)=='LOOKUP' else 'FAIL'}] : {q[:60]}")
for q in explore_queries:
    print(f"  EXPLORE [{'OK' if detect_mode(q)=='EXPLORE' else 'FAIL'}] : {q[:60]}")

# ── Test 5: Numeric extraction ────────────────────────────────────────────────
print(f"\n{SEP}")
print("TEST: Numeric extraction")
test_queries = [
    ("Id = 1mA, Vov = 0.5V",            {"Id": 1e-3, "Vov": 0.5}),
    ("Id = 500uA with Vgs = 1.2 V",     {"Id": 5e-4, "Vgs": 1.2}),
    ("Id = 2e-3 A and Vth = 0.4",       {"Id": 2e-3, "Vth": 0.4}),
    ("tox = 2nm, Cox = 0.02 F",         {"tox": 2e-9, "Cox": 0.02}),
]
for q, expected in test_queries:
    got = extract_numeric_values(q)
    ok  = all(abs(got.get(k, 0) - v) / max(abs(v), 1e-30) < 1e-6 for k, v in expected.items())
    print(f"  [{'PASS' if ok else 'FAIL'}] '{q[:50]}' → {got}")

print(f"\n{SEP}")
print("All tests complete.")
