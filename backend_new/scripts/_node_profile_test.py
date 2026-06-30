"""
Node profile smoke test — no model load, pure engine + SymPy.

Runs the same W/L design query against all four node profiles and prints a
table. W/L must differ across rows (different mu, Cox per node). If any two
rows are identical, profile loading is broken.

Run from backend/:
    python -X utf8 scripts/_node_profile_test.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from physics.equation_validator import EquationValidator
from physics.exploration_engine import ExplorationEngine
from physics.node_profile_manager import NodeProfileManager

validator = EquationValidator()
engine    = ExplorationEngine(validator.symbols)
manager   = NodeProfileManager()

# Corpus drain-current equation (as retrieved from corpus in the real pipeline)
CORPUS_EQ = "Id = 0.5 * mu * Cox * (W/L) * (Vgs - Vth)^2"
lhs, rhs, msg = validator.validate(CORPUS_EQ)
assert lhs is not None, f"corpus eq failed to parse: {msg}"

QUERY = "What W/L for Id=1mA, Vov=0.3V?"

print("=" * 78)
print(f"NODE PROFILE TEST")
print(f"  Query : {QUERY}")
print(f"  Eq    : {CORPUS_EQ}")
print("=" * 78)
print(f"{'Node':<16} {'W/L':<12} {'mu used':<12} {'Cox used':<12} {'Vth used':<10}")
print("-" * 78)

results = []
for node_name in ["100nm_CMOS", "28nm_FDSOI", "16nm_FinFET", "5nm_GAA"]:
    node_defaults = manager.as_tracker_defaults(node_name)
    result = engine.solve(lhs, rhs, QUERY, node_defaults=node_defaults)

    wl  = result.get("numeric")
    tr  = result.get("tracker")
    mu  = tr._values.get("mu")
    cox = tr._values.get("Cox")
    vth = tr._values.get("Vth")

    mu_v  = mu.value  if mu  else None
    cox_v = cox.value if cox else None
    vth_v = vth.value if vth else None

    wl_str = f"{wl:.4g}" if wl is not None else "FAIL"
    print(f"{node_name:<16} {wl_str:<12} "
          f"{mu_v:<12.4g} {cox_v:<12.4g} {vth_v if vth_v is not None else 'n/a':<10}")
    results.append(wl)

print("-" * 78)

# Verify all four W/L values are distinct
clean = [r for r in results if r is not None]
distinct = len(set(round(r, 6) for r in clean))
if distinct == len(results) and len(clean) == 4:
    print(f"PASS: all 4 nodes produced distinct W/L values.")
else:
    print(f"FAIL: expected 4 distinct values, got {distinct} distinct "
          f"({len(clean)}/4 solved). Profile loading may be broken.")

# Detection sanity
print()
print("Detection check:")
for q, expected in [
    ("Using the 5nm GAA node, what W/L?", "5nm_GAA"),
    ("16nm FinFET design", "16nm_FinFET"),
    ("28nm FDSOI process", "28nm_FDSOI"),
    ("standard 100nm question", "100nm_CMOS"),
    ("no node mentioned here", None),
]:
    got = manager.detect_from_query(q)
    ok = "OK" if got == expected else "FAIL"
    print(f"  [{ok}] detect('{q[:40]}') -> {got}")
