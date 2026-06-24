"""
Standalone tests for ExplorationEngine — run BEFORE wiring into live pipeline.
Tests:
  1. W/L = 8.0 (standard case, verify by hand)
  2. Different units: L = 100nm, W = 500nm
  3. Empty solution: ask for tox from drain current equation (tox not in it)
  4. ValueTracker provenance_fraction arithmetic
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from physics.equation_validator import EquationValidator
from physics.exploration_engine import ExplorationEngine, detect_mode
from physics.value_tracker import ValueTracker

validator = EquationValidator()
engine    = ExplorationEngine(validator)

# ── Parse the MOSFET drain current corpus equation ──────────────────────────
CORPUS_EQ = "Id = 0.5 * mu * Cox * (W/L) * (Vgs - Vth)**2"
lhs, rhs, msg = validator.validate(CORPUS_EQ)
assert lhs is not None, f"Could not parse corpus equation: {msg}"
print(f"Corpus equation parsed OK: {lhs} = {rhs}\n")

# ── TEST 1: W/L = 8.0 (hand-verified) ──────────────────────────────────────
print("=" * 60)
print("TEST 1 — W/L from Id=1mA, Vov=0.5V (standard 100nm defaults for mu, Cox)")
print("=" * 60)
tracker1 = ValueTracker()
tracker1.add_user("Id",  1e-3, "A",  "1 mA user-supplied")
tracker1.add_user("Vov", 0.5,  "V",  "0.5 V user-supplied")

# Map Vov into (Vgs - Vth) for substitution: need to sub Vov = Vgs - Vth
# The corpus eq uses (Vgs - Vth), not Vov directly. 
# We resolve by adding both Vgs and Vth with Vov constraint.
# Simplest: let the tracker provide Vgs=1.0 and Vth=0.4 (Vov=0.6 default)
# but user wants Vov=0.5, so set Vgs=0.9, Vth=0.4
tracker1.add_user("Vgs", 0.9, "V", "set for Vov=0.5V")
tracker1.add_user("Vth", 0.4, "V", "default threshold")

result1 = engine.solve_for(lhs, rhs, "WL", tracker1, CORPUS_EQ)
print(f"  success  : {result1['success']}")
print(f"  symbolic : {result1['symbolic']}")
print(f"  numeric  : {result1['numeric']:.4g}")
print(f"  sanity   : {result1['sanity_ok']}")
print(f"  error    : '{result1['error']}'")
print()
# Hand-check: WL = 2*Id / (mu*Cox*(Vgs-Vth)^2)
# = 2*1e-3 / (0.05 * 0.02 * (0.9-0.4)^2)
# = 2e-3 / (0.05 * 0.02 * 0.25)
# = 2e-3 / 2.5e-4 = 8.0
expected = 2*1e-3 / (0.05 * 0.02 * (0.9-0.4)**2)
print(f"  Hand-check: WL = 2*Id/(mu*Cox*Vov^2) = {expected:.4g}")
match = abs((result1['numeric'] or 0) - expected) < 0.001
print(f"  Match: {'PASS' if match else 'FAIL'}")

# ── TEST 2: Different unit prefix (nm) ─────────────────────────────────────
print()
print("=" * 60)
print("TEST 2 — W/L with W=500nm, L=100nm (unit parsing check)")
print("=" * 60)
query2 = "How do I choose W/L for Id = 1mA, Vov = 0.5V, L = 100nm?"
parsed2 = engine.extract_user_values(query2)
target2 = engine.detect_target(query2)
print(f"  Detected target: {target2}")
print(f"  Parsed values:")
for line in parsed2.audit_lines():
    print(f"  {line}")
print(f"  L value: {parsed2._values.get('L').value if 'L' in parsed2 else 'NOT FOUND'} m")
expected_L = 100e-9
actual_L = parsed2._values.get('L').value if 'L' in parsed2 else None
match2 = actual_L is not None and abs(actual_L - expected_L) < 1e-12
print(f"  L = 100nm = 1e-7m check: {'PASS' if match2 else 'FAIL'}")

# ── TEST 3: Empty solution (tox not in drain current equation) ──────────────
print()
print("=" * 60)
print("TEST 3 — Empty solution: ask for tox from drain current eq (not in it)")
print("=" * 60)
tracker3 = ValueTracker()
result3 = engine.solve_for(lhs, rhs, "tox", tracker3, CORPUS_EQ)
print(f"  success  : {result3['success']}")
print(f"  symbolic : {result3['symbolic']}")
print(f"  numeric  : {result3['numeric']}")
print(f"  error    : '{result3['error']}'")
# Must NOT raise IndexError, must return success=False with descriptive error
assert not result3['success'], "Expected failure for tox not in equation"
assert "does not appear" in result3['error'], f"Expected 'does not appear' in error, got: {result3['error']}"
print(f"  PASS — clean failure, no IndexError")

# ── TEST 4: ValueTracker provenance_fraction ────────────────────────────────
print()
print("=" * 60)
print("TEST 4 — provenance_fraction arithmetic")
print("=" * 60)
t4 = ValueTracker()
t4.add_user("Id", 1e-3, "A")
t4.add_user("Vov", 0.5, "V")
t4.resolve({"mu", "Cox"})   # fills 2 defaults
frac = t4.provenance_fraction
print(f"  2 user / 4 total = {frac:.2f}  (expected 0.50)")
assert abs(frac - 0.50) < 0.01, f"Expected 0.50, got {frac}"
print(f"  PASS")

# ── TEST 5: detect_mode ─────────────────────────────────────────────────────
print()
print("=" * 60)
print("TEST 5 — detect_mode")
print("=" * 60)
cases = [
    ("How do I choose W/L for Id = 1mA?",           "EXPLORE"),
    ("What is the MOSFET drain current equation?",   "LOOKUP"),
    ("Design the W/L ratio for a target Vov",        "EXPLORE"),
    ("What is the body effect threshold voltage?",   "LOOKUP"),
]
all_ok = True
for q, expected_mode in cases:
    got = detect_mode(q)
    ok = got == expected_mode
    print(f"  [{('PASS' if ok else 'FAIL')}] '{q[:50]}...' -> {got}")
    all_ok = all_ok and ok

print()
print("=" * 60)
if all_ok and not result3['success'] and match and match2:
    print("ALL TESTS PASSED — safe to wire into pipeline")
else:
    print("SOME TESTS FAILED — fix before connecting pipeline")
print("=" * 60)
