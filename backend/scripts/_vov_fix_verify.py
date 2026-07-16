"""
Vov fix verification:
  1. W/L = 8.0 hand-check (must match 2*Id/(mu*Cox*Vov^2))
  2. Audit log shows Vov derived into Vgs (displayed provenance == actual computation)
  3. Double-count test: explicit Vgs wins over Vov-derived
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from physics.equation_validator import EquationValidator
from physics.exploration_engine import ExplorationEngine
from physics.value_tracker import ValueTracker

validator = EquationValidator()
engine    = ExplorationEngine(validator)
lhs, rhs, _ = validator.validate("Id = 0.5 * mu * Cox * (W/L) * (Vgs - Vth)**2")

# ── Test A: hand-check W/L = 8.0 ──────────────────────────────────────────
print("=== TEST A: Id=1mA, Vov=0.5V → W/L must be 8.0 ===")
tracker_a = ValueTracker()
tracker_a.add_user("Id",  1e-3, "A", "1mA from query")
tracker_a.add_user("Vov", 0.5,  "V", "0.5V from query")
r = engine.solve_for(lhs, rhs, "WL", tracker_a,
                     "Id = 0.5*mu*Cox*(W/L)*(Vgs-Vth)^2")

print(f"  WL numeric  = {r['numeric']:.4g}  (expected 8.0)")
print(f"  sanity      = {r['sanity_ok']}")
hand = 2 * 1e-3 / (0.05 * 0.02 * 0.5**2)
print(f"  Hand calc   = {hand:.4g}  [2*Id/(mu*Cox*Vov^2)]")
print(f"  Match       = {'PASS' if r['numeric'] is not None and abs(r['numeric'] - hand) < 0.001 else 'FAIL'}")
print()
print("  Audit log:")
for line in tracker_a.audit_lines():
    print(line)
prov = tracker_a.provenance_fraction
print(f"  provenance_fraction = {prov:.2f}  (expected ~0.50)")
vgs_val = tracker_a._values.get("Vgs")
print(f"  Vgs = {vgs_val.value if vgs_val else 'MISSING'} V, prov = {vgs_val.provenance if vgs_val else 'N/A'}")
print(f"  Audit matches computation: {'PASS' if vgs_val and vgs_val.value == 0.9 else 'FAIL'}")

# ── Test B: double-count — explicit Vgs must win ───────────────────────────
print()
print("=== TEST B: explicit Vgs=1.2V must win over Vov=0.5V ===")
tracker_b = ValueTracker()
tracker_b.add_user("Id",  1e-3, "A", "1mA")
tracker_b.add_user("Vov", 0.5,  "V", "0.5V")
tracker_b.add_user("Vgs", 1.2,  "V", "1.2V explicit")   # explicit — must win
r2 = engine.solve_for(lhs, rhs, "WL", tracker_b, "")
vgs_used = tracker_b._values.get("Vgs")
print(f"  Vgs used = {vgs_used.value if vgs_used else 'N/A'} V  (expected 1.2, NOT 0.9)")
print(f"  No double-count = {'PASS' if vgs_used and vgs_used.value == 1.2 else 'FAIL'}")
print(f"  Vgs provenance  = {vgs_used.provenance if vgs_used else 'N/A'}  (expected user)")
