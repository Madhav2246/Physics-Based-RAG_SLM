"""
Diagnostic: print exactly what values are extracted for S1 and S6 regression failures.
Run from backend/:
    python -X utf8 scripts/_regression_diag.py
"""
import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Enable debug logging so we see SLM stage output
logging.basicConfig(level=logging.DEBUG,
                    format="%(name)s %(levelname)s: %(message)s")

from physics.equation_validator import EquationValidator
from physics.exploration_engine import ExplorationEngine

validator = EquationValidator()
engine    = ExplorationEngine(validator.symbols)

# Try to wire SLM (same as eval script would do)
try:
    from reasoning.slm_model import TinySLM
    slm = TinySLM()
    engine.set_slm_model(slm)
    print("[DIAG] SLM wired into engine")
except Exception as e:
    print(f"[DIAG] SLM not loaded ({e}) — regex only")

QUERIES = [
    # (label, query)
    ("S1", "How do I choose W/L for Id = 1mA, Vov = 0.5V?"),
    ("S6", "What Vth do I get with Vsb = 1.0V, gamma = 0.4?"),
    ("S2", "How do I choose W/L for Id = 200uA, Vov = 0.3V?"),  # known PASS — control
]

print("\n" + "=" * 80)
print("EXTRACTION DIAGNOSTIC")
print("=" * 80)

for label, query in QUERIES:
    print(f"\n[{label}] {query}")
    tracker = engine.extract_user_values(query)
    print("  Extracted values:")
    for sym, tv in sorted(tracker._values.items()):
        print(f"    {tv.provenance:7s}  {sym:10s} = {tv.value:.6g}  ({tv.description})")
    user_vals = {k: v for k, v in tracker._values.items() if v.provenance == "user"}
    total = len(tracker._values)
    user_count = len(user_vals)
    prov = user_count / total if total else 0
    print(f"  prov={prov:.0%}  ({user_count} user / {total} total registered)")
