import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from physics.equation_validator import EquationValidator

text = """SS = T*k*(Cd/Cox + 1)*log(10)/q

Here's the explanation of the key symbols in the equation for the subthreshold swing:

1. **Subthreshold Swing Equation**:
   - The equation is given as `SS = T*k*(Cd/Cox + 1)*log(10)/q`, where:
     - `T` is the temperature in Kelvin (K).
     - `k` is a constant representing the speed of light in a vacuum, approximately equal to 299,792,458 m/s.
     - `Cd` is the depletion width of the oxide layer in meters (m).
     - `Cox` is the capacitance of the oxide layer per unit area in Farads per meter squared (F/m²).
     - `log(10)` is the logarithm base 10 of 10, which equals 1.
     - `q` is the charge of a single electron, approximately 1.602256648 × 10^-19 Coulombs (C)."""

val = EquationValidator()
print("Candidates extracted:")
cands = val._candidate_equations(text)
for c in cands:
    print(f"  Candidate: {c}")
    norm = val.normalize_equation(c)
    print(f"    Normalized: {norm}")
    try:
        lhs_str, rhs_str = norm.split("=", 1)
        lhs_expr, rhs_expr, msg = val.validate(c)
        print(f"    SUCCESS: {lhs_expr} = {rhs_expr} ({msg})")
    except Exception as e:
        print(f"    FAILED: {type(e).__name__}: {e}")
