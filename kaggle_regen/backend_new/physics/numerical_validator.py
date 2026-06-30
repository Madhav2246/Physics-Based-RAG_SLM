import sympy as sp
import math


class NumericalValidator:
    """
    Substitutes realistic MOSFET parameter values into a SymPy RHS expression
    and checks whether the result is physically plausible.

    Fixes applied:
    - Added all new symbols (Vth0, gamma, Phi_f, Vsb, k, T, q, Cd, Vds, n, Vt).
    - Result interpretation is now general — no longer hardcoded as "drain current".
    - Range check uses relative magnitude bands instead of drain-current-specific bounds.
    """

    def __init__(self):
        # Realistic SI-unit values for a 100-nm MOSFET
        self.test_values = {
            # Core drain current
            "mu":    0.05,      # carrier mobility [m²/Vs]
            "Cox":   0.02,      # gate oxide capacitance [F/m²]
            "W":     1e-6,      # gate width [m]
            "L":     1e-7,      # gate length [m]
            "Vgs":   1.0,       # gate-source voltage [V]
            "Vth":   0.4,       # threshold voltage [V]
            "Vds":   1.0,       # drain-source voltage [V]
            # Body effect
            "Vth0":  0.5,       # zero-bias threshold voltage [V]
            "Vsb":   0.0,       # source-body voltage [V]
            "Phi_f": 0.35,      # Fermi potential [V]
            "gamma": 0.4,       # body-effect coefficient [V^0.5]
            # Subthreshold swing
            "k":     1.38e-23,  # Boltzmann constant [J/K]
            "T":     300,       # temperature [K]
            "q":     1.6e-19,   # electron charge [C]
            "Cd":    5e-3,      # depletion capacitance [F/m²]
            # Small-signal / general
            "gm":    1e-3,      # transconductance [A/V]
            "ro":    1e4,       # output resistance [Ω]
            "Vt":    0.02585,   # thermal voltage at 300 K [V]
            "n":     1.0,       # ideality factor (dimensionless)
            # Channel length modulation / saturation
            "lam":   0.1,       # CLM parameter [1/V] — typical 100nm node
            "Vov":   0.6,       # overdrive voltage (Vgs-Vth) [V]
            "Vdsat": 0.6,       # saturation drain voltage [V]
            "Idsat": 1.8e-3,    # saturation drain current [A]
            "Vbs":   0.0,       # bulk-source voltage [V]
            # Second-order
            "tox":   2e-9,      # oxide thickness [m] — 2nm
            "Iref":  1e-4,      # reference current [A]
            "Av":    10.0,      # voltage gain (dimensionless)
            "DIBL":  0.05,      # DIBL coefficient [dimensionless]
            # -- SymPy name overrides ---------------------------------------
            "S":     0.07,      # subthreshold swing [V/dec]
            "E":     1e6,       # electric field [V/m]
            "I":     1e-3,      # current [A]
            "C":     1e-12,     # capacitance [F]
            "N":     1e22,      # concentration [1/m3]
            "O":     1.0,       # generic constant / scale factor
            "Q":     1e-15,     # charge [C]
            "S0":    0.06,      # subthreshold swing baseline [V/dec]
            "E_max": 1e6,       # max electric field [V/m]
        }

    def evaluate(self, lhs_expr, rhs_expr) -> str:
        try:
            numeric_expr = rhs_expr.subs(self.test_values)

            # Check for remaining unresolved symbols
            remaining = numeric_expr.free_symbols
            if remaining:
                return f"[WARN] Unresolved symbols remain: {remaining}"

            numeric_value = float(numeric_expr.evalf())

            # -- Sanity checks ---------------------------------------------
            if math.isnan(numeric_value):
                return "[FAIL] Numerical result is NaN"
            if math.isinf(numeric_value):
                return "[FAIL] Numerical result is infinite"

            abs_val = abs(numeric_value)

            # Equation-specific bounds
            if lhs_expr is not None and isinstance(lhs_expr, sp.Symbol):
                name = lhs_expr.name
                if name == "SS":
                    if not (0.059 <= abs_val <= 0.150):
                        return f"[WARN] Unrealistic Subthreshold Swing: {numeric_value*1000:.1f} mV/dec (expected 60-100)"
                elif name in ["Vth", "Vth0"]:
                    if not (0.1 <= abs_val <= 1.5):
                        return f"[WARN] Unrealistic Threshold Voltage: {numeric_value:.2f} V"
                elif name in ["Id", "Ids"]:
                    if not (1e-12 <= abs_val <= 1.0):
                        return f"[WARN] Unrealistic Drain Current: {numeric_value:.2e} A"

            # Very small (below floating-point noise floor)
            if abs_val < 1e-30 and abs_val != 0.0:
                return f"[WARN] Extremely small result: {numeric_value:.4e} — possible unit error"

            # Extremely large (likely a unit inconsistency)
            if abs_val > 1e10:
                return f"[WARN] Very large result: {numeric_value:.4e} — check units"

            return f"[OK] Numerically plausible: {numeric_value:.6e}"

        except Exception as e:
            return f"[WARN] Numerical evaluation failed: {str(e)}"