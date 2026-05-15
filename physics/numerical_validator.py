import sympy as sp
import math


class NumericalValidator:

    def __init__(self):

        # Realistic nanoscale MOSFET parameters
        self.test_values = {
            "mu": 0.05,
            "Cox": 0.02,
            "W": 1e-6,
            "L": 1e-7,
            "Vgs": 1.0,
            "Vth": 0.4
        }

    def evaluate(self, rhs_expr):

        try:
            # Substitute numeric values
            numeric_expr = rhs_expr.subs(self.test_values)

            # 🔥 NEW: check for remaining symbols
            if len(numeric_expr.free_symbols) > 0:
                return f"⚠ Unresolved symbols remain: {numeric_expr.free_symbols}"

            # Force numeric evaluation
            numeric_value = float(numeric_expr.evalf())

            # Sanity checks
            if math.isnan(numeric_value):
                return "❌ Numerical result is NaN"

            if math.isinf(numeric_value):
                return "❌ Numerical result is infinite"

            if numeric_value < 0:
                return f"❌ Negative drain current: {numeric_value} A"

            if numeric_value > 1:
                return f"❌ Unrealistically high current: {numeric_value} A"

            if numeric_value < 1e-12:
                return f"⚠ Extremely low current: {numeric_value} A"

            return f"✔ Numerically realistic drain current: {numeric_value:.6e} A"

        except Exception as e:
            return f"⚠ Numerical evaluation failed: {str(e)}"