import sympy as sp


class DimensionChecker:
    """
    Recursive SymPy AST dimensional analysis engine.

    Fixes applied:
    - int(exponent) → float(exponent): now handles sqrt() = Pow(x, 1/2) correctly.
    - Expanded dim_map from 7 → 19 symbols (body effect, subthreshold swing, small-signal).
    - sp.Number/Integer/Float/Rational all return {} (dimensionless) — prevents false errors.
    - sp.Function nodes (log, ln) return {} — dimensionless by convention.
    - Renamed internal method _add → _mul_dims for clarity (multiply = add exponents).
    """

    def __init__(self):
        # Base dimension keys: I (current), V (voltage), L (length),
        #                      T (time), K (temperature), C (charge), J (energy)
        self.dim_map = {
            # -- Core MOSFET drain current ---------------------------------
            "Id":    {"I": 1},
            "Ids":   {"I": 1},
            "mu":    {"L": 2, "V": -1, "T": -1},         # mobility [m²/Vs]
            "Cox":   {"I": 1, "T": 1, "V": -1, "L": -2}, # gate capacitance [F/m²]
            "W":     {"L": 1},
            "L":     {"L": 1},
            "Vgs":   {"V": 1},
            "Vth":   {"V": 1},
            "Vds":   {"V": 1},
            # -- Body effect -----------------------------------------------
            "Vth0":  {"V": 1},
            "gamma": {"V": 0.5},                          # body-effect coeff [V^½]
            "Phi_f": {"V": 1},                            # Fermi potential [V]
            "Vsb":   {"V": 1},                            # source-body voltage [V]
            # -- Subthreshold swing ----------------------------------------
            "k":     {"V": 1, "C": 1, "K": -1},                  # Boltzmann [J/K] -> [V*C/K]
            "T":     {"K": 1},                            # Temperature [K]
            "q":     {"C": 1},                            # Electron charge [C]
            "SS":    {"V": 1},                            # subthreshold swing [V/dec]
            "Cd":    {"I": 1, "T": 1, "V": -1, "L": -2}, # depletion cap [F/m²]
            # -- Small-signal ----------------------------------------------
            "gm":    {"I": 1, "V": -1},                  # transconductance [A/V]
            "ro":    {"V": 1, "I": -1},                  # output resistance [Ω]
            "Vt":    {"V": 1},                            # thermal voltage [V]
            "n":     {},                                  # ideality factor (dimensionless)
            # -- Channel length modulation / saturation --------------------
            "lam":   {"V": -1},                           # CLM parameter [1/V]
            "Vov":   {"V": 1},                            # overdrive voltage [V]
            "Vdsat": {"V": 1},                            # saturation drain voltage [V]
            "Idsat": {"I": 1},                            # saturation drain current [A]
            "Vbs":   {"V": 1},                            # bulk-source voltage [V]
            # -- Second-order effects --------------------------------------
            "DIBL":  {},                                  # dimensionless ratio [V/V]
            "Av":    {},                                  # voltage gain (dimensionless)
            "tox":   {"L": 1},                            # oxide thickness [m]
            "Iref":  {"I": 1},                            # reference current [A]
            # -- pn junction / diode ---------------------------------------
            "J":     {"I": 1, "L": -2},                  # current density [A/m2]
            "Jo":    {"I": 1, "L": -2},                  # sat. current density [A/m2]
            "J0":    {"I": 1, "L": -2},
            "V":     {"V": 1},                            # generic voltage [V]
            "phi":   {"V": 1},                            # surface potential [V]
            "Vbi":   {"V": 1},                            # built-in voltage [V]
            "NA":    {"L": -3},                           # acceptor conc. [1/m3]
            "ND":    {"L": -3},                           # donor conc. [1/m3]
            "eps":   {"I": 1, "T": 1, "V": -1, "L": -1}, # permittivity [F/m]
            "xd":    {"L": 1},                            # depletion width [m]
            # -- SymPy name overrides ---------------------------------------
            "S":     {"V": 1},                            # subthreshold swing
            "E":     {"V": 1, "L": -1},                  # electric field
            "I":     {"I": 1},                            # current
            "C":     {"I": 1, "T": 1, "V": -1},          # capacitance
            "N":     {"L": -3},                           # concentration
            "O":     {},                                  # oxide/other
            "Q":     {"C": 1},                            # charge
            "S0":    {"V": 1},                            # subthreshold swing baseline
            "E_max": {"V": 1, "L": -1},                  # max electric field
        }

    # -- Internal helpers ------------------------------------------------------

    def _mul_dims(self, d1: dict, d2: dict) -> dict:
        """Combine two dimension dicts via multiplication (add exponents)."""
        result = d1.copy()
        for k, v in d2.items():
            result[k] = result.get(k, 0) + v
        return result

    def _scale(self, dims: dict, factor: float) -> dict:
        """Scale all exponents by factor — used for Pow nodes."""
        return {k: v * factor for k, v in dims.items()}

    def _simplify(self, dims: dict) -> dict:
        """Drop zero-exponent entries."""
        return {k: v for k, v in dims.items() if v != 0}

    # -- Recursive evaluator ---------------------------------------------------

    def evaluate(self, expr) -> dict:
        # -- Named symbol --------------------------------------------------
        if isinstance(expr, sp.Symbol):
            return self.dim_map.get(expr.name, {})   # unknown → dimensionless

        # -- Numeric constant ----------------------------------------------
        if isinstance(expr, (sp.Number, sp.Integer, sp.Float,
                             sp.Rational, sp.core.numbers.Half)):
            return {}

        # -- Multiplication ------------------------------------------------
        if isinstance(expr, sp.Mul):
            dims = {}
            for arg in expr.args:
                dims = self._mul_dims(dims, self.evaluate(arg))
            return dims

        # -- Power (incl. sqrt = Pow(x, 1/2)) -----------------------------
        if isinstance(expr, sp.Pow):
            base_dims = self.evaluate(expr.args[0])
            try:
                # float() handles Integer, Rational (1/2), Float correctly
                exp_val = float(expr.args[1])
                return self._scale(base_dims, exp_val)
            except (TypeError, ValueError):
                return {}

        # -- Addition ------------------------------------------------------
        if isinstance(expr, sp.Add):
            dims_list = [self._simplify(self.evaluate(arg)) for arg in expr.args]
            first = dims_list[0]
            for d in dims_list[1:]:
                if d != first:
                    return {"DIMENSION_MISMATCH": 1}
            return first

        # -- Function calls (log, ln, sqrt-as-function) --------------------
        if isinstance(expr, sp.Function):
            return {}

        return {}

    # -- Public API ------------------------------------------------------------

    def check_equation(self, lhs_expr, rhs_expr) -> str:
        lhs_dims = self._simplify(self.evaluate(lhs_expr))
        rhs_dims = self._simplify(self.evaluate(rhs_expr))

        if lhs_dims == rhs_dims:
            return f"[OK] Dimensionally consistent: {lhs_dims}"
        return (
            "[FAIL] Dimension mismatch:\n"
            f"  LHS {lhs_dims}\n"
            f"  RHS {rhs_dims}"
        )