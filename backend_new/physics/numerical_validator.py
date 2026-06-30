"""
physics/numerical_validator.py
-------------------------------
Multi-point numerical validation across realistic technology nodes.

Problems with the old single-point approach
  - One hardcoded test value per symbol → a garbage equation that evaluates to
    something in [1e-30, 1e10] always passes, even if it is physically absurd.
  - abs_val < 1e10 is the only gate → almost never fails.
  - No knowledge of what range makes sense for the *left-hand-side* quantity.

New approach
  (1) Three technology nodes (100 nm, 22 nm, 5 nm) with self-consistent
      parameter sets.  Each node is a full dict of SI-unit values.
  (2) evaluate() tests at ALL three nodes and requires ≥ 2/3 to give a
      physically plausible result (majority vote).
  (3) A per-symbol RESULT_RANGES dict specifies [lo, hi] for every recognisable
      LHS quantity.  If the LHS symbol is known, the result at each node is
      checked against that range.  If the LHS is unknown, fall back to the
      generic order-of-magnitude sanity check (not NaN, not Inf, not < 1e-30,
      not > 1e15).
  (4) self.test_values is kept for backward compatibility with new_checker's
      _coverage() — set to the 22 nm node (middle of the range).
"""
from __future__ import annotations
import math
import re as _re
import sympy as sp

# Conductance-family base names: any indexed variant (go1, gm5, GL) falls
# back to the canonical value in the tech node.
_CONDUCTANCE_BASES = {"go", "gout", "GL", "gL"}
_GM_BASES = {"gm"}  # gm5 → gm


# ── Technology-node parameter sets (all SI units) ────────────────────────────
#
# Node naming: realistic values for bulk CMOS / FinFET / GAA at each node.
# Symbols match the keys used in EquationValidator and DimensionChecker.
#
_NODE_100NM = {
    # Geometry
    "W": 1e-7,      "L": 1e-7,      "tox": 4e-9,
    # Gate stack
    "Cox": 8.6e-3,  "EOT": 4e-9,
    # Voltages
    "Vgs": 1.0,     "Vth": 0.45,    "Vds": 1.0,     "Vov": 0.55,
    "Vth0": 0.45,   "Vsb": 0.0,     "Vbs": 0.0,     "Vdsat": 0.5,
    "Vbi": 0.8,     "V": 0.6,       "phi": 0.37,    "Phi_f": 0.37,
    "Vt": 0.02585,  "Vb": 0.0,
    # Mobility
    "mu": 0.04,     "mu_eff": 0.04, "mu_e": 0.135,  "mu_h": 0.048,
    # Currents
    "Id": 5e-4,     "Ids": 5e-4,    "Idsat": 5e-4,  "Iref": 1e-4,
    "gm": 2e-3,     "ro": 5e4,
    # pn junction
    "J": 1e4,       "Jo": 1e-2,     "J0": 1e-2,
    # Concentrations / quantum
    "NA": 1e23,     "ND": 1e23,     "ni": 1.5e16,
    "Nc": 2.8e25,   "Nv": 1.04e25,  "N": 1e23,
    # Physical constants
    "k": 1.38e-23,  "q": 1.6e-19,   "T": 300,       "n": 1.0,
    # Permittivities
    "eps": 1.04e-10,"eps_s": 1.04e-10, "eps_ox": 3.45e-11,
    # Lengths
    "xd": 1e-7,     "Wdep": 1e-7,   "Ln": 1e-5,     "Lp": 1e-5,
    # Diffusivities / velocity
    "Dn": 3.5e-3,   "Dp": 1.2e-3,   "vsat": 1e5,
    # Charges / capacitances
    "Qdep": 1e-3,   "Qinv": 1e-2,
    "Cdep": 1e-3,   "Cd": 5e-3,
    # Second-order
    "gamma": 0.4,   "lam": 0.1,     "DIBL": 0.05,
    "Av": 20.0,     "SS": 0.085,    "S": 0.085,
    # Misc
    "E": 1e6,       "E_max": 5e6,   "Idsat": 5e-4,
    "I": 5e-4,      "C": 1e-14,     "Q": 8e-17,
    "O": 1.0,       "S0": 0.062,
}

_NODE_22NM = {
    "W": 2.2e-8,    "L": 2.2e-8,    "tox": 1.5e-9,
    "Cox": 2.3e-2,  "EOT": 1.5e-9,
    "Vgs": 0.85,    "Vth": 0.35,    "Vds": 0.85,    "Vov": 0.50,
    "Vth0": 0.35,   "Vsb": 0.0,     "Vbs": 0.0,     "Vdsat": 0.45,
    "Vbi": 0.75,    "V": 0.5,       "phi": 0.32,    "Phi_f": 0.32,
    "Vt": 0.02585,  "Vb": 0.0,
    "mu": 0.025,    "mu_eff": 0.025,"mu_e": 0.09,   "mu_h": 0.03,
    "Id": 8e-5,     "Ids": 8e-5,    "Idsat": 8e-5,  "Iref": 2e-5,
    "gm": 1.5e-3,   "ro": 1e5,
    "J": 5e4,       "Jo": 1e-1,     "J0": 1e-1,
    "NA": 5e23,     "ND": 5e23,     "ni": 1.5e16,
    "Nc": 2.8e25,   "Nv": 1.04e25,  "N": 5e23,
    "k": 1.38e-23,  "q": 1.6e-19,   "T": 300,       "n": 1.0,
    "eps": 1.04e-10,"eps_s": 1.04e-10,"eps_ox": 3.45e-11,
    "xd": 3e-8,     "Wdep": 3e-8,   "Ln": 5e-6,     "Lp": 5e-6,
    "Dn": 2.3e-3,   "Dp": 8e-4,     "vsat": 8e4,
    "Qdep": 2e-3,   "Qinv": 2e-2,
    "Cdep": 2.5e-3, "Cd": 1e-2,
    "gamma": 0.3,   "lam": 0.15,    "DIBL": 0.08,
    "Av": 15.0,     "SS": 0.075,    "S": 0.075,
    "E": 3e6,       "E_max": 1.5e7,
    "I": 8e-5,      "C": 2e-15,     "Q": 1.2e-17,
    "O": 1.0,       "S0": 0.063,
}

_NODE_5NM = {
    "W": 5e-9,      "L": 5e-9,      "tox": 6e-10,
    "Cox": 5.75e-2, "EOT": 6e-10,
    "Vgs": 0.65,    "Vth": 0.25,    "Vds": 0.65,    "Vov": 0.40,
    "Vth0": 0.25,   "Vsb": 0.0,     "Vbs": 0.0,     "Vdsat": 0.35,
    "Vbi": 0.70,    "V": 0.4,       "phi": 0.28,    "Phi_f": 0.28,
    "Vt": 0.02585,  "Vb": 0.0,
    "mu": 0.012,    "mu_eff": 0.012,"mu_e": 0.05,   "mu_h": 0.018,
    "Id": 1e-5,     "Ids": 1e-5,    "Idsat": 1e-5,  "Iref": 2e-6,
    "gm": 8e-4,     "ro": 3e5,
    "J": 2e5,       "Jo": 1.0,      "J0": 1.0,
    "NA": 2e24,     "ND": 2e24,     "ni": 1.5e16,
    "Nc": 2.8e25,   "Nv": 1.04e25,  "N": 2e24,
    "k": 1.38e-23,  "q": 1.6e-19,   "T": 300,       "n": 1.0,
    "eps": 1.04e-10,"eps_s": 1.04e-10,"eps_ox": 3.45e-11,
    "xd": 5e-9,     "Wdep": 5e-9,   "Ln": 1e-6,     "Lp": 1e-6,
    "Dn": 1.2e-3,   "Dp": 4e-4,     "vsat": 6e4,
    "Qdep": 5e-3,   "Qinv": 5e-2,
    "Cdep": 6e-3,   "Cd": 2.5e-2,
    "gamma": 0.2,   "lam": 0.25,    "DIBL": 0.15,
    "Av": 10.0,     "SS": 0.068,    "S": 0.068,
    "E": 8e6,       "E_max": 4e7,
    "I": 1e-5,      "C": 5e-16,     "Q": 3e-18,
    "O": 1.0,       "S0": 0.064,
}

# Symbols added to the vocabulary after initial node definitions.
# These don't vary strongly with technology node, so one shared value is fine.
_COMMON_EXTRAS = {
    # Resistance / conductance
    "R":     1e3,       "Rs":    50.0,   "Rd":    50.0,   "Rch":  500.0,
    "gds":   1e-5,      "gmb":   2e-4,
    # Supply / I-O voltages
    "Vdd":   1.0,       "Vcc":   1.8,    "Vin":   0.5,    "Vout":  0.9,
    "Vid":   0.01,      "Vod":   0.5,    "Vcm":   0.9,    "Vfb":  -0.9,
    # SPICE terminal voltages (after normalisation from vDS/vGS etc.)
    "Vgd":   0.35,      "Vgb":   0.5,    "Vbd":   0.0,    "Vsb":   0.0,
    "Vb":    0.0,       "Vn":    0.0,    "Vp":    0.0,
    # Barrier / surface potential
    "phi_b": 0.32,      "phi_s": 0.32,   "phi_f": 0.32,
    # Capacitances (total, not per-area)
    "Cgs":   5e-15,     "Cgd":   1e-15,  "Cdb":   2e-15,  "Csb":  2e-15,
    "Cj":    5e-4,
    # Bandgap / energy (in Joules)
    "Eg":    1.8e-19,   "Ei":    0.0,    "Ef":    0.0,    "kT":   4.14e-21,
    # Time / frequency
    "tau":   1e-9,      "tau_n": 1e-6,   "tau_p": 1e-6,
    "f":     1e9,       "fT":    5e10,   "omega": 6.28e9,
    "omegar": 6.28e9,   "omega0": 6.28e9, "omegac": 6.28e9,  # resonant/cutoff
    # Mobility aliases
    "mu0":   0.04,      "mu_n":  0.135,  "mu_p":  0.048,
    # Dimensionless factors
    "alpha": 0.99,      "beta":  100.0,  "eta":   0.8,
    "m":     1.0,       "p":     1.0,
    # Lengths (generic)
    "h":     1e-8,      "d":     1e-9,   "r":     5e-9,
    # Effective mass [kg]  (SI: kg = V·C·s²/m² — used numerically only)
    "me":    9.11e-31,  "meff":  1e-31,
}
for _node in (_NODE_100NM, _NODE_22NM, _NODE_5NM):
    for _k, _v in _COMMON_EXTRAS.items():
        _node.setdefault(_k, _v)

# Middle node used as the single test_values dict (backward compat)
_NOMINAL = _NODE_22NM

TECH_NODES = [_NODE_100NM, _NODE_22NM, _NODE_5NM]
NODE_LABELS = ["100nm", "22nm", "5nm"]

# ── Per-symbol expected output ranges ─────────────────────────────────────────
#
# (lo, hi) in SI units.  The result abs value must land in [lo, hi] for a node
# to count as "passing".  Ranges are deliberately wide to account for
# sub-threshold, saturation, and fringe cases — we only reject clearly absurd
# values.
#
RESULT_RANGES: dict[str, tuple[float, float]] = {
    # Drain current  (from deep sub-threshold to large-W saturation)
    "Id":     (1e-15, 5e-2),
    "Ids":    (1e-15, 5e-2),
    "Idsat":  (1e-14, 5e-2),
    # Transconductance
    "gm":     (1e-8,  1e-1),
    # Output resistance
    "ro":     (50,    1e9),
    # Threshold / bias voltages
    "Vth":    (0.05,  2.5),
    "Vth0":   (0.05,  2.5),
    "Vov":    (0.01,  2.0),
    "Vt":     (0.015, 0.04),     # thermal voltage kT/q ≈ 26 mV
    # Subthreshold swing  (V/dec)
    "SS":     (0.055, 0.50),
    "S":      (0.055, 0.50),
    # Current density  (A/m²) — very wide: leakage to IGBT
    "J":      (1e-8,  1e10),
    "Jo":     (1e-12, 1e6),
    "J0":     (1e-12, 1e6),
    # Voltage gain  (dimensionless)
    "Av":     (1e-3,  1e5),
    # DIBL coefficient  (dimensionless V/V ratio)
    "DIBL":   (1e-4,  0.5),
    # Depletion / inversion charges  (C/m²)
    "Qdep":   (1e-5,  1.0),
    "Qinv":   (1e-5,  1.0),
    # Depletion width / diffusion lengths  (m)
    "xd":     (1e-9,  1e-5),
    "Wdep":   (1e-9,  1e-5),
    "Ln":     (1e-7,  1e-3),
    "Lp":     (1e-7,  1e-3),
    # Intrinsic carrier concentration  (1/m³)
    "ni":     (1e10,  1e20),
    # Electric field  (V/m)
    "E":      (1e2,   1e9),
    "E_max":  (1e4,   1e10),
    # Capacitance per area  (F/m²)
    "Cox":    (1e-4,  1.0),
    "Cd":     (1e-5,  1.0),
    "Cdep":   (1e-5,  1.0),
    # Mobility  (m²/Vs)
    "mu":     (1e-4,  0.5),
    "mu_eff": (1e-4,  0.5),
    "mu_e":   (1e-3,  0.5),
    "mu_h":   (5e-4,  0.2),
    # Velocity  (m/s)
    "vsat":   (1e3,   2e5),
    # Reference current
    "Iref":   (1e-10, 1e-2),
    # Generic current / charge (single-letter overrides)
    "I":      (1e-15, 5e-2),
    "Q":      (1e-20, 1e-10),
    # Geometry / lengths — catches `L = 100*m*n` style garbage
    "L":      (1e-9,  1e-4),     # channel / device length  [m]
    "W":      (1e-9,  1e-3),     # gate width  [m]
    "tox":    (1e-10, 1e-7),     # oxide thickness  [m]
    "EOT":    (1e-10, 1e-8),     # equivalent oxide thickness  [m]
    "h":      (1e-10, 1e-6),     # layer thickness  [m]
    "d":      (1e-11, 1e-6),     # distance / diameter  [m]
    "r":      (1e-11, 1e-6),     # radius  [m]
    # Resistance
    "R":      (1.0,   1e9),      # resistance  [Ω]
    "Rs":     (0.1,   1e6),
    "Rd":     (0.1,   1e6),
    # Terminal voltages — lo=0 so val==0 passes (saturation edge is valid)
    "Vgd":    (0.0,   2.5),
    "Vgb":    (0.0,   3.0),
    "Vbd":    (0.0,   1.0),
    "Vgs":    (0.0,   3.0),
    "Vds":    (0.0,   2.5),
    "Vbs":    (0.0,   2.0),
    "V":      (0.0,   5.0),
}

# Generic fallback bounds for any LHS not in RESULT_RANGES
_GENERIC_LO  = 1e-30
_GENERIC_HI  = 1e15


class NumericalValidator:
    """
    Multi-point numerical validator.

    evaluate(lhs, rhs) → str
      Tests rhs at each of the three technology nodes.
      Requires ≥ 2/3 nodes to give a result inside the expected range for the
      LHS symbol (or inside the generic bounds if LHS is unknown).
      Returns "[OK] ..." on success, "[WARN] ..." or "[FAIL] ..." otherwise.
    """

    def __init__(self):
        # Backward-compatible single-point dict — used by new_checker._coverage()
        self.test_values: dict[str, float] = dict(_NOMINAL)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _eval_at(rhs_expr, node: dict) -> float | None:
        """Substitute node values into rhs_expr. Returns float or None on fail."""
        try:
            result = rhs_expr.subs(node)
            remaining = result.free_symbols
            if remaining:
                # Fallback: indexed circuit params (gm5→gm, go1/GL→gds)
                fallback = {}
                for sym in remaining:
                    name = sym.name
                    base = _re.sub(r'\d+$', '', name)   # strip trailing digits
                    if base in node:
                        fallback[sym] = node[base]
                    elif base in _CONDUCTANCE_BASES:
                        fallback[sym] = node.get('gds', 1e-5)
                    elif base in _GM_BASES:
                        fallback[sym] = node.get('gm', 1e-3)
                if fallback:
                    result = result.subs(fallback)
                    remaining = result.free_symbols
            if remaining:
                return None          # unresolved symbol at this node
            val = float(result.evalf())
            if math.isnan(val) or math.isinf(val):
                return None
            return val
        except Exception:
            return None

    @staticmethod
    def _in_range(val: float, lo: float, hi: float) -> bool:
        av = abs(val)
        # Allow val == 0 only if lo == 0
        if val == 0.0:
            return lo == 0.0
        return lo <= av <= hi

    def _result_bounds(self, lhs_expr) -> tuple[float, float, str]:
        """Return (lo, hi, label) for the LHS symbol."""
        if lhs_expr is not None and isinstance(lhs_expr, sp.Symbol):
            name = lhs_expr.name
            if name in RESULT_RANGES:
                lo, hi = RESULT_RANGES[name]
                return lo, hi, name
        return _GENERIC_LO, _GENERIC_HI, "generic"

    # ── public API ────────────────────────────────────────────────────────────

    def evaluate(self, lhs_expr, rhs_expr) -> str:
        # ── Special case: RHS is a pure constant (no free symbols) ───────────
        # Evaluating `3.subs(node)` always returns 3 and trivially passes the
        # range check, even for garbage equations like `C*E*h = 3`.
        # Fix: compute LHS at each tech node and verify it matches the constant
        # within 3 orders of magnitude.
        rhs_free = rhs_expr.free_symbols if hasattr(rhs_expr, "free_symbols") else set()
        if not rhs_free:
            try:
                rhs_const = float(rhs_expr.evalf())
            except Exception:
                rhs_const = None
            if rhs_const is not None and rhs_const != 0:
                lhs_vals = [self._eval_at(lhs_expr, nd) for nd in TECH_NODES]
                lhs_vals = [v for v in lhs_vals if v is not None]
                if not lhs_vals:
                    return "[WARN] Unresolved symbols remain: LHS not evaluable vs constant RHS."
                consistent = sum(
                    1 for v in lhs_vals
                    if 1e-3 <= abs(v) / abs(rhs_const) <= 1e3
                )
                if consistent >= 2:
                    return (f"[OK] Consistent with constant RHS={rhs_const:.3e}: "
                            f"LHS≈{lhs_vals[0]:.3e}")
                return (f"[FAIL] LHS ({lhs_vals[0]:.3e}) inconsistent with "
                        f"constant RHS ({rhs_const:.3e}) — likely extracted number")
            # RHS = 0: only passes if LHS also evaluates to ~0
            if rhs_const == 0:
                lhs_vals = [self._eval_at(lhs_expr, nd) for nd in TECH_NODES]
                lhs_vals = [v for v in lhs_vals if v is not None]
                if not lhs_vals:
                    return "[WARN] Unresolved symbols remain: LHS not evaluable vs zero RHS."
                near_zero = sum(1 for v in lhs_vals if abs(v) < 1e-10)
                if near_zero >= 2:
                    return "[OK] LHS evaluates near zero — consistent with RHS=0."
                return f"[FAIL] LHS ({lhs_vals[0]:.3e}) not near zero; RHS=0."

        # ── Normal case: evaluate RHS at each tech node ──────────────────────
        lo, hi, label = self._result_bounds(lhs_expr)
        node_results: list[tuple[str, float | None, bool]] = []

        for node_vals, node_name in zip(TECH_NODES, NODE_LABELS):
            val = self._eval_at(rhs_expr, node_vals)
            if val is None:
                node_results.append((node_name, None, False))
            else:
                in_range = self._in_range(val, lo, hi)
                node_results.append((node_name, val, in_range))

        evaluated = [(nm, v, ok) for nm, v, ok in node_results if v is not None]
        if not evaluated:
            return "[WARN] Unresolved symbols remain: substitution failed at all nodes."

        pass_count = sum(1 for _, _, ok in evaluated if ok)
        fail_nodes = [(nm, v) for nm, v, ok in evaluated if not ok]

        if pass_count >= 2:
            vals_str = ", ".join(f"{nm}:{v:.3e}" for nm, v, _ in evaluated)
            return f"[OK] Numerically plausible ({label} in [{lo:.0e},{hi:.0e}]): {vals_str}"

        if not fail_nodes:
            nm, v, _ = evaluated[0]
            return (f"[WARN] Only one node resolved ({nm}={v:.3e}); "
                    f"need ≥2 for confidence.")
        details = "; ".join(
            f"{nm}={v:.3e} (out of [{lo:.0e},{hi:.0e}])" for nm, v in fail_nodes
        )
        return f"[FAIL] Out of physical range for '{label}': {details}"
