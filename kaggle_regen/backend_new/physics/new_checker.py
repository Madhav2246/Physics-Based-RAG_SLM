"""
physics/new_checker.py
----------------------
SEPARATE, self-contained physics scorer for the Stage-1 re-evaluation.
Does NOT modify the original validators — drop this file and revert to
physics_scorer.py if results are worse.

Two fixes over the original scorer:

  (A) Letter-soup rejection.
      The original extractor parses mangled prose into "equations" made of
      single letters multiplied together, e.g.  E*o = E**2*c*h*o**3*t .
      These parse (inflating parse-rate) but are physically meaningless and
      legitimately fail the dimensional check. We now reject any equation
      whose free symbols are mostly OUT-OF-VOCABULARY single letters.

  (B) Expanded device-physics vocabulary.
      The original dim_map / test_values cover ~40 core MOSFET symbols. The
      golden set spans broader device physics (tunnelling, junctions, memory),
      so symbols like ni, eps_s, eps_ox, mu_eff, EOT, Dn... were unknown ->
      RHS collapsed to dimensionless (dim fail) and coverage dropped. We add a
      curated set of high-confidence SI dimensions + realistic values.

Public API (mirrors physics_scorer.score_text):
    score_text(text, model_label="") -> dict with keys:
        parseable, dimensional, numerical, coverage, coverage_frac,
        total, equation, sym_msg, dim_msg, num_msg
"""
from __future__ import annotations

import re
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication,
)

from physics.equation_validator import EquationValidator
from physics.dimension_checker import DimensionChecker
from physics.numerical_validator import NumericalValidator

# Transformations WITHOUT `split_symbols`.
# The original validator used implicit_multiplication_application, which bundles
# split_symbols → it shatters every unknown compound symbol into single letters
# (EOT → E*O*T, Avd → A*v*d, IGIDL → I*G*I*L*D). That turned real equations into
# letter-soup and falsely inflated parse-rate. Using plain implicit_multiplication
# keeps compound symbols atomic (EOT stays Symbol('EOT')), so equations parse
# correctly; unknown symbols simply stay unknown (honest N/A on dim/coverage).
_TRANSFORM = standard_transformations + (implicit_multiplication,)

# --------------------------------------------------------------------------
# (B) Curated device-physics vocabulary expansion.
#     Base dimension keys: I(current) V(voltage) L(length) T(time)
#                          K(temperature) C(charge).  F/m = I·T·V⁻¹·L⁻¹.
#     Only high-confidence SI dimensions are added (wrong dims would be worse
#     than leaving a symbol unknown).
# --------------------------------------------------------------------------
EXTRA = {
    # name      dimension dict                         test value (SI)
    "ni":      ({"L": -3},                              1.0e16),    # intrinsic carrier conc [1/m^3]
    "Nc":      ({"L": -3},                              2.8e25),    # eff. DOS conduction band
    "Nv":      ({"L": -3},                              1.04e25),   # eff. DOS valence band
    "eps_s":   ({"I": 1, "T": 1, "V": -1, "L": -1},    1.04e-10),  # Si permittivity [F/m]
    "eps_ox":  ({"I": 1, "T": 1, "V": -1, "L": -1},    3.45e-11),  # oxide permittivity [F/m]
    "mu_eff":  ({"L": 2, "V": -1, "T": -1},            0.04),      # effective mobility [m^2/Vs]
    "mu_e":    ({"L": 2, "V": -1, "T": -1},            0.135),     # electron mobility
    "mu_h":    ({"L": 2, "V": -1, "T": -1},            0.048),     # hole mobility
    "EOT":     ({"L": 1},                              1.0e-9),    # equiv. oxide thickness [m]
    "Ln":      ({"L": 1},                              1.0e-5),    # electron diffusion length [m]
    "Lp":      ({"L": 1},                              1.0e-5),    # hole diffusion length [m]
    "Wdep":    ({"L": 1},                              1.0e-7),    # depletion width [m]
    "Dn":      ({"L": 2, "T": -1},                     3.5e-3),    # electron diffusivity [m^2/s]
    "Dp":      ({"L": 2, "T": -1},                     1.2e-3),    # hole diffusivity [m^2/s]
    "Qdep":    ({"I": 1, "T": 1, "L": -2},             1.0e-3),    # depletion charge/area [C/m^2]
    "Qinv":    ({"I": 1, "T": 1, "L": -2},             1.0e-2),    # inversion charge/area [C/m^2]
    "vsat":    ({"L": 1, "T": -1},                     1.0e5),     # saturation velocity [m/s]
    "Cdep":    ({"I": 1, "T": 1, "V": -1, "L": -2},    1.0e-3),    # depletion cap/area [F/m^2]
}

# Greek / variant forms the SLM emits -> ASCII names matching EXTRA / core dict.
# Applied BEFORE the base normalizer so the symbols resolve.
_GREEK_PRE = [
    (r"ε_?s", "eps_s"), (r"ε_?ox", "eps_ox"), (r"ε_?0", "eps"),
    (r"μ_?eff", "mu_eff"), (r"μ_?e\b", "mu_e"), (r"μ_?h\b", "mu_h"),
    (r"μ_?n", "mu"), (r"μ", "mu"),
    (r"φ_?f", "Phi_f"), (r"φ", "phi"), (r"Φ_?f", "Phi_f"),
    (r"γ", "gamma"), (r"ε", "eps"), (r"Δ", "Delta"),
]

# --------------------------------------------------------------------------
# Shared, expanded validator instances.
# --------------------------------------------------------------------------
_val = EquationValidator()
_dim = DimensionChecker()
_num = NumericalValidator()

for _name, (_d, _v) in EXTRA.items():
    _val.symbols.setdefault(_name, sp.Symbol(_name))
    _dim.dim_map.setdefault(_name, _d)
    _num.test_values.setdefault(_name, _v)

_KNOWN = set(_val.symbols.keys())


# --------------------------------------------------------------------------
def strip_latex(text: str) -> str:
    if not text:
        return ""
    text = text.replace(r"\[", " ").replace(r"\]", " ")
    text = text.replace(r"\(", " ").replace(r"\)", " ")
    text = text.replace("$$", " ").replace("$", " ")
    text = text.replace("\\frac", "/").replace("\\sqrt", "sqrt")
    text = text.replace("\\cdot", "*").replace("\\times", "*")
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    return text


def _greek_pre(eq: str) -> str:
    for pat, repl in _GREEK_PRE:
        eq = re.sub(pat, repl, eq)
    return eq


def _is_letter_soup(free) -> bool:
    """
    (A) True only for prose mis-parsed as an equation — distinguished by
    STRUCTURE, not by unknown-symbol count (real physics equations legitimately
    use many symbols outside our dictionary, so an unknown-count rule would
    wrongly reject them).

    Soup signature = a pile of SINGLE letters with NO multi-character symbol to
    anchor it (e.g. E*o*c*h*o*t). Real equations almost always contain at least
    one multi-char symbol (Cox, Vth, Vgs, mu, eps, ...).
    Reject when:
      - there is NO multi-char symbol AND >=5 distinct single-letter symbols, OR
      - there are >=6 unknown single-letter symbols (overwhelming soup).
    """
    if not free:
        return False
    names   = [s.name for s in free]
    singles = [n for n in names if len(n) == 1]
    multis  = [n for n in names if len(n) >= 2]
    unknown_singles = [n for n in singles if n not in _KNOWN]
    if len(multis) == 0 and len(singles) >= 5:
        return True
    if len(unknown_singles) >= 6:
        return True
    return False


def _free(expr) -> set:
    if isinstance(expr, (list, tuple, set)):
        out = set()
        for x in expr:
            out |= _free(x)
        return out
    return expr.free_symbols if hasattr(expr, "free_symbols") else set()


def validate(text: str):
    """Best-of-candidates parse with letter-soup rejection. Returns (lhs,rhs,msg)."""
    candidates = _val._candidate_equations(text)
    if not candidates:
        return None, None, "[WARN] No equation detected."

    best_rank = best_lhs = best_rhs = None
    last_err = None
    for cand in candidates:
        norm = _val.normalize_equation(_greek_pre(cand))
        if "=" not in norm:
            continue
        try:
            ls, rs = norm.split("=", 1)
            lhs = parse_expr(ls.strip(), local_dict=_val.symbols, transformations=_TRANSFORM)
            rhs = parse_expr(rs.strip(), local_dict=_val.symbols, transformations=_TRANSFORM)
        except Exception as e:
            last_err = e
            continue
        if not (isinstance(lhs, sp.Basic) and isinstance(rhs, sp.Basic)):
            continue
        free = _free(lhs) | _free(rhs)
        if _is_letter_soup(free):
            continue                                  # (A) reject garbage
        known = sum(1 for s in free if s.name in _KNOWN)
        rank = (known, -(len(free) - known))
        if best_rank is None or rank > best_rank:
            best_rank, best_lhs, best_rhs = rank, lhs, rhs

    if best_rank is None:
        if last_err is not None:
            return None, None, f"[WARN] Parsing failed: {type(last_err).__name__}"
        return None, None, "[WARN] No valid equation (letter-soup rejected)."
    return best_lhs, best_rhs, "[OK] Equation parsed successfully"


def _coverage(rhs) -> float:
    if rhs is None:
        return 0.0
    free = rhs.free_symbols
    if not free:
        return 1.0
    return sum(1 for s in free if s.name in _num.test_values) / len(free)


def score_text(text: str, model_label: str = "") -> dict:
    text = strip_latex(text or "")
    lhs, rhs, sym_msg = validate(text)
    parseable = lhs is not None
    dim_msg = "[WARN] Dimensional check skipped."
    num_msg = "[WARN] Numerical check skipped."
    dimensional = numerical = False
    cov = 0.0
    equation = ""
    if parseable:
        equation = f"{lhs} = {rhs}"
        dim_msg = _dim.check_equation(lhs, rhs)
        num_msg = _num.evaluate(lhs, rhs)
        dimensional = "[OK]" in dim_msg
        numerical = "[OK]" in num_msg
        cov = _coverage(rhs)
    total = float(parseable) + float(dimensional) + float(numerical) + cov
    return {
        "parseable": parseable, "dimensional": dimensional, "numerical": numerical,
        "coverage": cov >= 0.999, "coverage_frac": round(cov, 4),
        "total": round(total, 4), "equation": equation,
        "sym_msg": sym_msg, "dim_msg": dim_msg, "num_msg": num_msg,
    }
