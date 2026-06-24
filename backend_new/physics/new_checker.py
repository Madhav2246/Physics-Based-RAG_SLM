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
    # -- Carrier physics --------------------------------------------------
    "ni":      ({"L": -3},                              1.0e16),    # intrinsic carrier conc [1/m^3]
    "Nc":      ({"L": -3},                              2.8e25),    # eff. DOS conduction band
    "Nv":      ({"L": -3},                              1.04e25),   # eff. DOS valence band
    # -- Permittivities ---------------------------------------------------
    "eps_s":   ({"I": 1, "T": 1, "V": -1, "L": -1},    1.04e-10),  # Si permittivity [F/m]
    "eps_ox":  ({"I": 1, "T": 1, "V": -1, "L": -1},    3.45e-11),  # oxide permittivity [F/m]
    # -- Mobilities -------------------------------------------------------
    "mu_eff":  ({"L": 2, "V": -1, "T": -1},             0.04),      # effective mobility [m^2/Vs]
    "mu_e":    ({"L": 2, "V": -1, "T": -1},             0.135),     # electron mobility
    "mu_h":    ({"L": 2, "V": -1, "T": -1},             0.048),     # hole mobility
    "mu0":     ({"L": 2, "V": -1, "T": -1},             0.04),      # reference mobility
    "mu_n":    ({"L": 2, "V": -1, "T": -1},             0.135),     # electron mobility alias
    "mu_p":    ({"L": 2, "V": -1, "T": -1},             0.048),     # hole mobility alias
    # -- Lengths / widths -------------------------------------------------
    "EOT":     ({"L": 1},                               1.0e-9),    # equiv. oxide thickness [m]
    "Ln":      ({"L": 1},                               1.0e-5),    # electron diffusion length [m]
    "Lp":      ({"L": 1},                               1.0e-5),    # hole diffusion length [m]
    "Wdep":    ({"L": 1},                               1.0e-7),    # depletion width [m]
    "h":       ({"L": 1},                               1.0e-8),    # layer thickness [m]
    "d":       ({"L": 1},                               1.0e-9),    # distance/diameter [m]
    "r":       ({"L": 1},                               5.0e-9),    # radius [m]
    # -- Diffusivities / velocity -----------------------------------------
    "Dn":      ({"L": 2, "T": -1},                      3.5e-3),    # electron diffusivity [m^2/s]
    "Dp":      ({"L": 2, "T": -1},                      1.2e-3),    # hole diffusivity [m^2/s]
    "vsat":    ({"L": 1, "T": -1},                      1.0e5),     # saturation velocity [m/s]
    # -- Charges / capacitances -------------------------------------------
    "Qdep":    ({"I": 1, "T": 1, "L": -2},              1.0e-3),    # depletion charge/area [C/m^2]
    "Qinv":    ({"I": 1, "T": 1, "L": -2},              1.0e-2),    # inversion charge/area [C/m^2]
    "Cdep":    ({"I": 1, "T": 1, "V": -1, "L": -2},     1.0e-3),    # depletion cap/area [F/m^2]
    "Cgs":     ({"I": 1, "T": 1, "V": -1},              5.0e-15),   # gate-source cap [F]
    "Cgd":     ({"I": 1, "T": 1, "V": -1},              1.0e-15),   # gate-drain cap [F]
    "Cdb":     ({"I": 1, "T": 1, "V": -1},              2.0e-15),   # drain-bulk cap [F]
    "Csb":     ({"I": 1, "T": 1, "V": -1},              2.0e-15),   # source-bulk cap [F]
    "Cj":      ({"I": 1, "T": 1, "V": -1, "L": -2},     5.0e-4),    # junction cap/area [F/m^2]
    # -- Resistance / conductance -----------------------------------------
    "R":       ({"V": 1, "I": -1},                      1.0e3),     # resistance [Ω]
    "Rs":      ({"V": 1, "I": -1},                      50.0),      # source resistance [Ω]
    "Rd":      ({"V": 1, "I": -1},                      50.0),      # drain resistance [Ω]
    "Rch":     ({"V": 1, "I": -1},                      500.0),     # channel resistance [Ω]
    "gds":     ({"I": 1, "V": -1},                      1.0e-5),    # drain-source conductance [A/V]
    "gmb":     ({"I": 1, "V": -1},                      2.0e-4),    # body transconductance [A/V]
    # -- Supply / I/O voltages --------------------------------------------
    "Vdd":     ({"V": 1},                               1.0),       # supply voltage [V]
    "Vcc":     ({"V": 1},                               1.8),       # supply voltage [V]
    "Vin":     ({"V": 1},                               0.5),       # input voltage [V]
    "Vout":    ({"V": 1},                               0.9),       # output voltage [V]
    "Vid":     ({"V": 1},                               0.01),      # differential input [V]
    "Vod":     ({"V": 1},                               0.5),       # differential output [V]
    "Vcm":     ({"V": 1},                               0.9),       # common-mode voltage [V]
    "Vfb":     ({"V": 1},                              -0.9),       # flatband voltage [V]
    # -- Energy / bandgap -------------------------------------------------
    "Eg":      ({"V": 1, "C": 1},                       1.8e-19),   # bandgap energy [J] (1.12 eV)
    "Ei":      ({"V": 1, "C": 1},                       0.0),       # intrinsic Fermi level [J]
    "Ef":      ({"V": 1, "C": 1},                       0.0),       # Fermi level [J]
    "kT":      ({"V": 1, "C": 1},                       4.14e-21),  # thermal energy [J]
    # -- Time / frequency -------------------------------------------------
    "tau":     ({"T": 1},                               1.0e-9),    # carrier lifetime / time const [s]
    "tau_n":   ({"T": 1},                               1.0e-6),    # electron lifetime [s]
    "tau_p":   ({"T": 1},                               1.0e-6),    # hole lifetime [s]
    "f":       ({"T": -1},                              1.0e9),     # frequency [Hz]
    "fT":      ({"T": -1},                              5.0e10),    # transit frequency [Hz]
    "omega":   ({"T": -1},                              6.28e9),    # angular frequency [rad/s]
    # -- Dimensionless ratios / gains -------------------------------------
    "alpha":   ({},                                      0.99),      # BJT transport factor
    "beta":    ({},                                      100.0),     # BJT current gain
    "eta":     ({},                                      0.8),       # efficiency / ideality
    # NOTE: single-letter "m" and "p" intentionally omitted — too ambiguous
    # (m = mass OR meters prefix; p = hole density OR power) and would let
    # dimensionless garbage equations pass numerical checks trivially.

    # -- Fix B: Unambiguous single-letter physics symbols --------------------
    # Scope: semiconductor / device physics.  Each entry documents the chosen
    # interpretation and why it is unambiguous in this domain.
    "x":  ({"L": 1},                               5e-9),    # position along device [m]
    "t":  ({"T": 1},                               1e-9),    # time / carrier lifetime [s]
                                                             #   (tox is already the oxide thickness symbol)
    "D":  ({"L": 2, "T": -1},                      2e-3),    # diffusion coefficient [m²/s]
                                                             #   semiconductor context; electric displacement
                                                             #   (C/m²) is rarely written bare 'D' here
    "K":  ({},                                      11.7),   # relative dielectric constant [dimensionless]
                                                             #   Si ε_r ≈ 11.7; HfO2 ≈ 25
    "g":  ({"I": 1, "V": -1},                      1e-4),   # conductance [A/V]
    "i":  ({"I": 1},                               1e-4),   # lowercase current [A]
                                                             #   distinct from uppercase I in our vocab
    "Nf": ({},                                      1.0),    # number of gate fingers [dimensionless]
    "a":  ({"L": -4},                               1e20),   # doping gradient (linearly graded jxn) [m⁻⁴]
    # -- Effective mass ---------------------------------------------------
    "me":      ({"C": -1, "T": 2, "V": -1, "L": -2},   9.11e-31),  # electron mass [kg] ≈ V·C·s²/m²
    "meff":    ({"C": -1, "T": 2, "V": -1, "L": -2},   1e-31),     # effective mass [kg]
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

# Also inject EXTRA symbols into every tech node so multi-point evaluation
# can resolve them (the base node dicts don't include EXTRA symbols).
from physics.numerical_validator import TECH_NODES as _TECH_NODES
for _name, (_d, _v) in EXTRA.items():
    for _node in _TECH_NODES:
        _node.setdefault(_name, _v)


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

    Soup signature = equations dominated by UNKNOWN single letters with no
    known multi-char symbol to anchor them (e.g. E*o*c*h*o*t, Av=o*t*u*v).

    Key fix over previous version:
      Rule 1 now uses UNKNOWN singles (not all singles).  This stops rejecting
      valid equations like `V = I*R` (V and I are KNOWN; only R is unknown →
      unknown_singles=1, well below threshold) while still catching corpus
      garbage like `D*V = 0` when all symbols happen to be unknown.

    Reject when:
      - there is NO known multi-char symbol AND >=3 *unknown* single-letter
        symbols, OR
      - there are >=4 unknown single-letter symbols regardless of multi-char
        (catches Av = o*t*u*v style hallucinations).
    """
    if not free:
        return False
    names            = [s.name for s in free]
    singles          = [n for n in names if len(n) == 1]
    multis           = [n for n in names if len(n) >= 2]
    unknown_singles  = [n for n in singles if n not in _KNOWN]
    known_multis     = [n for n in multis  if n in _KNOWN]
    if len(known_multis) == 0 and len(unknown_singles) >= 3:
        return True
    if len(unknown_singles) >= 4:
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


def _coverage(lhs, rhs) -> float:
    """
    Coverage = fraction of ALL free symbols (both sides) that have known test
    values.  Using only RHS was wrong: equations like `F*O = 2` had RHS=2
    (no free symbols) → coverage 1.0 even though LHS symbols were unknown.
    Equations like `Av = o*t*u*v` now score 1/5 instead of 0/0=1.0.
    """
    free = set()
    if lhs is not None:
        free |= lhs.free_symbols
    if rhs is not None:
        free |= rhs.free_symbols
    if not free:
        return 1.0
    return sum(1 for s in free if s.name in _num.test_values) / len(free)


def _guard_trivial_dimensionless(lhs, rhs, dimensional: bool, dim_msg: str) -> tuple:
    """
    Guard against false-positive dimensional passes where BOTH sides evaluate
    to {} (dimensionless) solely because unknown symbols default to {}.
    Example: `F*O = 2` — F unknown → {}, O → {} → LHS={}, RHS={} → [OK].
    Fix: if dim passes AND both sides are {} AND any symbol is outside dim_map,
    the check is unresolvable — do NOT award the +1 point.
    """
    if not dimensional:
        return dimensional, dim_msg
    all_free = (lhs.free_symbols if lhs else set()) | (rhs.free_symbols if rhs else set())
    has_unknowns = any(s.name not in _dim.dim_map for s in all_free)
    if not has_unknowns:
        return dimensional, dim_msg  # all symbols known → genuine pass
    lhs_dims = _dim._simplify(_dim.evaluate(lhs))
    rhs_dims = _dim._simplify(_dim.evaluate(rhs))
    if lhs_dims == {} == rhs_dims:
        # Both sides dimensionless only because unknowns defaulted to {} — unresolvable
        return False, "[WARN] Dimensional check unresolvable: both sides {} due to unknown symbols."
    return dimensional, dim_msg


# Fix C: coverage threshold for dimensional credit.
# Dimensional check is only awarded when at least this fraction of ALL free
# symbols (LHS ∪ RHS) have known SI dimensions.  Below the threshold the
# check is "unresolvable" — we cannot verify, so we neither pass nor fail.
# Rationale: BSIM4 / compact-model parameters (K1, UA, UB, …) absent from
# our vocabulary should not silently default to dimensionless and fake a pass,
# but they also should not penalise equations that are otherwise correct.
_DIM_COVERAGE_GATE = 0.70   # ≥70 % of symbols must be in dim_map


def _coverage_for_dim(lhs, rhs) -> float:
    """Fraction of ALL free symbols that are in the dimension map."""
    free = set()
    if lhs is not None:
        free |= lhs.free_symbols
    if rhs is not None:
        free |= rhs.free_symbols
    if not free:
        return 1.0
    return sum(1 for s in free if s.name in _dim.dim_map) / len(free)


def _score_parsed(lhs, rhs, sym_msg: str) -> dict:
    """Shared scoring logic for an already-parsed (lhs, rhs) pair."""
    equation = f"{lhs} = {rhs}"
    cov = _coverage(lhs, rhs)
    dim_cov = _coverage_for_dim(lhs, rhs)

    # Fix C: gate dimensional check on symbol coverage
    if dim_cov >= _DIM_COVERAGE_GATE:
        dim_msg = _dim.check_equation(lhs, rhs)
        dimensional = "[OK]" in dim_msg
        dimensional, dim_msg = _guard_trivial_dimensionless(lhs, rhs, dimensional, dim_msg)
    else:
        dimensional = False
        dim_msg = (f"[WARN] Dimensional check skipped: only {dim_cov*100:.0f}% of symbols "
                   f"have known dimensions (need ≥{_DIM_COVERAGE_GATE*100:.0f}%).")

    num_msg = _num.evaluate(lhs, rhs)
    numerical = "[OK]" in num_msg
    total = 1.0 + float(dimensional) + float(numerical) + cov  # parseable=True
    return {
        "parseable": True, "dimensional": dimensional, "numerical": numerical,
        "coverage": cov >= 0.999, "coverage_frac": round(cov, 4),
        "dim_coverage_frac": round(dim_cov, 4),
        "total": round(total, 4), "equation": equation,
        "sym_msg": sym_msg, "dim_msg": dim_msg, "num_msg": num_msg,
    }


def _score_none(sym_msg: str) -> dict:
    return {
        "parseable": False, "dimensional": False, "numerical": False,
        "coverage": False, "coverage_frac": 0.0, "dim_coverage_frac": 0.0,
        "total": 0.0, "equation": "",
        "sym_msg": sym_msg,
        "dim_msg": "[WARN] Dimensional check skipped.",
        "num_msg": "[WARN] Numerical check skipped.",
    }


def score_text(text: str, model_label: str = "") -> dict:
    text = strip_latex(text or "")
    lhs, rhs, sym_msg = validate(text)
    if lhs is None:
        return _score_none(sym_msg)
    return _score_parsed(lhs, rhs, sym_msg)


def score_equation(eq_str: str) -> dict:
    """
    Score an already-extracted equation string (e.g. from answers_dump.jsonl
    or a stored stage1 JSON).  Skips candidate-extraction; parses the string
    directly and applies the same letter-soup + dim + coverage fixes.
    Returns the same dict shape as score_text().
    """
    if not eq_str or "=" not in eq_str:
        return _score_none("[WARN] No equation string provided.")
    norm = _val.normalize_equation(_greek_pre(strip_latex(eq_str)))
    if "=" not in norm:
        return _score_none("[WARN] Normalisation removed '=' sign.")
    try:
        ls, rs = norm.split("=", 1)
        lhs = parse_expr(ls.strip(), local_dict=_val.symbols, transformations=_TRANSFORM)
        rhs = parse_expr(rs.strip(), local_dict=_val.symbols, transformations=_TRANSFORM)
    except Exception as exc:
        return _score_none(f"[WARN] Parsing failed: {type(exc).__name__}: {exc}")
    free = _free(lhs) | _free(rhs)
    if _is_letter_soup(free):
        return _score_none("[WARN] No valid equation (letter-soup rejected).")
    return _score_parsed(lhs, rhs, "[OK] Equation parsed successfully")
