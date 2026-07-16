"""
physics/physics_scorer.py
-------------------------
Single source of truth for the 0-4 physics score (Tier-2 fair scoring).

Previously three scripts (compare_physics_validators, quick_eval_20,
eval_3model_compare) each carried their own copy of this logic, all sharing the
same two flaws:

  * Coverage was ALL-OR-NOTHING — one unknown symbol zeroed the whole point, so
    a near-perfect equation scored the same as gibberish. It is now PARTIAL:
    the fraction of RHS symbols with a known test value.
  * The sub-checks were COUPLED through the numerical message string
    (coverage depended on "Unresolved"/"failed" substrings in num_msg). They
    are now computed independently from the parsed expression.

Score (max 4.0):
  +1.0  parseable    an equation was found and SymPy parsed it
  +1.0  dimensional  LHS and RHS resolve to the same physical dimension
  +1.0  numerical    full substitution gives a physically plausible value
  +cov  coverage     fraction (0..1) of RHS symbols with a known test value

LaTeX is stripped before scoring so every caller scores the same normalized
text (the 0.5B model frequently emits \\[ ... \\] blocks that crash SymPy).
"""
from __future__ import annotations

import re

from physics.equation_validator import EquationValidator
from physics.dimension_checker import DimensionChecker
from physics.numerical_validator import NumericalValidator

# Module-level singletons — these validators are stateless and cheap to share.
_validator = EquationValidator()
_dim_checker = DimensionChecker()
_num_validator = NumericalValidator()


def strip_latex(text: str) -> str:
    """Remove LaTeX math wrappers but keep the equation content inside them."""
    if not text:
        return ""
    # Replace LaTeX math wrappers with spaces, preserving the content inside
    text = text.replace(r'\[', ' ').replace(r'\]', ' ')
    text = text.replace(r'\(', ' ').replace(r'\)', ' ')
    text = text.replace('$$', ' ').replace('$', ' ')
    # Translate common LaTeX ops to Python
    text = text.replace('\\frac', '/').replace('\\sqrt', 'sqrt')
    text = text.replace('\\cdot', '*').replace('\\times', '*')
    # Remove remaining backslash commands (e.g. \text, \mu, etc.)
    text = re.sub(r'\\[a-zA-Z]+', ' ', text)
    return text


def coverage_fraction(rhs_expr) -> float:
    """
    Fraction of the RHS's free symbols that have a known realistic test value.
    1.0 if fully numeric (no free symbols), 0.0 if nothing is recognized.
    """
    if rhs_expr is None:
        return 0.0

    def _get_free(expr) -> set:
        if isinstance(expr, (list, tuple, set)):
            s = set()
            for x in expr:
                s |= _get_free(x)
            return s
        if hasattr(expr, "free_symbols"):
            return expr.free_symbols
        return set()

    free = _get_free(rhs_expr)
    if not free:
        return 1.0
    resolved = sum(1 for s in free if s.name in _num_validator.test_values)
    return resolved / len(free)


def score_text(text: str, model_label: str = "") -> dict:
    """
    Run all validators on a text answer and return a structured score.

    The returned dict is a superset of the old per-script format, so existing
    aggregate/print code keeps working. New fields: ``coverage_frac`` (partial
    credit) and a float ``total``.
    """
    text = strip_latex(text or "")
    lhs, rhs, sym_msg = _validator.validate(text)

    parseable = lhs is not None
    dim_msg = "[WARN] Dimensional check skipped."
    num_msg = "[WARN] Numerical check skipped."
    dimensional = False
    numerical = False
    cov = 0.0
    equation = ""

    if parseable:
        equation = f"{lhs} = {rhs}"
        dim_msg = _dim_checker.check_equation(lhs, rhs)
        num_msg = _num_validator.evaluate(lhs, rhs)
        dimensional = "[OK]" in dim_msg
        numerical = "[OK]" in num_msg
        cov = coverage_fraction(rhs)

    total = float(parseable) + float(dimensional) + float(numerical) + cov

    return {
        "parseable":     parseable,
        "dimensional":   dimensional,
        "numerical":     numerical,
        "coverage":      cov >= 0.999,      # bool, back-compat = FULL coverage
        "coverage_frac": round(cov, 4),     # NEW: partial credit (0..1)
        "total":         round(total, 4),
        "equation":      equation,
        "sym_msg":       sym_msg,
        "dim_msg":       dim_msg,
        "num_msg":       num_msg,
    }
