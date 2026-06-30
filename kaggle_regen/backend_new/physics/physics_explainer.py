"""
physics/physics_explainer.py

Converts raw validator outputs (terse strings) into:
  - plain-English reason: what was checked and what went wrong
  - feedback_hint: what the user should write in the correction box
  - verdict: PASS / WARN / FAIL per component

Used by:
  api_server.py          → Inspector panel in the frontend
  compare_physics_validators.py → per-question breakdown in the paper report
"""

import re
from typing import Optional


# -- Known hallucinated / non-standard symbols and their corrections ------------
SYMBOL_CORRECTIONS = {
    "C_s":      ("Cox",       "gate oxide capacitance per unit area [F/m²]"),
    "C_x":      ("Cox",       "gate oxide capacitance per unit area [F/m²]"),
    "C_gate":   ("Cox",       "gate oxide capacitance per unit area [F/m²]"),
    "Cgate":    ("Cox",       "gate oxide capacitance per unit area [F/m²]"),
    "beta_s":   ("Cox",       "likely meant gate oxide capacitance (Cox)"),
    "beta":     ("mu*Cox*(W/L)", "process transconductance — use explicit mu*Cox*(W/L)"),
    "kn":       ("mu*Cox",    "process transconductance parameter mu*Cox"),
    "kp":       ("mu*Cox",    "process transconductance parameter mu*Cox"),
    "mu_n":     ("mu",        "electron mobility [m²/Vs] — use 'mu'"),
    "mu_p":     ("mu",        "hole mobility [m²/Vs] — use 'mu'"),
    "V_T":      ("Vt",        "thermal voltage kT/q ≈ 26mV at 300K — use 'Vt'"),
    "VT":       ("Vt",        "thermal voltage — use 'Vt'"),
    "I_sat":    ("Idsat",     "saturation drain current — use 'Idsat'"),
    "I_D_sat":  ("Idsat",     "saturation drain current — use 'Idsat'"),
    "V_ov":     ("Vov",       "overdrive voltage Vgs − Vth — use 'Vov'"),
    "lam":      ("lam",       "channel length modulation parameter λ [1/V] — already correct"),
    "lambda":   ("lam",       "use 'lam' for channel length modulation parameter"),
}

# -- Physical dimension descriptions --------------------------------------------
DIM_DESCRIPTIONS = {
    "I":  "current [A]",
    "V":  "voltage [V]",
    "L":  "length [m]",
    "T":  "time [s]",
    "K":  "temperature [K]",
    "C":  "charge [C]",
    "J":  "energy [J]",
}


def _describe_dims(dim_dict: dict) -> str:
    """Turn {'I':1, 'V':-1} into human-readable 'current/voltage = [A/V]'."""
    if not dim_dict:
        return "dimensionless"
    parts = []
    for k, v in dim_dict.items():
        base = DIM_DESCRIPTIONS.get(k, k)
        if v == 1:
            parts.append(base)
        elif v == -1:
            parts.append(f"1/{base}")
        elif v > 0:
            parts.append(f"{base}^{v}")
        else:
            parts.append(f"1/{base}^{abs(v)}")
    return " · ".join(parts)


def _extract_dim_dict(msg: str) -> Optional[dict]:
    """Parse '{I: 1, V: -1}' from a validator message string."""
    match = re.search(r'\{([^}]+)\}', msg)
    if not match:
        return None
    try:
        raw = match.group(1)
        result = {}
        for part in raw.split(","):
            part = part.strip()
            if ":" in part:
                k, v = part.split(":", 1)
                result[k.strip().strip("'")] = float(v.strip())
        return result
    except Exception:
        return None


def _extract_unresolved_symbols(msg: str) -> list:
    """Pull symbol names out of 'Unresolved symbols remain: {gamma, Phi_f}'."""
    match = re.search(r'Unresolved symbols.*?:\s*\{([^}]+)\}', msg)
    if not match:
        return []
    raw = match.group(1)
    return [s.strip() for s in raw.split(",") if s.strip()]


# -----------------------------------------------------------------------------
# Core explainer
# -----------------------------------------------------------------------------

def explain_physics_score(
    sym_msg:  str,
    dim_msg:  str,
    num_msg:  str,
    equation: str = "",
    model_label: str = "",   # e.g. "70B" or "RAG 0.5B" — used in phrasing
) -> dict:
    """
    Takes the raw validator message strings and returns a structured explanation.

    Returns:
        {
          "symbolic":  { "verdict": "PASS"|"WARN"|"FAIL", "reason": str, "hint": str },
          "dimensional": { ... },
          "numerical":   { ... },
          "coverage":    { ... },
          "feedback_hint": str,   # consolidated actionable hint for the user
          "summary": str,         # one-sentence overall verdict
        }
    """
    result = {}

    # -- 1. Symbolic check ------------------------------------------------------
    if sym_msg is None:
        sym_msg = ""

    if "No equation" in sym_msg or "not detected" in sym_msg.lower():
        result["symbolic"] = {
            "verdict": "FAIL",
            "reason":  (
                f"The {'model' if not model_label else model_label} did not write a mathematical "
                "equation — it responded with prose only. Physics questions should contain an "
                "explicit formula. This is a strong indicator the model did not know the equation."
            ),
            "hint": "Write the correct equation in the correction box using standard notation, "
                    "e.g. 'Id = 0.5 * mu * Cox * (W/L) * (Vgs - Vth)^2'.",
        }
    elif "Parsing failed" in sym_msg or "SympifyError" in sym_msg or "error" in sym_msg.lower():
        # Try to identify the bad token
        bad_token_match = re.search(r"'([^']+)'", sym_msg)
        bad_token = bad_token_match.group(1) if bad_token_match else "unknown"
        correction = SYMBOL_CORRECTIONS.get(bad_token, None)
        hint_str = (
            f"The symbol '{bad_token}' is not standard MOSFET notation. "
            + (f"Replace it with '{correction[0]}' ({correction[1]})." if correction
               else "Use standard symbols from the MOSFET equation set.")
        )
        result["symbolic"] = {
            "verdict": "FAIL",
            "reason":  (
                f"An equation was found but SymPy could not parse it. "
                f"This usually means the model invented a non-standard symbol "
                f"(found near: '{bad_token}'). Standard MOSFET symbols: "
                "Id, mu, Cox, W, L, Vgs, Vth, Vds, gamma, Phi_f, Vsb, k, T, q, SS, Cd, gm, ro."
            ),
            "hint": hint_str,
        }
    elif "parsed successfully" in sym_msg.lower() or "[OK]" in sym_msg:
        eq_display = f" `{equation}`" if equation else ""
        result["symbolic"] = {
            "verdict": "PASS",
            "reason":  f"Equation{eq_display} was successfully parsed by SymPy. "
                       "All tokens map to recognized physics symbols.",
            "hint":    "",
        }
    else:
        result["symbolic"] = {
            "verdict": "WARN",
            "reason":  f"Symbolic check returned an unexpected message: {sym_msg}",
            "hint":    "Manually verify the equation structure.",
        }

    # -- 2. Dimensional check ---------------------------------------------------
    if dim_msg is None:
        dim_msg = ""

    if "skipped" in dim_msg.lower() or "skip" in dim_msg.lower():
        result["dimensional"] = {
            "verdict": "WARN",
            "reason":  "Dimensional analysis was skipped because no parseable equation was found.",
            "hint":    "Fix the equation first (see symbolic check feedback).",
        }
    elif "[OK]" in dim_msg or "consistent" in dim_msg.lower():
        dim_dict = _extract_dim_dict(dim_msg)
        dim_human = _describe_dims(dim_dict) if dim_dict else "matching units"
        result["dimensional"] = {
            "verdict": "PASS",
            "reason":  f"Both sides of the equation resolve to the same physical dimension: "
                       f"{dim_human}. The equation is unit-consistent.",
            "hint":    "",
        }
    elif "mismatch" in dim_msg.lower() or "[FAIL]" in dim_msg:
        # Extract LHS and RHS dimension dicts from the message
        lhs_match = re.search(r'LHS\s*(\{[^}]*\})', dim_msg)
        rhs_match = re.search(r'RHS\s*(\{[^}]*\})', dim_msg)
        lhs_str = lhs_match.group(1) if lhs_match else "unknown"
        rhs_str = rhs_match.group(1) if rhs_match else "unknown"

        lhs_d = _extract_dim_dict(lhs_str) if lhs_match else {}
        rhs_d = _extract_dim_dict(rhs_str) if rhs_match else {}
        lhs_h = _describe_dims(lhs_d) if lhs_d else lhs_str
        rhs_h = _describe_dims(rhs_d) if rhs_d else rhs_str

        empty_side = ""
        if rhs_d == {} and lhs_d:
            empty_side = (
                " The right-hand side is dimensionless — this almost always means one or more "
                "symbols on the RHS are not in the standard physics dictionary. "
                "The model likely used an invented or misspelled symbol."
            )

        result["dimensional"] = {
            "verdict": "FAIL",
            "reason":  (
                f"Dimension mismatch: left-hand side is '{lhs_h}' but right-hand side "
                f"is '{rhs_h}'.{empty_side} "
                "A physically correct equation must have matching units on both sides."
            ),
            "hint": (
                "The equation has a unit inconsistency. In your correction, make sure the "
                "right-hand side has the same units as the left. For drain current (A), "
                "the RHS must combine [m²/Vs] × [F/m²] × [m] × [m]^-1 × [V]² = [A]."
            ),
        }
    else:
        result["dimensional"] = {
            "verdict": "WARN",
            "reason":  f"Dimensional check: {dim_msg}",
            "hint":    "",
        }

    # -- 3. Numerical check -----------------------------------------------------
    if num_msg is None:
        num_msg = ""

    if "skipped" in num_msg.lower():
        result["numerical"] = {
            "verdict": "WARN",
            "reason":  "Numerical substitution was skipped (no valid equation to evaluate).",
            "hint":    "Fix the equation first.",
        }
    elif "Unresolved symbols" in num_msg:
        symbols = _extract_unresolved_symbols(num_msg)
        corrections = []
        for s in symbols:
            corr = SYMBOL_CORRECTIONS.get(s)
            if corr:
                corrections.append(f"'{s}' → use '{corr[0]}' ({corr[1]})")
            else:
                corrections.append(
                    f"'{s}' — not a standard symbol; define it or use the nearest standard equivalent"
                )
        hint_parts = "\n".join(f"  • {c}" for c in corrections) if corrections else "  • No known corrections — define these symbols explicitly."
        result["numerical"] = {
            "verdict": "FAIL",
            "reason":  (
                f"The equation contains {len(symbols)} symbol(s) with no known physical value: "
                f"{', '.join(symbols)}. Substituting standard 100nm MOSFET parameters could not "
                "produce a numeric result."
            ),
            "hint": (
                f"In your correction, replace the unknown symbol(s) with standard notation:\n"
                f"{hint_parts}"
            ),
        }
    elif "NaN" in num_msg or "infinite" in num_msg:
        result["numerical"] = {
            "verdict": "FAIL",
            "reason":  (
                "Substituting realistic values produced a NaN or infinite result. "
                "This indicates a division by zero or a structurally broken equation — "
                "e.g. dividing by (Vgs - Vth) when they are equal, or a sign error."
            ),
            "hint": "Check the denominator of the equation — it may be zero at the test point.",
        }
    elif "Unrealistic" in num_msg or "Extremely small" in num_msg or "Very large" in num_msg:
        value_match = re.search(r'([\d.e+\-]+)\s*[AV]', num_msg)
        value_str = value_match.group(0) if value_match else "an extreme value"
        result["numerical"] = {
            "verdict": "WARN",
            "reason":  (
                f"The equation evaluates to {value_str} with standard 100nm MOSFET parameters, "
                "which is outside the physically realistic range. This often indicates a unit "
                "mismatch (e.g. using nm instead of m for W or L, or µA/V² instead of A/V²)."
            ),
            "hint": (
                "In your correction, use SI units throughout: "
                "W, L in metres [m], Cox in [F/m²], mu in [m²/Vs]. "
                "A correct drain current should be in the range 1pA–1A."
            ),
        }
    elif "plausible" in num_msg.lower() or "[OK]" in num_msg:
        value_match = re.search(r'([\d.e+\-]+)', num_msg.split(":")[-1])
        value_str = value_match.group(1) if value_match else "a realistic value"
        result["numerical"] = {
            "verdict": "PASS",
            "reason":  (
                f"Substituting standard 100nm MOSFET parameters gives {value_str}, "
                "which is within the physically realistic range for this quantity."
            ),
            "hint": "",
        }
    else:
        result["numerical"] = {
            "verdict": "WARN",
            "reason":  f"Numerical check: {num_msg}",
            "hint":    "",
        }

    # -- 4. Coverage check (derived from numerical) -----------------------------
    if "Unresolved" in num_msg:
        symbols = _extract_unresolved_symbols(num_msg)
        result["coverage"] = {
            "verdict": "FAIL",
            "reason":  (
                f"The equation uses {len(symbols)} symbol(s) not in the standard physics "
                f"dictionary: {', '.join(symbols)}. "
                "These may be hallucinated names, LaTeX artifacts, or uncommon notation."
            ),
            "hint": "Use standard MOSFET symbol names. See the symbolic check hint above.",
        }
    elif result["symbolic"]["verdict"] == "PASS" and "failed" not in num_msg.lower():
        result["coverage"] = {
            "verdict": "PASS",
            "reason":  "All symbols in the equation are recognized standard physics symbols.",
            "hint":    "",
        }
    elif result["symbolic"]["verdict"] == "FAIL":
        result["coverage"] = {
            "verdict": "FAIL",
            "reason":  "Coverage could not be checked — equation was not parseable.",
            "hint":    "Fix the equation syntax first.",
        }
    else:
        result["coverage"] = {
            "verdict": "WARN",
            "reason":  "Symbol coverage partially verified.",
            "hint":    "",
        }

    # -- 5. Consolidated feedback hint ------------------------------------------
    failures = [
        (k, v) for k, v in result.items()
        if isinstance(v, dict) and v.get("verdict") == "FAIL" and v.get("hint")
    ]
    warnings = [
        (k, v) for k, v in result.items()
        if isinstance(v, dict) and v.get("verdict") == "WARN" and v.get("hint")
    ]

    if not failures and not warnings:
        feedback_hint = (
            "This answer passes all physics checks. If the explanation or context is wrong, "
            "use 'Mark as correct' or write a more complete answer in the correction box."
        )
        summary = "All physics checks passed — equation is structurally and numerically correct."
    elif failures:
        parts = [v["hint"] for _, v in failures if v["hint"]]
        feedback_hint = "Correction needed:\n" + "\n".join(f"  [{k.upper()}] {p}" for (k, _), p in zip(failures, parts))
        n_fail = len(failures)
        summary = (
            f"{n_fail} physics check(s) failed. "
            + ("The model likely hallucinated equation structure." if n_fail >= 2
               else failures[0][1]["reason"][:80] + "…")
        )
    else:
        parts = [v["hint"] for _, v in warnings if v["hint"]]
        feedback_hint = "Minor issues:\n" + "\n".join(f"  [{k.upper()}] {p}" for (k, _), p in zip(warnings, parts))
        summary = "Equation parsed but has minor physics warnings — verify manually."

    result["feedback_hint"] = feedback_hint
    result["summary"] = summary

    return result


# -----------------------------------------------------------------------------
# Compact version for API response (smaller payload)
# -----------------------------------------------------------------------------

def explain_compact(
    sym_msg: str,
    dim_msg: str,
    num_msg: str,
    equation: str = "",
    model_label: str = "",
) -> dict:
    """
    Lighter version for the frontend Inspector panel.
    Returns one reason + one hint per component, plus a single feedback_hint.
    """
    full = explain_physics_score(sym_msg, dim_msg, num_msg, equation, model_label)
    return {
        "symbolic":    {"verdict": full["symbolic"]["verdict"],
                        "reason":  full["symbolic"]["reason"]},
        "dimensional": {"verdict": full["dimensional"]["verdict"],
                        "reason":  full["dimensional"]["reason"]},
        "numerical":   {"verdict": full["numerical"]["verdict"],
                        "reason":  full["numerical"]["reason"]},
        "coverage":    {"verdict": full["coverage"]["verdict"],
                        "reason":  full["coverage"]["reason"]},
        "feedback_hint": full["feedback_hint"],
        "summary":       full["summary"],
    }
