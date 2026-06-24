import re
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)


class EquationValidator:
    """
    Extracts and symbolically parses physics equations from model output.

    Fixes applied:
    - Expanded symbol dictionary from 7 → 21 symbols (body effect, subthreshold swing, etc.)
    - Added comprehensive LaTeX normalization (\\text{}, \\frac{}{}, \\sqrt{}, etc.)
    - extract_equation() now skips evidence/prompt header lines to avoid validating
      the evidence block instead of the model's own answer.
    - split("=", 1) prevents false splits on ">=" inside equations.
    """

    # Lines whose content indicates they belong to the prompt/evidence block
    _SKIP_PATTERNS = frozenset([
        "evidence:", "question:", "use only", "answer in",
        "system", "assistant", "user\n",
    ])

    # English connective words that never appear inside a real equation. Their
    # presence means the "equation" is actually a prose description such as
    # "Cox = oxide capacitance per unit area" — which must NOT be scored as an
    # equation (it would parse into garbage symbols and poison the score).
    _CONNECTIVE_WORDS = frozenset([
        "per", "the", "of", "in", "is", "are", "to", "for", "with", "where",
        "here", "which", "that", "this", "means", "meaning", "value", "values",
        "unit", "units", "denotes", "denote", "represents", "represent",
        "called", "known", "such", "and", "or", "from", "as", "by", "than",
        "into", "when", "because", "gives", "give", "given", "equals", "equal",
        "where", "about", "approximately", "roughly",
    ])

    def __init__(self):
        self.symbols = {
            # -- Core MOSFET drain current ----------------------------------
            "Id":    sp.Symbol("Id"),
            "Ids":   sp.Symbol("Ids"),
            "mu":    sp.Symbol("mu"),
            "Cox":   sp.Symbol("Cox"),
            "W":     sp.Symbol("W"),
            "L":     sp.Symbol("L"),
            "Vgs":   sp.Symbol("Vgs"),
            "Vth":   sp.Symbol("Vth"),
            "Vds":   sp.Symbol("Vds"),
            # -- Body effect ------------------------------------------------
            "Vth0":  sp.Symbol("Vth0"),
            "gamma": sp.Symbol("gamma"),
            "Phi_f": sp.Symbol("Phi_f"),
            "Vsb":   sp.Symbol("Vsb"),
            # -- Subthreshold swing -----------------------------------------
            "k":     sp.Symbol("k"),
            "T":     sp.Symbol("T"),
            "q":     sp.Symbol("q"),
            "SS":    sp.Symbol("SS"),
            "Cd":    sp.Symbol("Cd"),
            # -- Small-signal / general -------------------------------------
            "gm":    sp.Symbol("gm"),
            "ro":    sp.Symbol("ro"),
            "Vt":    sp.Symbol("Vt"),    # thermal voltage kT/q
            "n":     sp.Symbol("n"),     # ideality factor
            # -- Channel length modulation / saturation ---------------------
            "lam":   sp.Symbol("lam"),   # lambda (CLM parameter) [1/V]
            "Vov":   sp.Symbol("Vov"),   # overdrive voltage Vgs-Vth [V]
            "Vdsat": sp.Symbol("Vdsat"), # saturation drain voltage [V]
            "Idsat": sp.Symbol("Idsat"), # saturation drain current [A]
            # -- Second-order effects ---------------------------------------
            "DIBL":  sp.Symbol("DIBL"),  # drain-induced barrier lowering [dimensionless]
            "Av":    sp.Symbol("Av"),    # voltage gain [dimensionless]
            "tox":   sp.Symbol("tox"),   # oxide thickness [m]
            "Iref":  sp.Symbol("Iref"),  # reference current [A]
            "Vbs":   sp.Symbol("Vbs"),   # bulk-source voltage [V]
            # -- pn junction / diode (common in 70B answers) ---------------
            "J":     sp.Symbol("J"),     # current density [A/m2]
            "Jo":    sp.Symbol("Jo"),    # saturation current density
            "J0":    sp.Symbol("J0"),    # saturation current density
            "V":     sp.Symbol("V"),     # generic voltage [V]
            "phi":   sp.Symbol("phi"),   # surface potential [V]
            "Vbi":   sp.Symbol("Vbi"),   # built-in voltage [V]
            "NA":    sp.Symbol("NA"),    # acceptor doping [1/m3]
            "ND":    sp.Symbol("ND"),    # donor doping [1/m3]
            "eps":   sp.Symbol("eps"),   # permittivity [F/m]
            "xd":    sp.Symbol("xd"),    # depletion width [m]
            # -- Overrides for SymPy predefined names ------------------------
            "S":     sp.Symbol("S"),     # subthreshold swing
            "E":     sp.Symbol("E"),     # electric field
            "I":     sp.Symbol("I"),     # current
            "C":     sp.Symbol("C"),     # capacitance / constant
            "N":     sp.Symbol("N"),     # carrier concentration
            "O":     sp.Symbol("O"),     # oxide/other
            "Q":     sp.Symbol("Q"),     # charge
            "S0":    sp.Symbol("S0"),    # subthreshold swing baseline
            "E_max": sp.Symbol("E_max"), # max electric field
        }

        self.transformations = (
            standard_transformations
            + (implicit_multiplication_application,)
        )

    # --------------------------------------------------------------------------
    # Normalization
    # --------------------------------------------------------------------------

    def normalize_equation(self, eq: str) -> str:
        """
        Convert LaTeX notation and symbol variants to SymPy-parseable ASCII.
        """
        eq = eq.replace('`', '')
        # -- LaTeX structural removal ---------------------------------------
        eq = re.sub(r'\\text\{([^}]+)\}',             r'\1',       eq)   # \text{Id} → Id
        eq = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}',  r'(\1)/(\2)', eq)  # \frac{W}{L} → (W)/(L)
        eq = re.sub(r'\\sqrt\{([^}]+)\}',             r'sqrt(\1)', eq)   # \sqrt{x} → sqrt(x)
        eq = eq.replace(r'\times', '*').replace(r'\cdot', '*')
        eq = eq.replace(r'\ln', 'ln')                                    # \ln → ln
        
        # Unwrap subscripts (e.g. V_{th} -> V_th) so alias regexes can match them
        eq = re.sub(r'_\{([^}]+)\}', r'_\1', eq)

        # Strip \left, \right, \middle and surrounding brackets
        eq = re.sub(r'\\(?:left|right|middle)[|()\[\].]?', '', eq)
        # Drop any other backslash commands before braces, then convert braces
        eq = re.sub(r'\\\w+\{', '(', eq)
        eq = eq.replace('{', '(').replace('}', ')')
        # Remove display math markers
        eq = eq.replace(r'\[', '').replace(r'\]', '').replace('$', '')

        # -- Greek / symbol aliases -----------------------------------------
        eq = eq.replace("μ_n", "mu").replace("mu_n", "mu").replace("μ", "mu")
        eq = eq.replace("γ", "gamma")
        eq = re.sub(r'[ΦΦ]_?f', "Phi_f", eq)   # Φf, Φ_f
        eq = eq.replace("Phi_f", "Phi_f")       # ensure canonical
        eq = re.sub(r'V_?[Ss][Bb]', "Vsb", eq)
        eq = re.sub(r'V_?[Gg][Ss]', "Vgs", eq)
        eq = re.sub(r'V_?[Tt][Hh]0', "Vth0", eq)
        eq = re.sub(r'V_?[Tt][Hh]', "Vth", eq)
        eq = re.sub(r'V_?[Dd][Ss]', "Vds", eq)
        eq = re.sub(r'I_?[Dd]',     "Id",  eq)
        eq = re.sub(r'C_?[Oo][Xx]', "Cox", eq)
        # Common hallucinations — model renames Cox to C_s, C_x. (Don't clobber C_d)
        eq = re.sub(r'C_[sSxX]', "Cox", eq)
        # Channel length modulation: \lambda, λ, lambda → lam
        eq = re.sub(r'\\lambda|λ|lambda(?!_)', "lam", eq)
        # Overdrive voltage
        eq = re.sub(r'V_?[Oo][Vv]', "Vov", eq)
        # Saturation drain voltage
        eq = re.sub(r'V_?[Dd][Ss][Aa][Tt]', "Vdsat", eq)
        # Bulk-source voltage
        eq = re.sub(r'V_?[Bb][Ss]', "Vbs", eq)
        # Saturation current
        eq = re.sub(r'I_?[Dd][Ss][Aa][Tt]|I_?[Dd][Ss][Aa][Tt]', "Idsat", eq)

        # -- 70B notation patterns -----------------------------------------
        # Square brackets used as multiplication: Jo[expr] → Jo*(expr)
        eq = re.sub(r'(\w+)\[([^\]]+)\]', r'\1*(\2)', eq)
        # e^x / e^{x} → exp(x)  (common in 70B diode/BJT equations)
        eq = re.sub(r'\be\*\*\{([^}]+)\}', r'exp(\1)', eq)
        eq = re.sub(r'\be\*\*\(([^)]+)\)', r'exp(\1)', eq)
        eq = re.sub(r'\be\*\*(\S+)',       r'exp(\1)', eq)
        # phi / φ → phi symbol
        eq = eq.replace("φ", "phi")
        # J_0, J0 normalisation (diode saturation current)
        eq = re.sub(r'J_?0\b', "Jo", eq)
        # V_bi, V_built-in → Vbi
        eq = re.sub(r'V_?bi\b|V_?[Bb]uilt', "Vbi", eq)
        # N_A, N_D → NA, ND
        eq = re.sub(r'N_?[Aa]\b', "NA", eq)
        eq = re.sub(r'N_?[Dd]\b', "ND", eq)
        # epsilon, ε → eps
        eq = re.sub(r'epsilon|ε', "eps", eq)

        # -- Operator normalization -----------------------------------------
        eq = eq.replace("^", "**")
        eq = eq.replace("muCox", "mu*Cox")
        eq = eq.replace("Lg", "L").replace("Leff", "L")

        # -- Final cleanup --------------------------------------------------
        eq = eq.replace('\\', '')  # Remove any stray backslashes that cause SyntaxError
        return eq.strip()

    # --------------------------------------------------------------------------
    # Equation extraction
    # --------------------------------------------------------------------------

    def _looks_like_prose(self, lhs: str, rhs: str) -> bool:
        """
        True if a candidate "lhs = rhs" is really a natural-language description
        rather than an equation. Two signals:
          1. Any English connective word (per, in, of, is, …) appears — these
             never occur inside a math expression.
          2. The RHS contains two consecutive >=4-letter alphabetic tokens
             (e.g. "oxide capacitance", "drain current") — descriptive prose.
        """
        words = re.findall(r"[A-Za-z]+", f"{lhs} {rhs}")
        if any(w.lower() in self._CONNECTIVE_WORDS for w in words):
            return True
        rhs_tokens = rhs.split()
        long_alpha = [bool(re.fullmatch(r"[A-Za-z]{4,}", t)) for t in rhs_tokens]
        return any(long_alpha[i] and long_alpha[i + 1]
                   for i in range(len(long_alpha) - 1))

    def _clean_candidate(self, raw_line: str) -> str | None:
        """
        Turn one raw line into a cleaned "lhs = rhs" candidate, or None if the
        line is not a usable equation. Strips evidence/prompt lines, LaTeX
        wrappers, "Label:" prefixes, trailing natural-language clauses, and
        rejects prose descriptions.
        """
        line = raw_line.strip()
        if not line or "=" not in line:
            return None

        # Skip evidence/prompt header lines
        if any(pat in line.lower() for pat in self._SKIP_PATTERNS):
            return None

        # Strip LaTeX display math wrappers
        line = line.lstrip(r"\[$").rstrip(r"\]$").strip()

        # Handle "Label: equation" format — take the part after the last ":"
        if ":" in line:
            for part in reversed(line.split(":")):
                if "=" in part:
                    line = part.strip()
                    break

        # Must still contain a "=" (reject "==" comparisons by taking first split)
        parts = line.split("=")
        if len(parts) < 2:
            return None

        lhs_tokens = parts[0].strip().split()
        lhs = lhs_tokens[-1] if lhs_tokens else parts[0].strip()
        rhs = parts[1].strip()
        if not (lhs and rhs):
            return None

        # Strip trailing prose explanation (", where J is the current density")
        rhs = re.sub(
            r',\s+(where|with|and|here|in which|J\s+is|I\s+is|V\s+is)\b.*$',
            '', rhs, flags=re.IGNORECASE
        ).strip()
        # Cut at a period followed by a capital letter (new sentence)
        rhs = re.sub(r'\.\s+[A-Z].*$', '', rhs).strip()
        # Cut a trailing natural-language clause ("… in saturation", "… where …")
        rhs = re.sub(
            r'\s+(in|for|when|at|where|with|which|that|is|are|denotes|'
            r'represents|called|given)\b.*$',
            '', rhs, flags=re.IGNORECASE
        ).strip()
        if not rhs:
            return None

        # Reject prose bullets like "Cox = oxide capacitance per unit area"
        if self._looks_like_prose(lhs, rhs):
            return None

        return f"{lhs} = {rhs}"

    def _candidate_equations(self, text: str) -> list[str]:
        """
        All cleaned equation candidates in priority order. Lines starting with
        "Equation:" (forced by the prompt template / explore path) come first,
        then every other "=" line. De-duplicated, order preserved.
        """
        priority, others = [], []
        for raw_line in text.split("\n"):
            line = raw_line.strip()
            if line.lower().startswith("equation:") and "=" in line:
                cleaned = self._clean_candidate(line[len("equation:"):].strip())
                if cleaned:
                    priority.append(cleaned)
                continue
            cleaned = self._clean_candidate(raw_line)
            if cleaned:
                others.append(cleaned)

        seen, ordered = set(), []
        for cand in priority + others:
            if cand not in seen:
                seen.add(cand)
                ordered.append(cand)
        return ordered

    def extract_equation(self, text: str) -> str | None:
        """
        Return the best equation-like line from the model's generated answer
        (first priority candidate). Kept for back-compat with api_server's
        display path; validate() now scans *all* candidates.
        """
        candidates = self._candidate_equations(text)
        return candidates[0] if candidates else None

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------

    def validate(self, text: str):
        """
        Returns (best_lhs, best_rhs, message).
        """
        candidates = self._candidate_equations(text)
        if not candidates:
            return None, None, "[WARN] No equation detected."

        best_rank = None
        best_lhs = best_rhs = None
        last_error = None

        def _get_free_symbols(expr) -> set:
            if expr is None:
                return set()
            if isinstance(expr, (list, tuple, set)):
                syms = set()
                for item in expr:
                    syms |= _get_free_symbols(item)
                return syms
            if hasattr(expr, "free_symbols"):
                return expr.free_symbols
            return set()

        for candidate in candidates:
            normalized = self.normalize_equation(candidate)
            if "=" not in normalized:
                continue
            try:
                lhs_str, rhs_str = normalized.split("=", 1)
                lhs_expr = parse_expr(
                    lhs_str.strip(),
                    local_dict=self.symbols,
                    transformations=self.transformations,
                )
                rhs_expr = parse_expr(
                    rhs_str.strip(),
                    local_dict=self.symbols,
                    transformations=self.transformations,
                )
                # Rank: prefer more recognized symbols, then fewer unknown ones.
                free = _get_free_symbols(lhs_expr) | _get_free_symbols(rhs_expr)
                known = sum(1 for s in free if s.name in self.symbols)
                rank = (known, -(len(free) - known))
                if best_rank is None or rank > best_rank:
                    best_rank, best_lhs, best_rhs = rank, lhs_expr, rhs_expr
            except Exception as e:
                last_error = e
                continue

        if best_rank is None:
            if last_error is not None:
                return None, None, f"[WARN] Parsing failed: {type(last_error).__name__}: {last_error}"
            return None, None, "[WARN] No equation detected after normalization."

        return best_lhs, best_rhs, "[OK] Equation parsed successfully"