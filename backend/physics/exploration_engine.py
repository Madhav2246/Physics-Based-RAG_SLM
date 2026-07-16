"""
ExplorationEngine — physics-grounded design synthesis.

No SLM call. No hallucination.
Every answer is: corpus equation + SymPy algebra + stated value provenance.

The three steps:
  1. detect_target(query)         — what variable is the user solving for?
  2. extract_user_values(query)   — what numeric values did the user supply?
  3. solve_for(lhs, rhs, target,
               tracker)          — SymPy solve + numeric substitution

All failure modes return a result dict with success=False and a human-readable
error — no IndexError on solutions[0], no silent wrong answers.
"""
from __future__ import annotations
import re
import math
import sympy as sp
from sympy import solve, Eq

from physics.value_tracker import ValueTracker


# EXPLORE trigger phrases — design/synthesis intent
_EXPLORE_TRIGGERS = [
    "how do i", "how should i", "how to design", "how to choose",
    "what w/l", "what tox", "what doping", "what cox",
    "choose the", "design the", "optimize", "find the value",
    "calculate the", "what aspect ratio", "what oxide thickness",
    "what channel length", "what width", "what vth", "what gamma",
    "what id", "what overdrive", "what threshold"
]


def detect_mode(query: str) -> str:
    """
    Module-level mode classifier.
    Returns 'SWEEP'   if the query asks for a parametric sweep/plot over a range,
            'EXPLORE' if it has single-point design/synthesis intent,
            'LOOKUP'  otherwise (default safe path).

    SWEEP is checked first (more specific) and only fires when a range is
    actually parseable — a bare "plot" with no range falls through to EXPLORE.
    """
    from physics.sweep_engine import parse_sweep_request
    if parse_sweep_request(query) is not None:
        return "SWEEP"

    q = query.lower()
    for trigger in _EXPLORE_TRIGGERS:
        if trigger in q:
            return "EXPLORE"
    return "LOOKUP"


class ExplorationEngine:
    """
    Rearranges corpus equations to answer design questions deterministically.
    """

    # ── Target variable detection ───────────────────────────────────────────
    TARGET_PATTERNS: list[tuple[str, str]] = [
        (r'\bW\s*/\s*L\b|\baspect\s+ratio\b|\bWL\b',           'WL'),
        (r'\btox\b|\boxid[e]?\s+thickness\b',                   'tox'),
        (r'\bCox\b|\boxid[e]?\s+cap',                           'Cox'),
        (r'\bNA\b|\bacceptor\s+dop|\bdoping\s+conc',            'NA'),
        (r'\bgamma\b|\bbody.effect\s+param',                    'gamma'),
        (r'\bgm\b|\btransconductance\b',                        'gm'),
        (r'\bVth\b|\bthreshold\s+voltage\b',                    'Vth'),
        (r'\bVth0\b',                                            'Vth0'),
        (r'\bId\b|\bdrain\s+current\b',                         'Id'),
        (r'\bro\b|\boutput\s+resistance\b',                     'ro'),
        (r'\bPhi_f\b|\bfermi\s+potential\b',                    'Phi_f'),
        (r'\bVov\b|\boverdrive\s+voltage\b',                    'Vov'),
    ]

    # ── Numeric value extraction from query ─────────────────────────────────
    VALUE_PATTERNS: list[tuple[str, str, dict]] = [
        # -- Structured format: Sym = Value Unit  (the original patterns) ----
        (r'Id\s*[=:]\s*(\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)\s*(mA|uA|nA|A)',   'Id',  {'mA':1e-3,'uA':1e-6,'nA':1e-9,'A':1.0}),
        (r'Vov\s*[=:]\s*(\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)\s*(V|mV)',         'Vov', {'V':1.0,'mV':1e-3}),
        (r'Vgs\s*[=:]\s*(\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)\s*(V|mV)',         'Vgs', {'V':1.0,'mV':1e-3}),
        (r'Vth\s*[=:]\s*(\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)\s*(V|mV)',         'Vth', {'V':1.0,'mV':1e-3}),
        (r'Vsb\s*[=:]\s*(\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)\s*(V|mV)',         'Vsb', {'V':1.0,'mV':1e-3}),
        (r'tox\s*[=:]\s*(\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)\s*(nm|um|m)',      'tox', {'nm':1e-9,'um':1e-6,'m':1.0}),
        (r'L\s*[=:]\s*(\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)\s*(nm|um|m)',        'L',   {'nm':1e-9,'um':1e-6,'m':1.0}),
        (r'W\s*[=:]\s*(\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)\s*(nm|um|m)',        'W',   {'nm':1e-9,'um':1e-6,'m':1.0}),
        (r'mu\s*[=:]\s*(\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)',                    'mu',    {}),
        (r'Cox\s*[=:]\s*(\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)',                   'Cox',   {}),
        (r'gamma\s*[=:]\s*(\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)',                 'gamma', {}),
        (r'Phi_f\s*[=:]\s*(\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)\s*(V|mV)?',      'Phi_f', {'V':1.0,'mV':1e-3,'':1.0}),
        (r'Vth0\s*[=:]\s*(\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)\s*(V|mV)',         'Vth0',  {'V':1.0,'mV':1e-3}),
        (r'T\s*[=:]\s*(\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)\s*(K|C)?',           'T',     {'K':1.0,'C':1.0,'':1.0}),

        # -- Natural language: "X milliamp(s)", "X microamp(s)", etc. -------
        (r'(\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)\s*milliamp',                     'Id',  {'':1e-3}),
        (r'(\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)\s*microamp',                     'Id',  {'':1e-6}),
        (r'(\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)\s*nanoamp',                      'Id',  {'':1e-9}),
        # "drain current is X mA/A" or "current of X microamps"
        (r'(?:drain\s+current|current)\s+(?:of\s+|is\s+|roughly\s+)?(\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)\s*(mA|uA|nA|A)',
                                                           'Id',  {'mA':1e-3,'uA':1e-6,'nA':1e-9,'A':1.0}),
        # "Vgs of X volts", "threshold at X millivolts"
        (r'Vgs\s+of\s+(\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)\s*(volts?|V|mV)',    'Vgs', {'volts':1.0,'volt':1.0,'V':1.0,'mV':1e-3,'':1.0}),
        (r'threshold\s+at\s+(\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)\s*(millivolts?|volts?|mV|V)',
                                                           'Vth', {'millivolts':1e-3,'millivolt':1e-3,'mV':1e-3,
                                                                   'volts':1.0,'volt':1.0,'V':1.0,'':1.0}),
        # "overdrive voltage is half a volt" — no numeric capture group; value is preset
        (r'overdrive\s+voltage\s+is\s+half\b',             'Vov', {'':0.5}),
        # "X micron(s)", "X nanometer(s)" for W and L
        (r'W\s+is\s+(\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)\s*(microns?|micrometers?|um|nm|nanometers?)',
                                                           'W',   {'microns':1e-6,'micron':1e-6,'micrometers':1e-6,
                                                                   'micrometer':1e-6,'um':1e-6,'nm':1e-9,
                                                                   'nanometers':1e-9,'nanometer':1e-9,'':1.0}),
        (r'L\s+is\s+(\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)\s*(microns?|micrometers?|um|nm|nanometers?)',
                                                           'L',   {'microns':1e-6,'micron':1e-6,'micrometers':1e-6,
                                                                   'micrometer':1e-6,'um':1e-6,'nm':1e-9,
                                                                   'nanometers':1e-9,'nanometer':1e-9,'':1.0}),
        (r'gate\s+width\s+of\s+(\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)\s*(microns?|um|nm|nanometers?)',
                                                           'W',   {'microns':1e-6,'micron':1e-6,'um':1e-6,
                                                                   'nm':1e-9,'nanometers':1e-9,'nanometer':1e-9,'':1.0}),
        (r'gate\s+length\s+of\s+(\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)\s*(microns?|um|nm|nanometers?)',
                                                           'L',   {'microns':1e-6,'micron':1e-6,'um':1e-6,
                                                                   'nm':1e-9,'nanometers':1e-9,'nanometer':1e-9,'':1.0}),
        # "tox equals X nm/um"
        (r'tox\s+equals\s+(\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)\s*(nm|um|m)',    'tox', {'nm':1e-9,'um':1e-6,'m':1.0,'':1.0}),
        # "gamma equals X", "Phi_f is X"
        (r'gamma\s+equals\s+(\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)',               'gamma', {}),
        (r'Phi_f\s+is\s+(\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)',                   'Phi_f', {}),
        # "Vsb is X volts"
        (r'Vsb\s+is\s+(\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)\s*(volts?|V|mV)?',  'Vsb', {'volts':1.0,'volt':1.0,'V':1.0,'mV':1e-3,'':1.0}),
    ]

    # ── Physical sanity ranges for derived values ────────────────────────────
    SANITY_RANGES: dict[str, tuple[float, float]] = {
        'WL':    (0.5,    10_000),
        'tox':   (0.3e-9, 200e-9),
        'Cox':   (1e-4,   2.0),
        'Id':    (1e-15,  1.0),
        'gm':    (1e-9,   1.0),
        'Vth':   (0.0,    3.0),
        'Vth0':  (0.0,    3.0),
        'NA':    (1e12,   1e22),
        'Phi_f': (0.1,    0.8),
        'gamma': (0.01,   2.0),
        'Vov':   (0.05,   2.0),
    }

    def __init__(self, validator_or_symbols) -> None:
        """
        Accepts either a full EquationValidator or just its symbols dict.
        Pipeline passes self.validator; older code may pass self.validator.symbols.
        """
        if hasattr(validator_or_symbols, 'symbols'):
            # Full EquationValidator passed
            self.validator = validator_or_symbols
        else:
            # Just the symbols dict passed — wrap in a minimal namespace
            class _SymbolsOnly:
                def __init__(self, syms):
                    self.symbols = syms
            self.validator = _SymbolsOnly(validator_or_symbols)

        if 'WL' not in self.validator.symbols:
            self.validator.symbols['WL'] = sp.Symbol('WL')

        # Lazy sweep engine (only used in SWEEP mode)
        from physics.sweep_engine import SweepEngine
        self._sweep_engine = SweepEngine()

        # SLM extractor — set via set_slm_model() after pipeline init.
        # None until wired in; falls back to regex-only extraction.
        self._slm_extractor = None

    # ── Public API ───────────────────────────────────────────────────────────

    def detect_target(self, query: str) -> str | None:
        """
        Return the design target variable from the query.

        Key rule: if a symbol appears with an explicit value (e.g. 'gamma = 0.4'),
        it is a KNOWN INPUT — not the target we're solving for. Exclude it.
        Without this, 'What Vth with gamma = 0.4?' wrongly identifies gamma as target.
        """
        # Collect symbols the user already provided values for
        user_provided: set[str] = set()
        for pat, sym, _ in self.VALUE_PATTERNS:
            if re.search(pat, query, re.IGNORECASE):
                user_provided.add(sym)

        for pattern, symbol in self.TARGET_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                if symbol in user_provided:
                    continue   # given value → not the unknown target
                return symbol
        return None

    def set_slm_model(self, model) -> None:
        """Wire the already-loaded SLM into the extractor (called by the pipeline)."""
        from physics.slm_extractor import SLMExtractor
        self._slm_extractor = SLMExtractor(model)

    def extract_user_values(self, query: str, node_defaults=None) -> ValueTracker:
        # node_defaults: optional dict of TrackedValue from a process-node profile.
        # When present, the tracker resolves unknowns from node-specific constants
        # instead of the generic 100nm baseline. None => unchanged 100nm behavior.
        if self._slm_extractor and self._slm_extractor.available:
            return self.extract_user_values_twostage(query, node_defaults)
        return self._regex_extract(query, node_defaults)

    def extract_user_values_twostage(self, query: str, node_defaults=None) -> ValueTracker:
        """
        Two-stage extractor for the benchmark and the pipeline.

        Stage 1 (SLM): extracts natural-language values, verifies each one
                       against query tokens before accepting.
        Stage 2 (Regex): fills in anything Stage 1 missed.

        NOTE on deduplication: a symbol is only added to slm_found (blocking
        Stage 2) if Stage 1 returned a *finite* float for it. This prevents
        a garbled SLM output from silently suppressing the regex fallback.
        """
        import math as _math
        import logging as _logging
        _log = _logging.getLogger(__name__)

        tracker = (ValueTracker.with_node_profile(node_defaults)
                   if node_defaults else ValueTracker())

        # Stage 1 — SLM
        slm_found: set[str] = set()
        if self._slm_extractor and self._slm_extractor.available:
            try:
                slm_vals = self._slm_extractor.extract(query)
                _log.debug('[TwoStage] SLM returned: %s', slm_vals)
                for sym, val in slm_vals.items():
                    # Only block Stage 2 if Stage 1 produced a usable finite value.
                    # A non-finite result (nan/inf) means the SLM garbled the number;
                    # in that case we want Stage 2 regex to fill it in instead.
                    if not _math.isfinite(val):
                        _log.debug('[TwoStage] SLM gave non-finite %s=%s — not blocking regex', sym, val)
                        continue
                    tracker.add_user(sym, val, 'SI', 'extracted (SLM, verified)')
                    slm_found.add(sym)
                    _log.debug('[TwoStage] SLM accepted: %s = %g', sym, val)
            except Exception as exc:
                _log.warning('[ExplorationEngine] SLM extraction stage failed: %s', exc)

        # Stage 2 — Regex fallback for anything Stage 1 missed (or to fix SLM units)
        for pattern, sym, unit_map in self.VALUE_PATTERNS:
            # We intentionally let regex overwrite Stage 1 if a match is found,
            # because regex correctly handles SI unit conversion (e.g. '500 microamps' -> 5e-4)
            # which 0.5B SLMs often fail at (extracting just '500').
            m = re.search(pattern, query, re.IGNORECASE)
            if m:
                try:
                    val_str = m.group(1)
                    unit    = m.group(2) if m.lastindex and m.lastindex >= 2 else ''
                    scale   = unit_map.get(unit, 1.0) if unit_map else 1.0
                    value   = float(val_str) * scale
                except IndexError:
                    # Pattern has no capture group (e.g. 'overdrive voltage is half')
                    value = unit_map.get('', None)
                    if value is None:
                        continue
                tracker.add_user(sym, value, 'SI', 'extracted (regex)')

        return tracker

    def _regex_extract(self, query: str, node_defaults=None) -> ValueTracker:
        """Pure-regex extraction (pre-Feature-3 behaviour, unchanged)."""
        tracker = (ValueTracker.with_node_profile(node_defaults)
                   if node_defaults else ValueTracker())
        for pattern, sym, unit_map in self.VALUE_PATTERNS:
            m = re.search(pattern, query, re.IGNORECASE)
            if m:
                try:
                    val_str = m.group(1)
                    unit    = m.group(2) if m.lastindex and m.lastindex >= 2 else ''
                    scale   = unit_map.get(unit, 1.0) if unit_map else 1.0
                    value   = float(val_str) * scale
                except IndexError:
                    # Pattern has no capture group (e.g. 'half a volt')
                    value = unit_map.get('', None)
                    if value is None:
                        continue
                tracker.add_user(sym, value, 'SI', 'extracted from query')
        return tracker

    def solve_for(
        self,
        lhs_expr,
        rhs_expr,
        target_symbol_name: str,
        tracker: ValueTracker,
        corpus_equation_str: str = "",
    ) -> dict:
        """
        Core solve step. Never raises. Always returns a dict.
        Check result['success'] before using result['numeric'].
        """
        result: dict = {
            'success':         False,
            'target':          target_symbol_name,
            'symbolic':        None,
            'numeric':         None,
            'tracker':         tracker,
            'sanity_ok':       False,
            'error':           '',
            'corpus_equation': corpus_equation_str,
        }

        # Guard: target must be known
        target_sym = self.validator.symbols.get(target_symbol_name)
        if target_sym is None:
            result['error'] = (
                f"Symbol '{target_symbol_name}' not in known physics dictionary."
            )
            return result

        # Special case: WL (aspect ratio) — corpus stores W and L separately.
        # Substitute W → WL*L before solving so the ratio appears as one symbol.
        # SymPy then simplifies 0.5*mu*Cox*(WL*L/L)*(Vgs-Vth)^2, L cancels,
        # and we get WL = 2*Id / (mu*Cox*(Vgs-Vth)^2) cleanly.
        lhs_to_solve = lhs_expr
        rhs_to_solve = rhs_expr
        if target_symbol_name == 'WL':
            W_sym = self.validator.symbols.get('W')
            L_sym = self.validator.symbols.get('L')
            all_eq_syms = lhs_expr.free_symbols | rhs_expr.free_symbols
            if target_sym not in all_eq_syms and W_sym and L_sym:
                lhs_to_solve = lhs_expr.subs(W_sym, target_sym * L_sym)
                rhs_to_solve = rhs_expr.subs(W_sym, target_sym * L_sym)

        # Symbolic solve
        try:
            solutions = solve(Eq(lhs_to_solve, rhs_to_solve), target_sym)
        except Exception as exc:
            result['error'] = f"SymPy solve() raised: {type(exc).__name__}: {exc}"
            return result

        # Guard: empty solution — target absent from equation
        if not solutions:
            result['error'] = (
                f"Cannot derive '{target_symbol_name}' — it does not appear "
                f"in the retrieved corpus equation: {corpus_equation_str}"
            )
            return result

        symbolic_solution  = self._pick_physical_branch(
            solutions, target_symbol_name, tracker
        )
        result['symbolic'] = f"{target_symbol_name} = {symbolic_solution}"
        result['symbolic_expr'] = symbolic_solution   # raw SymPy obj for sweep reuse
        result['success']  = True  # symbolic success even if numeric fails

        # Numeric substitution
        free_syms = symbolic_solution.free_symbols
        tracker.resolve({str(s) for s in free_syms})

        # ── Vov → Vgs mapper ──────────────────────────────────────────────────
        # Precedence rule (deterministic, no ambiguity):
        #   1. Explicit user-supplied Vgs wins — Vov ignored for substitution.
        #   2. If user gave Vov but not Vgs, derive Vgs = Vth + Vov.
        #      Vth used is whatever the tracker holds (user-supplied or default).
        # This ensures displayed audit log and actual computation always agree.
        vov_entry = tracker._values.get('Vov')
        vgs_entry = tracker._values.get('Vgs')
        if (vov_entry is not None and vov_entry.provenance == 'user'
                and (vgs_entry is None or vgs_entry.provenance == 'default')):
            vth_val = tracker._values['Vth'].value if 'Vth' in tracker else \
                      ValueTracker.DEFAULTS['Vth'].value
            vgs_derived = vth_val + vov_entry.value
            tracker.add_user(
                'Vgs', vgs_derived, 'V',
                f"derived: Vth({vth_val} V) + user Vov({vov_entry.value} V)"
            )

        subs = tracker.get_subs_dict()

        try:
            numeric_expr = symbolic_solution.subs(subs)
            if numeric_expr.free_symbols:
                result['error'] = (
                    f"Unresolved symbols: {numeric_expr.free_symbols}. "
                    f"Numeric result unavailable."
                )
                return result
            numeric_val = float(numeric_expr.evalf())
        except Exception as exc:
            result['error'] = f"Numeric substitution: {type(exc).__name__}: {exc}"
            return result

        if math.isnan(numeric_val) or math.isinf(numeric_val):
            result['error'] = (
                f"Result is {'NaN' if math.isnan(numeric_val) else 'Inf'} — check units."
            )
            return result

        result['numeric'] = numeric_val

        # Sanity check
        lo, hi = self.SANITY_RANGES.get(target_symbol_name, (1e-30, 1e15))
        result['sanity_ok'] = lo <= abs(numeric_val) <= hi
        return result

    def _pick_physical_branch(self, solutions: list, target_name: str,
                              tracker: "ValueTracker"):
        """
        Given multiple SymPy solutions (e.g. from a quadratic), return the one
        that is physically meaningful.

        Strategy:
          1. Single solution → return immediately (no ambiguity).
          2. Evaluate all solutions numerically with current tracker values.
          3. Keep only solutions whose numeric value falls inside SANITY_RANGES.
          4. Among those, prefer the one with the SMALLEST absolute value
             (avoids large spurious branches, e.g. Vth from drain-current eq
             gives two roots; the smaller positive one is the physical threshold).
          5. If nothing survives the range filter, fall back to solutions[0]
             with a logged note — the caller will still show the symbolic form.
        """
        if len(solutions) == 1:
            return solutions[0]

        lo, hi = self.SANITY_RANGES.get(target_name, (1e-30, 1e15))
        subs = tracker.get_subs_dict()

        in_range: list[tuple[float, object]] = []
        for sol in solutions:
            try:
                val = float(sol.subs(subs).evalf())
                if lo <= val <= hi:
                    in_range.append((abs(val), sol))
            except Exception:
                pass  # un-evaluable branch (complex, etc.) — skip

        if in_range:
            # Smallest absolute value within the physical range
            in_range.sort(key=lambda t: t[0])
            return in_range[0][1]

        # Nothing in range — fall back to first solution
        return solutions[0]

    def solve(self, lhs_expr, rhs_expr, query: str, node_defaults=None) -> dict:
        """
        High-level wrapper for the pipeline.
        Orchestrates: detect_target -> extract_user_values -> solve_for.
        Returns the solve_result dict (check result['success']).
        The response string is in result['response'] via format_response().

        node_defaults: optional process-node profile defaults (dict of
        TrackedValue). None => generic 100nm baseline (unchanged behavior).
        """
        target = self.detect_target(query)
        if target is None:
            return {
                'success': False,
                'error': 'Could not identify a design target variable in the query.',
                'target': None, 'symbolic': None, 'numeric': None,
                'tracker': ValueTracker(), 'sanity_ok': False,
                'corpus_equation': '',
            }

        tracker = self.extract_user_values(query, node_defaults=node_defaults)
        corpus_eq_str = f"{lhs_expr} = {rhs_expr}"
        result = self.solve_for(lhs_expr, rhs_expr, target, tracker, corpus_eq_str)
        result['response'] = self.format_response(result)
        return result

    def solve_sweep(self, lhs_expr, rhs_expr, query: str,
                    sweep_req, node_defaults=None) -> dict:
        """
        SWEEP mode: derive the symbolic solution once (reusing solve_for —
        the locked core), then vectorize it over the sweep range. Never
        re-solves per point. Returns the normal solve result dict with two
        extra keys: 'sweep_result' (SweepResult) and 'sweep_plot_path' (set
        later by the caller after plot()).

        The target variable is whatever solve_for would pick for this query
        (e.g. WL); the sweep variable comes from sweep_req. If they collide
        (user asked to sweep the very thing we solve for), return an error so
        the pipeline falls back to single-point EXPLORE.
        """
        target = self.detect_target(query)
        if target is None:
            return {'success': False, 'error': 'No design target for sweep.',
                    'target': None, 'symbolic': None, 'numeric': None,
                    'tracker': ValueTracker(), 'sanity_ok': False,
                    'corpus_equation': '', 'sweep_result': None}

        if target == sweep_req.sweep_var:
            return {'success': False,
                    'error': f"Cannot sweep '{sweep_req.sweep_var}' — it is the "
                             f"target being solved for.",
                    'target': target, 'symbolic': None, 'numeric': None,
                    'tracker': ValueTracker(), 'sanity_ok': False,
                    'corpus_equation': '', 'sweep_result': None}

        tracker = self.extract_user_values(query, node_defaults=node_defaults)
        corpus_eq_str = f"{lhs_expr} = {rhs_expr}"
        result = self.solve_for(lhs_expr, rhs_expr, target, tracker, corpus_eq_str)

        if not result.get('success') or result.get('symbolic_expr') is None:
            result['sweep_result'] = None
            result['response'] = self.format_response(result)
            return result

        # Carry the active node name into the sweep request for the plot label
        if node_defaults:
            # node_defaults values carry the profile name in their description;
            # fall back to the request default otherwise.
            pass  # node_name already set on sweep_req by the caller

        # Vov sweep fix: if sweeping Vov, replace Vgs with Vov + Vth in the expression
        symbolic_expr = result['symbolic_expr']
        if sweep_req.sweep_var == 'Vov':
            import sympy as sp
            vgs_sym, vth_sym, vov_sym = sp.Symbol('Vgs'), sp.Symbol('Vth'), sp.Symbol('Vov')
            if vgs_sym in symbolic_expr.free_symbols:
                symbolic_expr = symbolic_expr.subs(vgs_sym, vov_sym + vth_sym)

        sweep_result = self._sweep_engine.run_sweep(
            sweep_req, symbolic_expr, tracker, target
        )
        result['sweep_result'] = sweep_result
        result['target_var'] = target

        # Compose a sweep-aware response (audit log still shows symbolic form)
        if sweep_result.error:
            result['response'] = (
                f"Equation: {result['symbolic']}\n"
                f"[SWEEP] Could not generate curve: {sweep_result.error}"
            )
        else:
            n = len(sweep_result.x)
            result['response'] = (
                f"Equation: {result['symbolic']}\n"
                f"[SWEEP MODE] {target} vs {sweep_req.sweep_var} over "
                f"[{sweep_req.start:.4g}, {sweep_req.stop:.4g}] — {n} points.\n"
                f"  Algebra (deterministic): SymPy solve() — no SLM call\n"
                f"  Node profile: {sweep_req.node_name}"
            )
        return result

    def format_response(self, solve_result: dict) -> str:
        """Build the full derivation audit response."""
        if not solve_result['success']:
            return (
                f"Equation: NOT FOUND IN CORPUS\n"
                f"[EXPLORE] Could not derive: {solve_result['error']}"
            )

        target  = solve_result['target']
        numeric = solve_result['numeric']
        numeric_str = f"{numeric:.4g}" if numeric is not None else "N/A (unresolved)"
        tracker: ValueTracker = solve_result['tracker']
        sanity  = ("[OK] within physical range" if solve_result['sanity_ok']
                   else "[WARN] outside typical range — check inputs")

        lines = [
            f"Equation: {solve_result['symbolic']}",
            "",
            "[EXPLORE MODE] Derivation audit:",
            f"  Base equation  (corpus-sourced) : {solve_result['corpus_equation']}",
            f"  Algebra        (deterministic)  : SymPy solve() — no SLM call",
            f"  Derived form                    : {solve_result['symbolic']}",
            "",
            f"  Numerical result : {target} = {numeric_str}",
            f"  Sanity check     : {sanity}",
            "",
            "  Substituted values:",
        ]
        lines.extend(tracker.audit_lines())

        # If Vov was user-supplied and Vgs was derived from it, add a summary
        # line showing (Vgs - Vth) explicitly. Kills the glance-level ambiguity
        # where an examiner sees Vgs=0.9 and has to verify it equals the Vov
        # they typed. After this line, there is nothing to reconstruct.
        vov_entry = tracker._values.get('Vov')
        vgs_entry = tracker._values.get('Vgs')
        vth_entry = tracker._values.get('Vth')
        if (vov_entry is not None and vov_entry.provenance == 'user'
                and vgs_entry is not None and vth_entry is not None):
            vgs_minus_vth = vgs_entry.value - vth_entry.value
            lines.append(
                f"    {'':9}  (Vgs - Vth) = {vgs_minus_vth:.4g} V"
                f"  <- matches your Vov = {vov_entry.value} V"
            )

        lines += [
            "",
            "  [DEFAULT] values are 100nm node assumptions, not user-supplied.",
            "  Provide your own process values for project-specific results.",
        ]
        return "\n".join(lines)
