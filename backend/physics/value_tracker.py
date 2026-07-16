"""
ValueTracker — single source of truth for all numeric values in explore mode.

Every value is tagged with its provenance (user / corpus / default).
Three consumers read from this one object:
  - ExplorationEngine.solve_for()     — substitution dict
  - ConfidenceEngine.score_explore()  — provenance_fraction
  - Derivation audit log              — audit_lines()

This keeps the zero-hallucination story honest once numbers enter the picture.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Literal

Provenance = Literal["user", "corpus", "default"]


@dataclass
class TrackedValue:
    symbol:      str
    value:       float
    unit:        str
    provenance:  Provenance
    description: str = ""


class ValueTracker:
    """
    Registry of numeric values used in a single explore-mode query.

    Usage:
        tracker = ValueTracker()
        tracker.add_user("Id", 1e-3, "A", "user-supplied")
        tracker.resolve({"mu", "Cox"})   # fills defaults
        subs = tracker.get_subs_dict()   # {Symbol("Id"): 1e-3, ...}
    """

    # Standard 100 nm MOSFET node defaults — must match NumericalValidator.test_values
    DEFAULTS: Dict[str, TrackedValue] = {
        "mu":    TrackedValue("mu",    0.05,    "m^2/Vs",  "default", "electron mobility (100nm node)"),
        "Cox":   TrackedValue("Cox",   0.02,    "F/m^2",   "default", "gate oxide capacitance (100nm node)"),
        "W":     TrackedValue("W",     1e-6,    "m",       "default", "gate width (100nm node)"),
        "L":     TrackedValue("L",     1e-7,    "m",       "default", "gate length (100nm node)"),
        "Vgs":   TrackedValue("Vgs",   1.0,     "V",       "default", "gate-source voltage"),
        "Vth":   TrackedValue("Vth",   0.4,     "V",       "default", "threshold voltage"),
        "Vov":   TrackedValue("Vov",   0.6,     "V",       "default", "overdrive voltage"),
        "Vds":   TrackedValue("Vds",   1.0,     "V",       "default", "drain-source voltage"),
        "Vth0":  TrackedValue("Vth0",  0.5,     "V",       "default", "zero-bias threshold voltage"),
        "Vsb":   TrackedValue("Vsb",   0.0,     "V",       "default", "source-body voltage"),
        "Phi_f": TrackedValue("Phi_f", 0.35,    "V",       "default", "Fermi potential"),
        "gamma": TrackedValue("gamma", 0.4,     "V^0.5",   "default", "body-effect coefficient"),
        "k":     TrackedValue("k",     1.38e-23,"J/K",     "corpus",  "Boltzmann constant"),
        "T":     TrackedValue("T",     300.0,   "K",       "default", "temperature"),
        "q":     TrackedValue("q",     1.6e-19, "C",       "corpus",  "elementary charge"),
        "tox":   TrackedValue("tox",   2e-9,    "m",       "default", "gate oxide thickness (100nm node)"),
        "n":     TrackedValue("n",     1.0,     "-",       "default", "ideality factor"),
        "lam":   TrackedValue("lam",   0.1,     "1/V",     "default", "channel-length modulation"),
    }

    def __init__(self, defaults: Dict[str, TrackedValue] | None = None) -> None:
        self._values: Dict[str, TrackedValue] = {}
        # Per-instance defaults (e.g. a node profile). Falls back to the
        # class-level DEFAULTS when not provided — behavior unchanged for
        # all existing callers that do ValueTracker().
        self._defaults: Dict[str, TrackedValue] = defaults or dict(self.DEFAULTS)

    @classmethod
    def with_node_profile(cls, node_params: Dict[str, TrackedValue]) -> "ValueTracker":
        """
        Build a ValueTracker pre-loaded with a node profile's defaults.
        node_params already has every entry tagged provenance='default'
        (produced by NodeProfileManager.as_tracker_defaults()).
        """
        return cls(defaults=node_params)

    # ── Public write API ────────────────────────────────────────────────────

    def add_user(self, symbol: str, value: float, unit: str,
                 description: str = "from query") -> None:
        """Register a value the user explicitly supplied."""
        self._values[symbol] = TrackedValue(symbol, value, unit, "user", description)

    def add_corpus(self, symbol: str, value: float, unit: str,
                   description: str = "from corpus") -> None:
        """Register a value extracted directly from a corpus chunk."""
        self._values[symbol] = TrackedValue(symbol, value, unit, "corpus", description)

    def resolve(self, symbols_needed: set) -> "ValueTracker":
        """
        Fill any unregistered symbols from this instance's defaults
        (a node profile if one was loaded, else the 100nm DEFAULTS).
        Final fallback to class DEFAULTS guarantees constants (k, q, T) exist.
        Call after user values are registered, before numeric substitution.
        """
        for sym in symbols_needed:
            sym_str = str(sym)
            if sym_str in self._values:
                continue
            if sym_str in self._defaults:
                self._values[sym_str] = self._defaults[sym_str]
            elif sym_str in self.DEFAULTS:   # safety net — never lose a constant
                self._values[sym_str] = self.DEFAULTS[sym_str]
        return self

    # ── Public read API ─────────────────────────────────────────────────────

    def get_subs_dict(self) -> dict:
        """Return {sympy.Symbol(name): value} for SymPy .subs()."""
        import sympy as sp
        return {sp.Symbol(k): v.value for k, v in self._values.items()}

    @property
    def provenance_fraction(self) -> float:
        """
        Fraction of registered values that are user-supplied.
        0.0 = all defaults/corpus, 1.0 = all user.
        Used by ConfidenceEngine.score_explore() to penalise assumed values.
        """
        if not self._values:
            return 0.0
        user_count = sum(1 for v in self._values.values() if v.provenance == "user")
        return user_count / len(self._values)

    def audit_lines(self) -> list[str]:
        """
        Formatted lines for the derivation audit log.
        Clearly separates user-supplied from assumed defaults.
        """
        lines = []
        for v in self._values.values():
            tag = f"[{v.provenance.upper():<7}]"
            lines.append(
                f"    {tag} {v.symbol:<10} = {v.value:.4g} {v.unit}"
                f"  ({v.description})"
            )
        return lines

    def __contains__(self, key: str) -> bool:
        return key in self._values

    def __repr__(self) -> str:
        return f"ValueTracker({list(self._values.keys())})"
