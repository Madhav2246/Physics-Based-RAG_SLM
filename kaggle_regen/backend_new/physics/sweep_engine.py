"""
SweepEngine — parametric sweeps over a derived design equation.

Reuses the symbolic solution from ExplorationEngine.solve_for() — never
re-solves. Vectorizes one variable over a range, holds everything else fixed
from the ValueTracker, and renders a matplotlib trade-off curve.

Build discipline: parse_sweep_request() is verified in isolation
(scripts/_sweep_range_test.py) before any plotting code runs.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional


# Unit suffix → SI multiplier (shared with the query value parser)
_UNIT_SCALE = {
    "pa": 1e-12, "na": 1e-9, "ua": 1e-6, "ma": 1e-3, "a": 1.0,
    "uv": 1e-6, "mv": 1e-3, "kv": 1e3, "v": 1.0,
    "pm": 1e-12, "nm": 1e-9, "um": 1e-6, "mm": 1e-3, "m": 1.0,
    "": 1.0,
}

# Variables that can be swept / targeted (canonical names)
_KNOWN_VARS = [
    "Id", "Vov", "Vgs", "Vth", "Vds", "Vsb", "gamma", "tox", "L", "W",
    "mu", "Cox", "Phi_f", "WL", "gm", "ro", "lam",
]


@dataclass
class SweepRequest:
    sweep_var:  str                 # variable that varies, e.g. "Vov"
    target_var: Optional[str]       # variable plotted on Y, e.g. "WL" (may be None — inferred later)
    start:      float
    stop:       float
    n_points:   int = 50
    node_name:  str = "100nm_CMOS"  # active process node profile


def _canonical_var(token: str) -> Optional[str]:
    """Map a raw token like 'w/l', 'vov', 'id' to a canonical variable name."""
    t = token.strip().lower().replace(" ", "")
    aliases = {
        "w/l": "WL", "wl": "WL", "aspectratio": "WL",
        "id": "Id", "draincurrent": "Id",
        "vov": "Vov", "overdrive": "Vov", "overdrivevoltage": "Vov",
        "vgs": "Vgs", "vth": "Vth", "vsb": "Vsb", "vds": "Vds",
        "gamma": "gamma", "tox": "tox", "cox": "Cox",
        "mu": "mu", "gm": "gm", "ro": "ro", "phi_f": "Phi_f",
        "lam": "lam", "lambda": "lam",
        "l": "L", "w": "W",
    }
    return aliases.get(t)


def _parse_value_unit(num: str, unit: str) -> float:
    """'0.1' + 'V' -> 0.1 ; '10' + 'mA' -> 0.01."""
    scale = _UNIT_SCALE.get(unit.strip().lower(), 1.0)
    return float(num) * scale


def parse_sweep_request(query: str) -> Optional[SweepRequest]:
    """
    Parse a sweep query into a SweepRequest, or return None if the query is
    not a sweep (caller then falls through to single-point EXPLORE).

    Handles forms like:
      "Plot W/L versus Vov from 0.1V to 0.6V for Id=1mA"
      "sweep Id from 1uA to 10mA"
      "Vov vs W/L, range 0.2 to 0.8"
    """
    q = query.strip()
    ql = q.lower()

    # Must look like a sweep at all
    if not any(kw in ql for kw in ("plot", "sweep", "versus", " vs ", "vs.", "range", "from")):
        return None

    # ── 1. Find the range: "from X[unit] to Y[unit]"  or  "range X to Y" ──────
    num = r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
    unit = r'\s*([a-zA-Z]{0,3})'
    range_patterns = [
        rf'from\s+{num}{unit}\s+to\s+{num}{unit}',
        rf'range\s+{num}{unit}\s+to\s+{num}{unit}',
        rf'{num}{unit}\s+to\s+{num}{unit}',
    ]
    start = stop = None
    for pat in range_patterns:
        m = re.search(pat, q, re.IGNORECASE)
        if m:
            start = _parse_value_unit(m.group(1), m.group(2))
            stop  = _parse_value_unit(m.group(3), m.group(4))
            break
    if start is None:
        return None

    # ── 2. Identify sweep + target variables ──────────────────────────────────
    # Two grammars coexist; the range keyword disambiguates which:
    #   "from" binds to the variable immediately before it  → that var sweeps
    #       "W/L versus Vov from 0.1 to 0.6"  → Vov sweeps (before 'from')
    #   "range" is a standalone clause        → first var of the vs-pair sweeps
    #       "Vov vs W/L, range 0.2 to 0.8"    → Vov sweeps (first in pair)
    sweep_var = None
    target_var = None

    # Parse the "A versus/vs B" pair if present
    pair = re.search(r'(?:plot\s+)?([A-Za-z][\w/]*)\s+(?:versus|vs\.?)\s+([A-Za-z][\w/]*)',
                     q, re.IGNORECASE)
    pair_left  = _canonical_var(pair.group(1)) if pair else None
    pair_right = _canonical_var(pair.group(2)) if pair else None

    if re.search(r'\bfrom\b', ql):
        # "from" form: variable immediately before 'from' is the sweep var
        m = re.search(r'([A-Za-z][\w/]*)\s+from\b', q, re.IGNORECASE)
        if m:
            sweep_var = _canonical_var(m.group(1))
        # explicit "sweep <var> from"
        if sweep_var is None:
            m = re.search(r'sweep\s+([A-Za-z][\w/]*)', q, re.IGNORECASE)
            if m:
                sweep_var = _canonical_var(m.group(1))
        # target = the other side of the vs-pair
        if pair_left and pair_right:
            target_var = pair_left if pair_right == sweep_var else pair_right
    else:
        # "range" form (no 'from'): first var of the vs-pair sweeps
        if pair_left:
            sweep_var  = pair_left
            target_var = pair_right
        else:
            m = re.search(r'sweep\s+([A-Za-z][\w/]*)', q, re.IGNORECASE)
            if m:
                sweep_var = _canonical_var(m.group(1))

    if sweep_var is None:
        return None

    return SweepRequest(
        sweep_var=sweep_var,
        target_var=target_var,
        start=start,
        stop=stop,
    )


def parse_multi_curve_param(query: str) -> Optional[tuple[str, list[float]]]:
    """
    Detect a family-of-curves parameter spec in the query.

    Supported forms:
      "for Id = 0.5mA, 1mA, 2mA"
      "for Id in [0.5mA, 1mA, 2mA]"
      "at Id = 0.5mA, 1mA, 2mA"

    Returns (canonical_param_name, [SI_values]) when ≥2 values found, else None.
    Unit of first token is inherited by subsequent bare numbers.
    """
    m = re.search(
        r'(?:for|at)\s+([A-Za-z][\w/]*)\s*(?:=|:|\bin\b)\s*\[?'
        r'([\d.]+(?:[eE][+-]?\d+)?\s*[a-zA-Z]{0,3}'
        r'(?:\s*,\s*[\d.]+(?:[eE][+-]?\d+)?\s*[a-zA-Z]{0,3})+)\]?',
        query, re.IGNORECASE
    )
    if not m:
        return None

    param_raw = m.group(1)
    param = _canonical_var(param_raw) or param_raw

    num_unit_re = re.compile(
        r'([\d.]+(?:[eE][+-]?\d+)?)\s*([a-zA-Z]{0,3})'
    )
    tokens = num_unit_re.findall(m.group(2))
    if len(tokens) < 2:
        return None

    first_unit = tokens[0][1].lower()
    si_vals: list[float] = []
    for num, unit in tokens:
        u = unit.lower() if unit.strip() else first_unit
        scale = _UNIT_SCALE.get(u, 1.0)
        si_vals.append(float(num) * scale)

    return param, si_vals


# ─────────────────────────────────────────────────────────────────────────────
# Sweep execution + plotting
# ─────────────────────────────────────────────────────────────────────────────

# Display units per variable (for axis labels)
_VAR_UNIT = {
    "Id": "A", "Vov": "V", "Vgs": "V", "Vth": "V", "Vds": "V", "Vsb": "V",
    "gamma": "V^0.5", "tox": "m", "L": "m", "W": "m",
    "mu": "m^2/Vs", "Cox": "F/m^2", "Phi_f": "V", "WL": "(ratio)",
    "gm": "A/V", "ro": "Ohm", "lam": "1/V",
}


@dataclass
class SweepResult:
    sweep_var:  str
    target_var: str
    x:          list = field(default_factory=list)   # sweep variable values
    y:          list = field(default_factory=list)   # derived target values
    node_name:  str = "100nm_CMOS"
    fixed:      dict = field(default_factory=dict)    # {sym: value} held constant
    error:      str = ""


class SweepEngine:
    """
    Parametric sweep over an already-derived symbolic solution.
    Never re-solves — takes the SymPy expression from ExplorationEngine.
    """

    def detect_sweep(self, query: str) -> Optional[SweepRequest]:
        return parse_sweep_request(query)

    def run_sweep(self, sweep_req: SweepRequest, symbolic_solution,
                  tracker, target_var: str) -> SweepResult:
        """
        Vectorize the symbolic solution over the sweep variable.

        symbolic_solution : the SymPy expression for target_var, already solved
                            by ExplorationEngine.solve_for() — we do NOT re-solve.
        tracker           : ValueTracker holding all fixed values.
        """
        import numpy as np
        import sympy as sp

        result = SweepResult(
            sweep_var=sweep_req.sweep_var,
            target_var=target_var or "result",
            node_name=sweep_req.node_name,
        )

        sweep_sym = sp.Symbol(sweep_req.sweep_var)

        # Resolve every symbol in the solution except the sweep variable
        needed = {str(s) for s in symbolic_solution.free_symbols}
        tracker.resolve(needed)
        subs = tracker.get_subs_dict()
        subs.pop(sweep_sym, None)   # ensure the swept var stays free

        expr = symbolic_solution.subs(subs)
        remaining = expr.free_symbols - {sweep_sym}
        if remaining:
            result.error = (
                f"Cannot sweep — unresolved symbols after substitution: "
                f"{', '.join(str(s) for s in remaining)}."
            )
            return result

        # Record which values were held fixed (for the provenance annotation)
        result.fixed = {
            k: v.value for k, v in tracker._values.items()
            if k != sweep_req.sweep_var
        }

        try:
            f = sp.lambdify(sweep_sym, expr, modules="numpy")
            xs = np.linspace(sweep_req.start, sweep_req.stop, sweep_req.n_points)
            ys = f(xs)
            # Broadcast scalar results (constant expr) to array shape
            ys = np.broadcast_to(np.asarray(ys, dtype=float), xs.shape)
            result.x = xs.tolist()
            result.y = ys.tolist()
        except Exception as exc:
            result.error = f"Sweep evaluation failed: {type(exc).__name__}: {exc}"

        return result

    def plot(self, result: SweepResult, output_path: str,
             title: Optional[str] = None) -> str:
        """Render a single-curve trade-off plot to a PNG. Returns the path."""
        import matplotlib
        matplotlib.use("Agg")   # headless — no display needed
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(result.x, result.y, color="#0aa", linewidth=2)

        sx_unit = _VAR_UNIT.get(result.sweep_var, "")
        ty_unit = _VAR_UNIT.get(result.target_var, "")
        ax.set_xlabel(f"{result.sweep_var} [{sx_unit}]")
        ax.set_ylabel(f"{result.target_var} [{ty_unit}]")
        ax.set_title(title or f"{result.target_var} vs {result.sweep_var}  ({result.node_name})")
        ax.grid(True, alpha=0.3)

        # Provenance annotation (bottom-right)
        prov = self._provenance_note(result)
        ax.annotate(prov, xy=(0.98, 0.02), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=8, color="#555",
                    bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#ccc", alpha=0.8))

        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=120)
        plt.close(fig)
        return output_path

    def plot_multi(self, results: list, labels: list[str],
                  output_path: str, title: str = None) -> str:
        """
        Family-of-curves: multiple SweepResults on one axes, one line per label.
        Coloured with plasma colormap; legend shows the varied parameter values.
        Returns saved path.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(9, 5.5))
        colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(results)))

        for res, label, color in zip(results, labels, colors):
            ax.plot(res.x, res.y, label=label, color=color, linewidth=2)

        r0 = results[0]
        sx_unit = _VAR_UNIT.get(r0.sweep_var, "")
        ty_unit = _VAR_UNIT.get(r0.target_var, "")
        ax.set_xlabel(f"{r0.sweep_var}  [{sx_unit}]", fontsize=11)
        ax.set_ylabel(f"{r0.target_var}  [{ty_unit}]", fontsize=11)
        ax.set_title(
            title or f"{r0.target_var} vs {r0.sweep_var}  —  family of curves  ({r0.node_name})",
            fontsize=12,
        )
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9, framealpha=0.85)

        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=130)
        plt.close(fig)
        return output_path

    @staticmethod
    def _provenance_note(result: SweepResult) -> str:
        keys = [k for k in ("mu", "Cox", "tox", "Vth") if k in result.fixed]
        if keys:
            return f"{', '.join(keys)}: {result.node_name} defaults"
        return f"node: {result.node_name}"
