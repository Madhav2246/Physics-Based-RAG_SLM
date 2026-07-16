"""
_sweep_demo.py — Parametric sweep demo across all four process node profiles.

Plots W/L versus Vov (0.1V → 0.6V) for Id=1mA on all four nodes
overlaid on the same axes, saved to data/evaluation/sweep_demo.png.

Run from the backend/ directory:
    python scripts/_sweep_demo.py
"""
from __future__ import annotations
import sys, io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physics.exploration_engine import ExplorationEngine
from physics.equation_validator  import EquationValidator
from physics.node_profile_manager import NodeProfileManager
from physics.sweep_engine import SweepRequest, SweepEngine

# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
validator = EquationValidator()
engine    = ExplorationEngine(validator.symbols)
npm       = NodeProfileManager()
sweep_eng = SweepEngine()

QUERY = "Plot W/L versus Vov from 0.1V to 0.6V for Id=1mA"
CORPUS_CHUNK = "Id = 0.5 * mu * Cox * (W/L) * (Vgs - Vth)^2"

lhs, rhs, msg = validator.validate(CORPUS_CHUNK)
assert lhs is not None, f"Parse failed: {msg}"

NODES = npm.list_nodes()
COLORS = ["#0066cc", "#00aa66", "#cc6600", "#cc0044"]
NODE_LABELS = {
    "100nm_CMOS": "100nm CMOS",
    "28nm_FDSOI": "28nm FDSOI",
    "16nm_FinFET": "16nm FinFET",
    "5nm_GAA":    "5nm GAA",
}

# --------------------------------------------------------------------------- #
# Run sweep on each node
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(9, 6))

print("=" * 60)
print("SWEEP DEMO — W/L vs Vov across 4 process nodes")
print("=" * 60)

from physics.sweep_engine import SweepRequest, parse_sweep_request
sweep_req_template = parse_sweep_request(QUERY)
assert sweep_req_template is not None, "Sweep parse failed"

for node_name, color in zip(NODES, COLORS):
    node_defaults = npm.as_tracker_defaults(node_name)

    sweep_req = SweepRequest(
        sweep_var  = sweep_req_template.sweep_var,
        target_var = sweep_req_template.target_var,
        start      = sweep_req_template.start,
        stop       = sweep_req_template.stop,
        n_points   = 60,
        node_name  = node_name,
    )

    result = engine.solve_sweep(lhs, rhs, QUERY, sweep_req,
                                node_defaults=node_defaults)

    sr = result.get("sweep_result")
    if sr and not sr.error:
        ax.plot(sr.x, sr.y, color=color, linewidth=2,
                label=NODE_LABELS.get(node_name, node_name))
        wl_at_mid = np.interp(0.35, sr.x, sr.y)
        print(f"  {NODE_LABELS.get(node_name, node_name):<18} | "
              f"W/L at Vov=0.35V = {wl_at_mid:.2f}")
    else:
        err = sr.error if sr else result.get("error", "unknown")
        print(f"  {node_name:<20} | FAILED: {err}")

print("=" * 60)

# --------------------------------------------------------------------------- #
# Style the plot
# --------------------------------------------------------------------------- #
ax.set_xlabel("Vov  [V]", fontsize=13)
ax.set_ylabel("W/L  (ratio)", fontsize=13)
ax.set_title("Trade-off Curve: W/L vs Vov  (Id = 1 mA)\nAcross process node profiles",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=11, loc="upper right")
ax.grid(True, alpha=0.3)
ax.annotate("Algebra: SymPy solve() | Values: node profile defaults",
            xy=(0.02, 0.02), xycoords="axes fraction",
            ha="left", va="bottom", fontsize=8, color="#666",
            bbox=dict(boxstyle="round", fc="#f9f9f9", ec="#ccc", alpha=0.8))

out = Path(__file__).parent.parent / "data" / "evaluation" / "sweep_demo.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(out, dpi=150)
plt.close(fig)

print(f"\nPlot saved to: {out}")
