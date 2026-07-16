"""
gen_stage4b_chart.py
--------------------
Generates stage4b_validator.png from stage4b_validator_hard.json (real Kaggle data).
Two panels:
  Left  — Score per config at n=5 (SYS vs RAW grouped bars + gap annotations)
  Right — Validator gap vs n-samples (SYS gap / RAW gap / BoN gain)

Run from backend_new/:
  python scripts/gen_stage4b_chart.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT      = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "data" / "evaluation_new" / "stage4b_validator_hard" / "stage4b_validator_hard.json"
OUT_PNG   = ROOT.parent / "final_report" / "evaluation_stages" / "stage4b_validator.png"

if not JSON_PATH.exists():
    raise FileNotFoundError(f"Expected real data at {JSON_PATH}.")

data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
print(f"Loaded real data from {JSON_PATH}")

# ── Palette ───────────────────────────────────────────────────────────────────
BG    = "#FFFFFF"; GRID  = "#E2E8F0"; TEXT  = "#0F172A"; SUB   = "#475569"
CSYS  = "#1D4ED8"; CRAW  = "#64748B"; CGOOD = "#047857"
CBAD  = "#B91C1C"; CNEU  = "#D97706"

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
fig.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.15, wspace=0.35)

def _style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=9.5)
    for sp in ["top", "right"]:  ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]: ax.spines[sp].set_color(TEXT)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

# ── Panel 1: config breakdown at n=5 ─────────────────────────────────────────
ax = axes[0]
_style(ax)

n5 = data["n_5"]
config_names  = ["sys_best", "sys_first", "sys_rand", "raw_best", "raw_first"]
sys_labels = [f"SYS best-of-5\n(physics sel)", f"SYS first\n(−bestofN)", f"SYS random\n(−validator)"]
raw_labels = [f"RAW best-of-5\n(physics sel)", "RAW first\n(baseline)"]
all_labels = sys_labels + raw_labels

x_sys = np.array([0, 1, 2])
x_raw = np.array([4, 5])

sys_vals = [n5[c]["avg_score"] for c in ["sys_best", "sys_first", "sys_rand"]]
raw_vals = [n5[c]["avg_score"] for c in ["raw_best", "raw_first"]]

b_sys = ax.bar(x_sys, sys_vals, 0.5,
               color=[CGOOD, CNEU, CNEU], zorder=3, edgecolor=BG, linewidth=0.5)
b_raw = ax.bar(x_raw, raw_vals, 0.5,
               color=[CGOOD, CRAW], zorder=3, edgecolor=BG, linewidth=0.5)

for bar, v in zip(list(b_sys) + list(b_raw), sys_vals + raw_vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.3f}",
            ha="center", va="bottom", fontsize=9.5,
            color=bar.get_facecolor(), fontweight="bold")

sys_gap = data["n_5"]["_analysis"]["validator_gap_SYS"]
raw_gap = data["n_5"]["_analysis"]["validator_gap_RAW"]

# SYS gap bracket (best vs rand)
ax.annotate("", xy=(0, sys_vals[0] + 0.05), xytext=(2, sys_vals[2] + 0.05),
            arrowprops=dict(arrowstyle="<->", color=CSYS, lw=1.5))
ax.text(1.0, max(sys_vals) + 0.07,
        f"SYS validator gap = {sys_gap:+.3f}",
        ha="center", color=CSYS, fontsize=8.5, fontweight="bold")

# RAW gap bracket (best vs first)
ax.annotate("", xy=(4, raw_vals[0] + 0.05), xytext=(5, raw_vals[1] + 0.05),
            arrowprops=dict(arrowstyle="<->", color=CGOOD, lw=1.5))
ax.text(4.5, max(raw_vals) + 0.07,
        f"RAW validator gap = {raw_gap:+.3f}",
        ha="center", color=CGOOD, fontsize=8.5, fontweight="bold")

ax.set_xticks(list(x_sys) + list(x_raw))
ax.set_xticklabels(all_labels, color=TEXT, fontsize=8.5)
ax.set_ylim(0, max(sys_vals + raw_vals) + 0.35)
ax.set_ylabel("Physics Score (0–4)", color=TEXT, fontsize=10)
ax.set_title("Validator Contribution at n=5 Samples (hard questions)",
             color=TEXT, fontsize=10.5, fontweight="bold", pad=8)
ax.axvspan(3.5, 5.5, alpha=0.08, color=CGOOD, zorder=0)

# ── Panel 2: gap vs n-samples ─────────────────────────────────────────────────
ax = axes[1]
_style(ax)

ns_keys = ["n_1", "n_3", "n_5", "n_7", "n_9", "n_11", "n_13", "n_15", "n_17"]
ns_vals  = [1, 3, 5, 7, 9, 11, 13, 15, 17]
sys_gaps  = [data[k]["_analysis"]["validator_gap_SYS"] for k in ns_keys]
raw_gaps  = [data[k]["_analysis"]["validator_gap_RAW"] for k in ns_keys]
bon_gains = [data[k]["_analysis"]["bestofN_gain"]      for k in ns_keys]

x = np.array(ns_vals)
ax.plot(x, sys_gaps,  "o-", color=CSYS,  lw=2.2, ms=7, label="SYS validator gap (best−rand)")
ax.plot(x, raw_gaps,  "s-", color=CGOOD, lw=2.2, ms=7, label="RAW validator gap (best−first)")
ax.plot(x, bon_gains, "^--",color=CNEU,  lw=1.8, ms=6, label="BoN diversity gain (sys best−first)")

# Annotate n=17
ax.annotate(f"SYS +{sys_gaps[-1]:.3f}",
            xy=(17, sys_gaps[-1]), xytext=(14.5, sys_gaps[-1] - 0.08),
            arrowprops=dict(arrowstyle="->", color=CSYS, lw=1.3),
            color=CSYS, fontsize=8.5, fontweight="bold", ha="center")

ax.set_xticks(ns_vals)
ax.set_xticklabels([f"n={v}" for v in ns_vals], color=TEXT, fontsize=9)
ax.set_ylim(-0.05, 1.80)
ax.set_xlabel("Samples per question", color=TEXT, fontsize=10)
ax.set_ylabel("Validator Gap Δ Score (0–4)", color=TEXT, fontsize=10)
ax.set_title("Validator Gap vs. Sample Count (hard questions)",
             color=TEXT, fontsize=10.5, fontweight="bold", pad=8)
ax.legend(loc="upper left", fontsize=8.5, facecolor=BG, edgecolor="none", labelcolor=TEXT)

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(OUT_PNG), dpi=300, facecolor=BG, bbox_inches="tight")
print(f"Saved: {OUT_PNG}")

