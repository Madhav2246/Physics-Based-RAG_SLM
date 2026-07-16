import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT     = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "data" / "evaluation_new" / "stage4b_validator_hard" / "stage4b_validator_hard.json"
OUT_PNG   = ROOT.parent / "final_report" / "evaluation_stages" / "stage5_Physics-Score Selection under Generation Diversity.png"
OUT_PNG2  = ROOT.parent / "final_report" / "evaluation_stages" / "stage5_validator_power.png"

if not JSON_PATH.exists():
    raise FileNotFoundError(f"Expected real data at {JSON_PATH}.")

data = json.loads(JSON_PATH.read_text(encoding="utf-8"))

# ── Palette ───────────────────────────────────────────────────────────────────
BG    = "#FFFFFF"; GRID  = "#E2E8F0"; TEXT  = "#0F172A"; SUB   = "#475569"
CSYS  = "#1D4ED8"; CRAW  = "#64748B"; CGOOD = "#047857"
CBAD  = "#B91C1C"; CNEU  = "#D97706"; CPURP = "#701A75"

fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
fig.subplots_adjust(left=0.05, right=0.97, top=0.90, bottom=0.15, wspace=0.35)

def _style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=9.5)
    for sp in ["top", "right"]:  ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]: ax.spines[sp].set_color(TEXT)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

# ── Extract series ─────────────────────────────────────────────────────────────
ns_keys = ["n_1", "n_3", "n_5", "n_7", "n_9", "n_11", "n_13", "n_15", "n_17"]
ns_vals  = [1, 3, 5, 7, 9, 11, 13, 15, 17]

sys_gap  = [data[k]["_analysis"]["validator_gap_SYS"] for k in ns_keys]
raw_gap  = [data[k]["_analysis"]["validator_gap_RAW"] for k in ns_keys]
bon_gain = [data[k]["_analysis"]["bestofN_gain"]      for k in ns_keys]

# ── Panel 1: gap vs n_samples ──────────────────────────────────────────────────
ax = axes[0]
_style(ax)

x = np.array(ns_vals)
ax.plot(x, sys_gap,  "o-", color=CSYS,  lw=2.2, ms=7, label="SYS validator gap\n(best−rand, corpus-grounded)")
ax.plot(x, bon_gain, "s--",color=CNEU,  lw=1.8, ms=6, label="BoN diversity gain\n(sys best−first)")

# Annotate n=17 peak
ax.annotate(f"+{sys_gap[-1]:.3f}", xy=(17, sys_gap[-1]), xytext=(14.5, sys_gap[-1] + 0.05),
            arrowprops=dict(arrowstyle="->", color=CSYS, lw=1.3),
            color=CSYS, fontsize=9.5, fontweight="bold")

ax.set_xticks(ns_vals)
ax.set_xticklabels([f"n={v}" for v in ns_vals], color=TEXT, fontsize=9.5)
ax.set_ylim(-0.05, 1.80)
ax.set_xlabel("Samples per question", color=TEXT, fontsize=10)
ax.set_ylabel("Δ Score (0–4 scale)", color=TEXT, fontsize=10)
ax.set_title("Validator Discriminatory Power\nvs. Sample Count (hard questions)",
             color=TEXT, fontsize=10.5, fontweight="bold", pad=8)
ax.legend(loc="upper left", fontsize=8.5, facecolor=BG, edgecolor="none", labelcolor=TEXT)

# ── Panel 2: config breakdown vs n_samples ──────────────────────────────────────
ax = axes[1]
_style(ax)

sys_best  = [data[k]["sys_best"]["avg_score"] for k in ns_keys]
sys_first = [data[k]["sys_first"]["avg_score"] for k in ns_keys]
sys_rand  = [data[k]["sys_rand"]["avg_score"] for k in ns_keys]

ax.plot(x, sys_best,  "o-", color=CGOOD, lw=2.2, ms=7, label="sys_best (physics argmax)")
ax.plot(x, sys_first, "s-", color=CNEU,  lw=2.0, ms=7, label="sys_first (sample[0])")
ax.plot(x, sys_rand,  "v--",color=CRAW,  lw=1.8, ms=7, label="sys_rand (random)")

# Annotate sys_best peak at n=17
ax.annotate(f"sys_best = {sys_best[-1]:.3f}", xy=(17, sys_best[-1]), xytext=(13.0, sys_best[-1] + 0.08),
            arrowprops=dict(arrowstyle="->", color=CGOOD, lw=1.3),
            color=CGOOD, fontsize=9, fontweight="bold")

# Annotate flat sys_first
ax.text(9, 1.52, "sys_first flat ≈ 1.467", color=CNEU, fontsize=8.5, fontstyle="italic")

ax.set_xticks(ns_vals)
ax.set_xticklabels([f"n={v}" for v in ns_vals], color=TEXT, fontsize=9.5)
ax.set_ylim(0.8, 2.50)
ax.set_xlabel("Samples per question", color=TEXT, fontsize=10)
ax.set_ylabel("Physics Score (0–4)", color=TEXT, fontsize=10)
ax.set_title("Score by Config\n(sys_first flat vs. gains from selection)",
             color=TEXT, fontsize=10.5, fontweight="bold", pad=8)
ax.legend(loc="lower right", fontsize=8.5, facecolor=BG, edgecolor="none", labelcolor=TEXT)

# ── Panel 3: Stage 4 vs Stage 5 suppression → revelation ────────────────────
ax = axes[2]
_style(ax)

bar_labels = [
    "Stage 4\nablation\n(n=2 old)",
    "Stage 4\nnew (n=5)",
    "Stage 5\nn=7\n(hard only)",
    "Stage 5\nn=9\n(hard only)",
    "Stage 5\nn=17\n(hard only)",
]
sys_gaps_all = [0.013, 0.204, 0.606, 0.690, 0.902]

x3    = np.arange(len(bar_labels))
width = 0.45

b_sys = ax.bar(x3, sys_gaps_all, width,
               color=[CNEU, CSYS, CSYS, CSYS, CGOOD], zorder=3,
               edgecolor=BG, linewidth=0.5, label="SYS validator gap")

# Labels
for bar, v in zip(b_sys, sys_gaps_all):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.015,
            f"+{v:.3f}", ha="center", va="bottom",
            fontsize=9.5, color=bar.get_facecolor(), fontweight="bold")

# "suppressed" annotation on Stage 4
ax.annotate("Suppressed by easy questions\n+ low generation diversity",
            xy=(0, 0.013), xytext=(1.8, 0.35),
            arrowprops=dict(arrowstyle="->", color=CBAD, lw=1.2),
            color=CBAD, fontsize=8.5, ha="center", fontweight="bold")

ax.set_xticks(x3)
ax.set_xticklabels(bar_labels, color=TEXT, fontsize=8.5)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Validator Gap Δ Score", color=TEXT, fontsize=10)
ax.set_title("Stage 4 suppression → Stage 5 revealed\n(validator gap suppression & recovery)",
             color=TEXT, fontsize=10.5, fontweight="bold", pad=8)

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(OUT_PNG), dpi=300, facecolor=BG, bbox_inches="tight")
fig.savefig(str(OUT_PNG2), dpi=300, facecolor=BG, bbox_inches="tight")
print(f"Saved: {OUT_PNG}")
print(f"Saved: {OUT_PNG2}")
