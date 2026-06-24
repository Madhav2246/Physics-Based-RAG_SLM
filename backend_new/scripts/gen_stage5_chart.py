"""
gen_stage5_chart.py
-------------------
Generates stage5_validator_power.png from stage4b_validator_hard.json.

Three panels:
  Left   — Validator gap (SYS & RAW) vs n_samples + BoN gain overlay
  Middle — Score per config at n=5 (full breakdown)
  Right  — Stage 4 vs Stage 5 contrast: suppression → revelation

Run from backend_new/:
  python scripts/gen_stage5_chart.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT     = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "data" / "evaluation" / "stage4b_validator_hard.json"
OUT_PNG   = ROOT.parent / "evaluation_stages" / "stage5_validator_power.png"

data = json.loads(JSON_PATH.read_text(encoding="utf-8"))

# ── Palette ───────────────────────────────────────────────────────────────────
BG    = "#0F172A"; GRID  = "#1E293B"; TEXT  = "#F1F5F9"; SUB   = "#94A3B8"
CSYS  = "#3B82F6"; CRAW  = "#9CA3AF"; CGOOD = "#10B981"
CBAD  = "#EF4444"; CNEU  = "#F59E0B"; CPURP = "#A855F7"

fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), facecolor=BG)
fig.subplots_adjust(left=0.05, right=0.97, top=0.86, bottom=0.14, wspace=0.38)

def _style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=9.5)
    for sp in ["top", "right"]:  ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]: ax.spines[sp].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

# ── Extract series ─────────────────────────────────────────────────────────────
ns_keys = ["n_1", "n_2", "n_3", "n_5"]
ns_vals  = [1, 2, 3, 5]

sys_gap  = [data[k]["_analysis"]["validator_gap_SYS"] for k in ns_keys]
raw_gap  = [data[k]["_analysis"]["validator_gap_RAW"] for k in ns_keys]
bon_gain = [data[k]["_analysis"]["bestofN_gain"]      for k in ns_keys]

# ── Panel 1: gap vs n_samples ──────────────────────────────────────────────────
ax = axes[0]
_style(ax)

x = np.array(ns_vals)
ax.plot(x, sys_gap,  "o-", color=CSYS,  lw=2.2, ms=7, label="SYS validator gap\n(best−rand, corpus-grounded)")
ax.plot(x, raw_gap,  "s-", color=CGOOD, lw=2.2, ms=7, label="RAW validator gap\n(best−first, no corpus_eq)")
ax.plot(x, bon_gain, "^--",color=CNEU,  lw=1.8, ms=6, label="BoN diversity gain\n(sys best−first)")

# Stage 4 reference line (+0.013)
ax.axhline(0.013, color=CBAD, lw=1.2, linestyle=":", alpha=0.8, zorder=1)
ax.text(5.1, 0.025, "Stage 4\nablation\n+0.013", color=CBAD, fontsize=7.5, va="bottom")

# Annotate n=5 peak
ax.annotate(f"+0.445", xy=(5, 0.445), xytext=(4.0, 0.48),
            arrowprops=dict(arrowstyle="->", color=CSYS, lw=1.3),
            color=CSYS, fontsize=8.5, fontweight="bold")
ax.annotate(f"34× Stage 4", xy=(5, 0.445), xytext=(3.4, 0.445),
            color=CSYS, fontsize=7.5, fontstyle="italic")

ax.set_xticks(ns_vals)
ax.set_xticklabels([f"n={v}" for v in ns_vals], color=TEXT, fontsize=10)
ax.set_ylim(-0.02, 0.58)
ax.set_xlabel("Samples per question", color=TEXT, fontsize=10)
ax.set_ylabel("Δ Score (0–4 scale)", color=TEXT, fontsize=10)
ax.set_title("Validator discriminatory power\nvs sample count  (hard questions only)",
             color=TEXT, fontsize=10, fontweight="bold", pad=8)
ax.legend(loc="upper left", fontsize=7.8, facecolor=GRID, edgecolor="none", labelcolor=TEXT)

# ── Panel 2: config breakdown at n=5 ──────────────────────────────────────────
ax = axes[1]
_style(ax)

n5 = data["n_5"]
config_names  = ["sys_best", "sys_first", "sys_rand", "raw_best", "raw_first"]
config_labels = [
    f"SYS best-of-5\n(physics sel)",
    f"SYS first\n(−bestofN)",
    f"SYS random\n(−validator)",
    f"RAW best-of-5\n(physics sel)",
    f"RAW first\n(baseline)",
]
config_scores = [n5[c]["avg_score"] for c in config_names]
config_parse  = [n5[c]["parseable"] for c in config_names]
config_colors = [CGOOD, CNEU, CNEU, CSYS, CRAW]

x2 = np.arange(len(config_names))
bars = ax.bar(x2, config_scores, 0.52, color=config_colors,
              zorder=3, edgecolor=BG, linewidth=0.5)
for bar, v, c in zip(bars, config_scores, config_colors):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.3f}",
            ha="center", va="bottom", fontsize=9.5, color=c, fontweight="bold")

# Parse% as text below bar
for i, (p, c) in enumerate(zip(config_parse, config_colors)):
    ax.text(i, 0.05, f"parse\n{p:.0f}%", ha="center", va="bottom",
            fontsize=7.5, color=c, alpha=0.85)

# Gap brackets
# SYS gap: sys_best vs sys_rand
y_top = max(config_scores) + 0.12
ax.annotate("", xy=(0, n5["sys_best"]["avg_score"]),
            xytext=(2, n5["sys_rand"]["avg_score"]),
            arrowprops=dict(arrowstyle="<->", color=CSYS, lw=1.5))
ax.text(1.0, y_top - 0.04, f"SYS gap = +{data['n_5']['_analysis']['validator_gap_SYS']:.3f}",
        ha="center", color=CSYS, fontsize=8.5, fontweight="bold")

# RAW gap: raw_best vs raw_first
ax.annotate("", xy=(3, n5["raw_best"]["avg_score"]),
            xytext=(4, n5["raw_first"]["avg_score"]),
            arrowprops=dict(arrowstyle="<->", color=CGOOD, lw=1.5))
ax.text(3.5, y_top - 0.04, f"RAW gap = +{data['n_5']['_analysis']['validator_gap_RAW']:.3f}",
        ha="center", color=CGOOD, fontsize=8.5, fontweight="bold")

ax.set_xticks(x2)
ax.set_xticklabels(config_labels, color=TEXT, fontsize=7.8)
ax.set_ylim(0, y_top + 0.06)
ax.set_ylabel("Physics Score (0–4)", color=TEXT, fontsize=10)
ax.set_title("Config breakdown at n=5\n(hard questions, Kaggle P100)",
             color=TEXT, fontsize=10, fontweight="bold", pad=8)
ax.axvspan(2.6, 4.4, alpha=0.06, color=CSYS, zorder=0)
ax.text(3.5, 0.25, "No corpus_eq\n→ true\ndiscriminator",
        ha="center", color=CSYS, fontsize=7.5, fontstyle="italic")

# ── Panel 3: Stage 4 vs Stage 5 suppression → revelation ────────────────────
ax = axes[2]
_style(ax)

bar_labels = [
    "Stage 4\nablation\n(mixed, n=2\nstored)",
    "Stage 5\nn=2\n(hard only)",
    "Stage 5\nn=3\n(hard only)",
    "Stage 5\nn=5\n(hard only)",
]
sys_gaps_all = [0.013, 0.250, 0.250, 0.445]
raw_gaps_all = [None, 0.450, 0.375, 0.420]

x3    = np.arange(len(bar_labels))
width = 0.35

b_sys = ax.bar(x3 - width/2, sys_gaps_all, width,
               color=[CNEU, CSYS, CSYS, CSYS], zorder=3,
               edgecolor=BG, linewidth=0.5, label="SYS validator gap")
b_raw_vals = [v if v else 0 for v in raw_gaps_all]
b_raw = ax.bar(x3 + width/2, b_raw_vals, width,
               color=[CRAW, CGOOD, CGOOD, CGOOD], zorder=3,
               edgecolor=BG, linewidth=0.5, label="RAW validator gap")

# Labels
for bar, v in zip(b_sys, sys_gaps_all):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
            f"+{v:.3f}", ha="center", va="bottom",
            fontsize=9, color=bar.get_facecolor(), fontweight="bold")
for bar, v, orig in zip(b_raw, b_raw_vals, raw_gaps_all):
    if orig:
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                f"+{v:.3f}", ha="center", va="bottom",
                fontsize=9, color=bar.get_facecolor(), fontweight="bold")
    else:
        ax.text(bar.get_x() + bar.get_width()/2, 0.015,
                "n/a\n(stored)", ha="center", va="bottom",
                fontsize=7, color=CRAW, alpha=0.7)

# "suppressed" arrow on Stage 4
ax.annotate("Suppressed by\ncorpus_eq clamping\n+ easy questions",
            xy=(0 - width/2, 0.013), xytext=(0.5, 0.30),
            arrowprops=dict(arrowstyle="->", color=CBAD, lw=1.2),
            color=CBAD, fontsize=7.5, ha="center")

ax.set_xticks(x3)
ax.set_xticklabels(bar_labels, color=TEXT, fontsize=8.5)
ax.set_ylim(0, 0.58)
ax.set_ylabel("Validator gap Δ Score", color=TEXT, fontsize=10)
ax.set_title("Stage 4 false-negative → Stage 5 revealed\n(validator gap suppression & recovery)",
             color=TEXT, fontsize=10, fontweight="bold", pad=8)
ax.legend(loc="upper left", fontsize=8, facecolor=GRID,
          edgecolor="none", labelcolor=TEXT)

# ── Supertitle + footer ────────────────────────────────────────────────────────
fig.suptitle(
    "Stage 5 — Validator Power Analysis  (hard questions, n∈{1,2,3,5} samples, Kaggle P100)",
    color=TEXT, fontsize=13, fontweight="bold", y=0.97)
fig.text(0.5, 0.005,
    "SYS gap = sys_best − sys_rand  |  RAW gap = raw_best − raw_first  |  "
    "BoN gain = sys_best − sys_first  |  n=20 hard questions",
    ha="center", color=SUB, fontsize=7.8)

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(OUT_PNG), dpi=160, facecolor=BG, bbox_inches="tight")
print(f"Saved: {OUT_PNG}")
