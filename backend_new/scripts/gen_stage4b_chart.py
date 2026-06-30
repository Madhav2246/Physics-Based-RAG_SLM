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
JSON_PATH = ROOT / "data" / "evaluation" / "stage4b_validator_hard.json"
OUT_PNG   = ROOT.parent / "evaluation_stages" / "stage4b_validator.png"

if not JSON_PATH.exists():
    raise FileNotFoundError(f"Expected real data at {JSON_PATH}. Run Kaggle script first.")

data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
print(f"Loaded real data from {JSON_PATH}")

# ── Palette ───────────────────────────────────────────────────────────────────
BG    = "#0F172A"; GRID  = "#1E293B"; TEXT  = "#F1F5F9"; SUB   = "#94A3B8"
CSYS  = "#3B82F6"; CRAW  = "#9CA3AF"; CGOOD = "#10B981"
CBAD  = "#EF4444"; CNEU  = "#F59E0B"

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), facecolor=BG)
fig.subplots_adjust(left=0.06, right=0.97, top=0.86, bottom=0.14, wspace=0.38)

def _style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=9.5)
    for sp in ["top", "right"]:  ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]: ax.spines[sp].set_color(GRID)
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
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.015, f"{v:.3f}",
            ha="center", va="bottom", fontsize=9.5,
            color=bar.get_facecolor(), fontweight="bold")

sys_gap = data["n_5"]["_analysis"]["validator_gap_SYS"]
raw_gap = data["n_5"]["_analysis"]["validator_gap_RAW"]

# SYS gap bracket (best vs rand)
ax.annotate("", xy=(0, sys_vals[0]), xytext=(2, sys_vals[2]),
            arrowprops=dict(arrowstyle="<->", color=CSYS, lw=1.5))
ax.text(1.0, max(sys_vals) + 0.06,
        f"SYS validator gap = {sys_gap:+.3f}",
        ha="center", color=CSYS, fontsize=8.5, fontweight="bold")

# RAW gap bracket (best vs first)
ax.annotate("", xy=(4, raw_vals[0]), xytext=(5, raw_vals[1]),
            arrowprops=dict(arrowstyle="<->", color=CGOOD, lw=1.5))
ax.text(4.5, max(raw_vals) + 0.06,
        f"RAW validator gap = {raw_gap:+.3f}",
        ha="center", color=CGOOD, fontsize=8.5, fontweight="bold")

ax.set_xticks(list(x_sys) + list(x_raw))
ax.set_xticklabels(all_labels, color=TEXT, fontsize=8.2)
ax.set_ylim(0, max(sys_vals + raw_vals) + 0.28)
ax.set_ylabel("Physics Score (0–4)", color=TEXT, fontsize=10)
ax.set_title(f"Validator contribution at n=5 samples (hard questions)\n"
             f"SYS = corpus-grounded  |  RAW = no corpus_eq",
             color=TEXT, fontsize=10, fontweight="bold", pad=8)
ax.axvspan(3.3, 5.7, alpha=0.06, color=CGOOD, zorder=0)
ax.text(4.5, 0.06, "No corpus_eq\nclamping →\ntrue discriminator",
        ha="center", color=CGOOD, fontsize=7.5, fontstyle="italic")

# Stage 4 stored reference line
ax.axhline(1.214, color=CSYS, lw=0.9, linestyle=":", alpha=0.5, zorder=1)
ax.text(5.7, 1.225, "Stage 4\nfull\n1.214", color=CSYS, fontsize=6.5,
        alpha=0.7, ha="right")

# ── Panel 2: gap vs n-samples ─────────────────────────────────────────────────
ax = axes[1]
_style(ax)

ns_keys = ["n_1", "n_2", "n_3", "n_5"]
ns_vals  = [1, 2, 3, 5]
sys_gaps  = [data[k]["_analysis"]["validator_gap_SYS"] for k in ns_keys]
raw_gaps  = [data[k]["_analysis"]["validator_gap_RAW"] for k in ns_keys]
bon_gains = [data[k]["_analysis"]["bestofN_gain"]      for k in ns_keys]

x = np.array(ns_vals)
ax.plot(x, sys_gaps,  "o-", color=CSYS,  lw=2.2, ms=8, label="SYS validator gap\n(best−rand, SYS mode)")
ax.plot(x, raw_gaps,  "s-", color=CGOOD, lw=2.2, ms=8, label="RAW validator gap\n(best−first, RAW mode)")
ax.plot(x, bon_gains, "^--",color=CNEU,  lw=1.8, ms=7, label="BoN diversity gain\n(sys best−first)")

# Stage 4 reference
ax.axhline(0.013, color=CBAD, lw=1.2, linestyle=":", alpha=0.8)
ax.text(5.1, 0.025, "Stage 4\n+0.013\n(false neg.)", color=CBAD, fontsize=7.5)

# Annotate n=5
ax.annotate(f"SYS +{sys_gaps[-1]:.3f}\n(34× Stage 4)",
            xy=(5, sys_gaps[-1]), xytext=(3.8, 0.49),
            arrowprops=dict(arrowstyle="->", color=CSYS, lw=1.3),
            color=CSYS, fontsize=8.5, fontweight="bold", ha="center")

# Annotate n=2 RAW peak
ax.annotate(f"RAW +{raw_gaps[1]:.3f}",
            xy=(2, raw_gaps[1]), xytext=(2.5, 0.47),
            arrowprops=dict(arrowstyle="->", color=CGOOD, lw=1.1),
            color=CGOOD, fontsize=8, ha="left")

ax.set_xticks(ns_vals)
ax.set_xticklabels([f"n={v}" for v in ns_vals], color=TEXT, fontsize=10)
ax.set_ylim(-0.02, 0.58)
ax.set_xlabel("Samples per question", color=TEXT, fontsize=10)
ax.set_ylabel("Validator gap Δ Score (0–4)", color=TEXT, fontsize=10)
ax.set_title("Validator gap vs sample count (hard questions)\n"
             "False negative at n=2 mixed → real signal at n=5 hard",
             color=TEXT, fontsize=10, fontweight="bold", pad=8)
ax.legend(loc="upper left", fontsize=8, facecolor=GRID, edgecolor="none", labelcolor=TEXT)

# ── Supertitle + footer ───────────────────────────────────────────────────────
fig.suptitle(
    "Stage 4b — Validator Matters? Real data (hard questions, n∈{1,2,3,5}, Kaggle P100)",
    color=TEXT, fontsize=13, fontweight="bold", y=0.97)
fig.text(0.5, 0.005,
    "SYS gap = sys_best − sys_rand  |  RAW gap = raw_best − raw_first  |  "
    "BoN gain = sys_best − sys_first  |  n=20 hard questions",
    ha="center", color=SUB, fontsize=7.8)

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(OUT_PNG), dpi=160, facecolor=BG, bbox_inches="tight")
print(f"Saved: {OUT_PNG}")
