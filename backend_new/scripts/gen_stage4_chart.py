import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Academic Palette
BG   = "#FFFFFF"; GRID = "#E2E8F0"; TEXT = "#0F172A"; SUB  = "#475569"
CFULL = "#1D4ED8"; CGOOD = "#047857"; CBAD = "#B91C1C"; CNEU  = "#D97706"
CRAW  = "#475569"

fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor=BG)
fig.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.15, wspace=0.35)

def _style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=9.5)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    for sp in ["bottom","left"]: ax.spines[sp].set_color(TEXT)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

# ── Panel 1: Physics score per config (horizontal bar) ──────────────────────
ax = axes[0]
_style(ax)

configs = ["raw_0.5b", "-bestofN", "-validator", "-dense", "-sparse", "-reranker", "full"]
scores  = [0.555,      1.061,      1.157,        1.194,    1.234,     1.254,       1.361 ]
colors  = [CRAW,       CBAD,       CBAD,         CNEU,     CNEU,      CGOOD,       CFULL ]
labels  = ["Raw 0.5B\n(no retrieval)", "−Best-of-N\n(n=1)", "−Validator\n(random sel.)",
           "−Dense\n(BM25 only)", "−Sparse\n(Dense only)", "−Reranker\n(RRF only)", "Full System\n★ baseline"]

y = np.arange(len(configs))
bars = ax.barh(y, scores, 0.55, color=colors, zorder=3, edgecolor=BG, linewidth=0.5)
for bar, v, c in zip(bars, scores, colors):
    ax.text(v + 0.02, bar.get_y()+bar.get_height()/2, f"{v:.3f}",
            va="center", ha="left", fontsize=9.5, color=c, fontweight="bold")

ax.set_yticks(y)
ax.set_yticklabels(labels, color=TEXT, fontsize=8.5)
ax.set_xlim(0, 1.65)
ax.set_xlabel("Physics Score (0–4 scale)", color=TEXT, fontsize=10)
ax.set_title("Ablation — Physics Score per Config\n(removing any component degrades correctness)",
             color=TEXT, fontsize=10.5, fontweight="bold", pad=8)

# Full system reference line
ax.axvline(1.361, color=CFULL, lw=1.2, linestyle="--", alpha=0.7, zorder=4)
ax.text(1.37, 0.3, "Full System", color=CFULL, fontsize=8, fontweight="bold")

# ── Panel 2: Contributions bar chart ────────────────────────────────────────
ax = axes[1]
ax.set_facecolor(BG)
ax.tick_params(colors=TEXT, labelsize=9.5)
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
for sp in ["bottom","left"]: ax.spines[sp].set_color(TEXT)
ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
ax.set_axisbelow(True)

comp_names  = ["Raw 0.5B\n(floor)", "Corpus Eq.\nGrounding", "Best-of-N\nSelection",
               "Physics\nScore Sel.", "Dense\nRetriever", "Sparse\nRetriever", "Reranker"]
comp_vals   = [0.555, 0.806, 0.300, 0.204, 0.167, 0.127, 0.107]
comp_colors = [CRAW,  CGOOD, CGOOD, CGOOD, CGOOD, CGOOD, CGOOD]

x_pos = np.arange(len(comp_names))
bars2 = ax.bar(x_pos, comp_vals, 0.55, color=comp_colors, zorder=3, edgecolor=BG, linewidth=0.5)
for bar, v, c in zip(bars2, comp_vals, comp_colors):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f"+{v:.3f}" if v != comp_vals[0] else f"{v:.3f}",
            ha="center", va="bottom", fontsize=9, color=c, fontweight="bold")

ax.set_xticks(x_pos)
ax.set_xticklabels(comp_names, color=TEXT, fontsize=8.5, rotation=15)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score Contribution (Δ or absolute)", color=TEXT, fontsize=10)
ax.set_title("Component Contributions to Physics Score\n(Δ = gain from adding this component)",
             color=TEXT, fontsize=10.5, fontweight="bold", pad=8)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT.parent / "final_report" / "evaluation_stages" / "stage4_ablation.png"
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(OUT), dpi=300, facecolor=BG, bbox_inches="tight")
print("Saved:", OUT)
