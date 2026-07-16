import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Professional academic palette
BG   = "#FFFFFF"; GRID = "#E2E8F0"; TEXT = "#0F172A"; SUB  = "#475569"
C1   = "#1D4ED8"; C2   = "#047857"; C3   = "#D97706"; C4   = "#64748B"

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), facecolor=BG)
fig.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.15, wspace=0.35)

def _style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=9.5)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    for sp in ["bottom","left"]: ax.spines[sp].set_color(TEXT)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

# ── Panel 1: Overall Retrieval Performance ────────────────────────────────────
ax = axes[0]
_style(ax)
metrics = ["Hit@1", "Hit@3", "MRR"]
vals = [0.560, 0.940, 0.715]
bars = ax.bar(metrics, vals, 0.45, color=[C1, C1, C2], zorder=3, edgecolor=BG, linewidth=0.5)

for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.015, f"{v:.3f}",
            ha="center", va="bottom", fontsize=10, color=bar.get_facecolor(), fontweight="bold")

ax.set_ylim(0, 1.10)
ax.set_ylabel("Score", color=TEXT, fontsize=10)
ax.set_title("Overall Retrieval Performance\n(threshold=0.50 anchor word overlap)",
             color=TEXT, fontsize=10.5, fontweight="bold", pad=8)

# ── Panel 2: Performance by Cohort ───────────────────────────────────────────
ax = axes[1]
_style(ax)

cohorts = ["Easy\n(n=40)", "Medium\n(n=40)", "Hard\n(n=20)"]
x = np.arange(len(cohorts))
width = 0.25

# Data: Easy, Medium, Hard
h1_vals  = [0.625, 0.500, 0.550]
h3_vals  = [0.950, 0.925, 0.950]
mrr_vals = [0.758, 0.667, 0.725]

b1 = ax.bar(x - width, h1_vals, width, color=C1, zorder=3, edgecolor=BG, linewidth=0.5, label="Hit@1")
b2 = ax.bar(x,         h3_vals, width, color=C3, zorder=3, edgecolor=BG, linewidth=0.5, label="Hit@3")
b3 = ax.bar(x + width, mrr_vals, width, color=C2, zorder=3, edgecolor=BG, linewidth=0.5, label="MRR")

for bars_group in [b1, b2, b3]:
    for bar in bars_group:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.015, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8, color=bar.get_facecolor(), fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(cohorts, color=TEXT, fontsize=9.5)
ax.set_ylim(0, 1.10)
ax.set_ylabel("Score", color=TEXT, fontsize=10)
ax.set_title("Retrieval Quality by Difficulty\n(robust performance across strata)",
             color=TEXT, fontsize=10.5, fontweight="bold", pad=8)
ax.legend(facecolor=BG, edgecolor="none", labelcolor=TEXT, fontsize=8.5, loc="upper right")

# ── Panel 3: Ablation of Retrieval Components ─────────────────────────────────
ax = axes[2]
_style(ax)

configs = ["Dense-only\n(FAISS)", "Hybrid RRF\n(Dense+Sparse)", "CrossEncoder\nRerank (SYS)"]
ab_vals = [0.650, 0.780, 0.940] # Hit@3 comparison
bars_ab = ax.bar(configs, ab_vals, 0.45, color=[C4, C3, C1], zorder=3, edgecolor=BG, linewidth=0.5)

for bar, v in zip(bars_ab, ab_vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.015, f"{v:.3f}",
            ha="center", va="bottom", fontsize=10, color=bar.get_facecolor(), fontweight="bold")

ax.set_ylim(0, 1.10)
ax.set_ylabel("Hit@3 Score", color=TEXT, fontsize=10)
ax.set_title("Impact of Retrieval Pipeline\n(reranking fused candidates lifts Hit@3 to 94%)",
             color=TEXT, fontsize=10.5, fontweight="bold", pad=8)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT.parent / "final_report" / "evaluation_stages" / "stage3_rag.png"
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(OUT), dpi=300, facecolor=BG, bbox_inches="tight")
print("Saved:", OUT)
