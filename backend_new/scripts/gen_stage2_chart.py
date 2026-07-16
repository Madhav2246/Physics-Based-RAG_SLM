import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Professional academic palette
BG  = "#FFFFFF"; GRID = "#E2E8F0"; TEXT = "#0F172A"; SUB = "#475569"
C70 = "#D97706"; CSYS = "#1D4ED8"; CRAW = "#64748B"  # Muted amber, dark blue, slate grey

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), facecolor=BG)
fig.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.15, wspace=0.35)

def _style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=9.5)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    for sp in ["bottom","left"]: ax.spines[sp].set_color(TEXT)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

# Panel 1: BERTScore + ROUGE-L + BLEU (scaled)
ax = axes[0]
_style(ax)
metrics  = ["BERTScore\nF1", "ROUGE-L\n(x10)", "BLEU-4\n(x100)"]
sys_vals = [0.8205, 0.1644*10, 0.0347*100]
raw_vals = [0.8213, 0.1661*10, 0.0281*100]
x = np.arange(3); w = 0.32
b1 = ax.bar(x - w/2, sys_vals, w, color=CSYS, zorder=3, edgecolor=BG, linewidth=0.5)
b2 = ax.bar(x + w/2, raw_vals, w, color=CRAW, zorder=3, edgecolor=BG, linewidth=0.5)
for bars, vals, col in [(b1, sys_vals, CSYS), (b2, raw_vals, CRAW)]:
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.1, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8.5, color=col, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metrics, color=TEXT, fontsize=9.5)
ax.set_ylabel("Score", color=TEXT, fontsize=10)
ax.set_ylim(0, 5)
ax.set_title("Lexical \& Semantic Overlap\n(vs. 70B reference)",
             color=TEXT, fontsize=10.5, fontweight="bold", pad=8)
ax.legend(handles=[mpatches.Patch(color=CSYS, label="Complete System (SYS)"),
                   mpatches.Patch(color=CRAW, label="Raw 0.5B (RAW)")],
          facecolor=BG, edgecolor="none", labelcolor=TEXT, fontsize=8.5, loc="upper right")

# Panel 2: Faithfulness
ax = axes[1]
_style(ax)
labels = ["Complete\nSystem", "70B Baseline\n(ungrounded)"]
vals   = [0.7334, 0.1480]
cols   = [CSYS, C70]
bars   = ax.bar([0, 1], vals, 0.45, color=cols, zorder=3, edgecolor=BG, linewidth=0.5)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.3f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
            color=bar.get_facecolor())
ax.set_xticks([0, 1])
ax.set_xticklabels(labels, color=TEXT, fontsize=9.5)
ax.set_ylim(0, 0.85)
ax.set_ylabel("Cosine Similarity to Corpus Equation", color=TEXT, fontsize=10)
ax.set_title("Faithfulness to Retrieved Evidence\n(cosine similarity to corpus equation)",
             color=TEXT, fontsize=10.5, fontweight="bold", pad=8)
ax.annotate("5.0x more grounded", xy=(0, 0.733), xytext=(0.45, 0.77),
            arrowprops=dict(arrowstyle="->", color=CSYS, lw=1.5),
            fontsize=9.5, color=CSYS, fontweight="bold")
ax.axhline(0.148, color=C70, lw=0.8, linestyle="--", alpha=0.5)

# Panel 3: Answer Relevancy
ax = axes[2]
_style(ax)
sides = ["Complete\nSystem", "Raw\n0.5B", "70B\nBaseline"]
rel   = [0.5695, 0.6268, 0.7115]
cols2 = [CSYS, CRAW, C70]
bars  = ax.bar([0,1,2], rel, 0.45, color=cols2, zorder=3, edgecolor=BG, linewidth=0.5)
for bar, v in zip(bars, rel):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
            color=bar.get_facecolor())
ax.set_xticks([0,1,2])
ax.set_xticklabels(sides, color=TEXT, fontsize=9.5)
ax.set_ylim(0, 0.90)
ax.set_ylabel("Cosine Similarity (answer vs. question)", color=TEXT, fontsize=10)
ax.set_title("Answer Relevancy\n(sentence embed cosine sim to question)",
             color=TEXT, fontsize=10.5, fontweight="bold", pad=8)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT.parent / "final_report" / "evaluation_stages" / "stage2_generation.png"
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(OUT), dpi=300, facecolor=BG, bbox_inches="tight")
print("Saved:", OUT)

