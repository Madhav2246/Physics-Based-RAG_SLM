import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BG  = "#0F172A"; GRID = "#1E293B"; TEXT = "#F1F5F9"; SUB = "#94A3B8"
C70 = "#F59E0B"; CSYS = "#3B82F6"; CRAW = "#9CA3AF"

fig, axes = plt.subplots(1, 3, figsize=(17, 6), facecolor=BG)
fig.subplots_adjust(left=0.05, right=0.97, top=0.87, bottom=0.13, wspace=0.38)

def _style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=9.5)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    for sp in ["bottom","left"]: ax.spines[sp].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

# Panel 1: BERTScore + ROUGE-L + BLEU (scaled)
ax = axes[0]
_style(ax)
metrics  = ["BERTScore\nF1", "ROUGE-L\n(x10)", "BLEU-4\n(x100)"]
sys_vals = [0.8155, 0.1127*10, 0.0231*100]
raw_vals = [0.8038, 0.1042*10, 0.0227*100]
x = np.arange(3); w = 0.32
b1 = ax.bar(x - w/2, sys_vals, w, color=CSYS, zorder=3, edgecolor=BG, linewidth=0.5)
b2 = ax.bar(x + w/2, raw_vals, w, color=CRAW, zorder=3, edgecolor=BG, linewidth=0.5)
for bars, vals, col in [(b1, sys_vals, CSYS), (b2, raw_vals, CRAW)]:
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.1, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8, color=col, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metrics, color=TEXT, fontsize=9)
ax.set_ylabel("Score", color=TEXT, fontsize=10)
ax.set_ylim(0, 12)
ax.set_title("Lexical & Semantic Overlap\n(vs 70B reference — bias noted)",
             color=TEXT, fontsize=10, fontweight="bold", pad=8)
ax.legend(handles=[mpatches.Patch(color=CSYS, label="Complete System"),
                   mpatches.Patch(color=CRAW, label="Raw 0.5B")],
          facecolor=GRID, edgecolor="none", labelcolor=TEXT, fontsize=8, loc="upper right")
ax.text(1.0, 10.8, "ROUGE/BLEU scaled for visibility", ha="center",
        color=SUB, fontsize=7.5, fontstyle="italic")

# Panel 2: Faithfulness
ax = axes[1]
_style(ax)
labels = ["Complete\nSystem", "70B Baseline\n(ungrounded)"]
vals   = [0.6479, 0.1396]
cols   = [CSYS, C70]
bars   = ax.bar([0, 1], vals, 0.45, color=cols, zorder=3, edgecolor=BG, linewidth=0.5)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.3f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
            color=bar.get_facecolor())
ax.set_xticks([0, 1])
ax.set_xticklabels(labels, color=TEXT, fontsize=9.5)
ax.set_ylim(0, 0.85)
ax.set_ylabel("Cosine Sim to Corpus Equation", color=TEXT, fontsize=10)
ax.set_title("Faithfulness to Retrieved Evidence\n(n=79, cosine sim to corpus equation)",
             color=TEXT, fontsize=10, fontweight="bold", pad=8)
ax.annotate("4.6x more grounded", xy=(0, 0.648), xytext=(0.45, 0.73),
            arrowprops=dict(arrowstyle="->", color=CSYS, lw=1.5),
            fontsize=9, color=CSYS, fontweight="bold")
ax.axhline(0.1396, color=C70, lw=0.8, linestyle="--", alpha=0.5)

# Panel 3: Answer Relevancy
ax = axes[2]
_style(ax)
sides = ["Complete\nSystem", "Raw\n0.5B", "70B\nBaseline"]
rel   = [0.4284, 0.4225, 0.7115]
cols2 = [CSYS, CRAW, C70]
bars  = ax.bar([0,1,2], rel, 0.45, color=cols2, zorder=3, edgecolor=BG, linewidth=0.5)
for bar, v in zip(bars, rel):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
            color=bar.get_facecolor())
ax.set_xticks([0,1,2])
ax.set_xticklabels(sides, color=TEXT, fontsize=9.5)
ax.set_ylim(0, 0.90)
ax.set_ylabel("Cosine Sim (answer vs question)", color=TEXT, fontsize=10)
ax.set_title("Answer Relevancy\n(sentence embed cosine sim to question)",
             color=TEXT, fontsize=10, fontweight="bold", pad=8)
ax.text(1.0, 0.80,
        "70B advantage = model size,\nnot architecture", ha="center",
        color=SUB, fontsize=8, fontstyle="italic")

fig.suptitle(
    "Stage 2 — Generation Quality  (n=100, answers_dump.jsonl, new_checker v2)",
    color=TEXT, fontsize=13, fontweight="bold", y=0.97)
fig.text(0.5, 0.005,
    "BERTScore: roberta-large  |  Faithfulness/Relevancy: all-MiniLM-L6-v2  |  "
    "Reference for overlap metrics = 70B (bias acknowledged)",
    ha="center", color=SUB, fontsize=7.8)

OUT = r"f:\AMRITA ALL SEMESTER\SEMESTER-6\NLP\Physics_Based_RAG_SLM\Physics_Based_RAG_SLM\evaluation_stages\stage2_generation.png"
fig.savefig(OUT, dpi=160, facecolor=BG, bbox_inches="tight")
print("Saved:", OUT)
