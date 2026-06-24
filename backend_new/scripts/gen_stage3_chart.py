import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BG   = "#0F172A"; GRID = "#1E293B"; TEXT = "#F1F5F9"; SUB = "#94A3B8"
CACC = "#3B82F6"; CCTX = "#10B981"; CBAR = "#8B5CF6"

fig, axes = plt.subplots(1, 3, figsize=(17, 6), facecolor=BG)
fig.subplots_adjust(left=0.05, right=0.97, top=0.87, bottom=0.13, wspace=0.40)

def _style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=9.5)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    for sp in ["bottom","left"]: ax.spines[sp].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

# Panel 1: Hit@k and MRR overall
ax = axes[0]
_style(ax)
labels = ["Hit@1\n(16%)", "Hit@3\n(27%)", "MRR\n(0.21)"]
vals   = [0.160, 0.270, 0.210]
colors = [CACC, CACC, CCTX]
bars = ax.bar([0,1,2], vals, 0.45, color=colors, zorder=3, edgecolor=BG, linewidth=0.5)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.004, f"{v:.3f}",
            ha="center", va="bottom", fontsize=10, color=bar.get_facecolor(), fontweight="bold")
ax.set_xticks([0,1,2])
ax.set_xticklabels(labels, color=TEXT, fontsize=9.5)
ax.set_ylim(0, 0.45)
ax.set_ylabel("Rate", color=TEXT, fontsize=10)
ax.set_title("Retrieval Accuracy\n(threshold=0.70, exact source_chunk match)",
             color=TEXT, fontsize=10, fontweight="bold", pad=8)
ax.text(1.0, 0.36, "Strict proxy — same-section\nchunk boundary mismatch\naccounts for misses",
        ha="center", color=SUB, fontsize=8, fontstyle="italic")

# Panel 2: Threshold sensitivity
ax = axes[1]
_style(ax)
thresholds = [0.55, 0.60, 0.65, 0.70, 0.75]
h1_vals    = [0.480, 0.320, 0.240, 0.160, 0.080]
h3_vals    = [0.670, 0.530, 0.380, 0.270, 0.110]
mrr_vals   = [0.567, 0.415, 0.303, 0.210, 0.093]
x = np.arange(len(thresholds))
w = 0.25
b1 = ax.bar(x - w, h1_vals, w, color=CACC,   zorder=3, edgecolor=BG, linewidth=0.5, label="Hit@1")
b2 = ax.bar(x,     h3_vals, w, color="#60A5FA", zorder=3, edgecolor=BG, linewidth=0.5, label="Hit@3")
b3 = ax.bar(x + w, mrr_vals, w, color=CCTX,  zorder=3, edgecolor=BG, linewidth=0.5, label="MRR")
ax.set_xticks(x)
ax.set_xticklabels([f"t={t}" for t in thresholds], color=TEXT, fontsize=9)
ax.set_ylim(0, 0.85)
ax.set_ylabel("Rate", color=TEXT, fontsize=10)
ax.set_title("Threshold Sensitivity\n(Hit@1, Hit@3, MRR vs cosine threshold)",
             color=TEXT, fontsize=10, fontweight="bold", pad=8)
ax.legend(facecolor=GRID, edgecolor="none", labelcolor=TEXT, fontsize=8)
# Highlight primary threshold
ax.axvspan(2.5, 3.5, alpha=0.12, color="#F59E0B", zorder=0)
ax.text(3, 0.77, "Primary\n(0.70)", ha="center", color="#F59E0B", fontsize=8, fontweight="bold")
ax.axvspan(1.5, 2.5, alpha=0.08, color=CCTX, zorder=0)
ax.text(2, 0.77, "Alt\n(0.65)", ha="center", color=CCTX, fontsize=8)

# Panel 3: Context Relevancy by difficulty + Faithfulness comparison
ax = axes[2]
_style(ax)
diffs   = ["Easy\n(n=40)", "Medium\n(n=40)", "Hard\n(n=20)", "Overall\n(n=100)"]
ctx_rel = [0.501, 0.536, 0.550, 0.525]
faith_full = [None, None, None, 0.278]
faith_eq   = [None, None, None, 0.648]   # from Stage 2

x = np.arange(4); w = 0.32
b1 = ax.bar(x - w/2, ctx_rel, w, color=CCTX, zorder=3, edgecolor=BG, linewidth=0.5, label="Context Relevancy")
# Faithfulness bars only for overall
ax.bar([3 + w/2 - 0.16], [0.278], 0.16, color=CACC, zorder=3, edgecolor=BG, linewidth=0.5, label="Faith. (full ev.)")
ax.bar([3 + w/2],         [0.648], 0.16, color="#F59E0B", zorder=3, edgecolor=BG, linewidth=0.5, label="Faith. (corpus eq, S2)")

for bar, v in zip(b1, ctx_rel):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.006, f"{v:.3f}",
            ha="center", va="bottom", fontsize=9, color=CCTX, fontweight="bold")
ax.text(3 + w/2 - 0.16, 0.278+0.01, "0.278", ha="center", va="bottom",
        fontsize=8, color=CACC, fontweight="bold")
ax.text(3 + w/2, 0.648+0.01, "0.648", ha="center", va="bottom",
        fontsize=8, color="#F59E0B", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(diffs, color=TEXT, fontsize=9)
ax.set_ylim(0, 0.82)
ax.set_ylabel("Cosine Similarity", color=TEXT, fontsize=10)
ax.set_title("Context Relevancy (by difficulty)\n+ Faithfulness comparison (overall)",
             color=TEXT, fontsize=10, fontweight="bold", pad=8)
ax.legend(facecolor=GRID, edgecolor="none", labelcolor=TEXT, fontsize=7.5, loc="upper left")
ax.text(3.0, 0.74, "Eq. embed\nsignal\nstronger", ha="center", color=SUB, fontsize=7.5, fontstyle="italic")

fig.suptitle(
    "Stage 3 — RAG Retrieval Quality  (n=100, ground truth = source_chunk, hybrid retrieval)",
    color=TEXT, fontsize=13, fontweight="bold", y=0.97)
fig.text(0.5, 0.005,
    "Dense (FAISS, all-MiniLM-L6-v2) + Sparse (BM25) + Reranker (ms-marco-MiniLM-L-6-v2)  |  "
    "Hit = cosine_sim(source_chunk, retrieved_chunk) >= threshold",
    ha="center", color=SUB, fontsize=7.8)

OUT = r"f:\AMRITA ALL SEMESTER\SEMESTER-6\NLP\Physics_Based_RAG_SLM\Physics_Based_RAG_SLM\evaluation_stages\stage3_rag.png"
fig.savefig(OUT, dpi=160, facecolor=BG, bbox_inches="tight")
print("Saved:", OUT)
