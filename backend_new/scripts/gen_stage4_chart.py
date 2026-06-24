import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BG   = "#0F172A"; GRID = "#1E293B"; TEXT = "#F1F5F9"; SUB  = "#94A3B8"
CFULL = "#3B82F6"; CGOOD = "#10B981"; CBAD = "#EF4444"; CNEU = "#F59E0B"
CRAW  = "#9CA3AF"

fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
fig.subplots_adjust(left=0.06, right=0.97, top=0.87, bottom=0.14, wspace=0.38)

def _style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=9.5)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    for sp in ["bottom","left"]: ax.spines[sp].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

# ── Panel 1: Physics score per config (horizontal bar) ──────────────────────
ax = axes[0]
_style(ax)

configs = ["raw_0.5b", "-validator", "-bestofN", "-dense", "full", "-reranker", "-sparse"]
scores  = [0.433,       1.091,        1.104,      1.173,   1.214,   1.288,       1.411 ]
colors  = [CRAW, CBAD, CBAD, CNEU, CFULL, CGOOD, CGOOD]
labels  = ["Raw 0.5B\n(no retrieval)", "−Validator\n(random sel.)", "−BestOfN\n(n=1)",
           "−Dense\n(BM25 only)", "Full System\n★ baseline", "−Reranker\n(RRF only)",
           "−Sparse\n(Dense only)"]

y = np.arange(len(configs))
bars = ax.barh(y, scores, 0.55, color=colors, zorder=3, edgecolor=BG, linewidth=0.5)
for bar, v, c in zip(bars, scores, colors):
    ax.text(v + 0.01, bar.get_y()+bar.get_height()/2, f"{v:.3f}",
            va="center", ha="left", fontsize=9.5, color=c, fontweight="bold")

ax.set_yticks(y)
ax.set_yticklabels(labels, color=TEXT, fontsize=8.5)
ax.set_xlim(0, 1.65)
ax.set_xlabel("Physics Score (0–4 scale)", color=TEXT, fontsize=10)
ax.set_title("Ablation — Physics Score per Config\n(↑ = component removal improved eq. recovery; ↓ = it hurt)",
             color=TEXT, fontsize=10, fontweight="bold", pad=8)

# Full system reference line
ax.axvline(1.214, color=CFULL, lw=1.2, linestyle="--", alpha=0.7, zorder=4)
ax.text(1.216, 0.3, "Full\nsystem", color=CFULL, fontsize=7.5)

# Region labels
ax.text(1.30, 6.4, "Dense-only best\nfor eq. recovery", color=CGOOD, fontsize=8, fontstyle="italic")
ax.text(0.05, 0.4, "No retrieval\n= baseline floor", color=CRAW, fontsize=8, fontstyle="italic")

# -validator bar is index 1 (y=1) — annotate false negative
# configs sorted ascending: raw_0.5b(0), -validator(1), -bestofN(2), -dense(3), full(4), -reranker(5), -sparse(6)
ax.text(1.095, 1.0, "⚠ Stage 5: gap=+0.445\n   (hard, n=5)", color=CBAD,
        fontsize=6.8, va="center", ha="left", fontstyle="italic")

# Legend
patches = [
    mpatches.Patch(color=CGOOD, label="Removing component ↑ score"),
    mpatches.Patch(color=CFULL, label="Full system"),
    mpatches.Patch(color=CNEU,  label="Marginal effect (±0.05)"),
    mpatches.Patch(color=CBAD,  label="Removing component ↓ score"),
    mpatches.Patch(color=CRAW,  label="No retrieval (floor)"),
]
ax.legend(handles=patches, loc="lower right", fontsize=7.5,
          facecolor=GRID, edgecolor="none", labelcolor=TEXT)

# ── Panel 2: Waterfall / delta chart ────────────────────────────────────────
ax = axes[1]
_style(ax)

# Waterfall bars showing contribution steps
steps = [
    ("Raw 0.5B\n(floor)",          0.433,  0.0,   CRAW,  "0.433"),
    ("+ Corpus Equation\n(retrieval)", 0.433, 0.781, CGOOD, "+0.781"),
    ("+ Best-of-N\n(n=2 → pick best)", 1.214, 0.110, CGOOD, "+0.110"),
    ("+ Physics\nSelection",        1.214+0.110, 0.013, CGOOD, "+0.013"),
    ("Full System\n(local)",        0.0,   1.214, CFULL, "1.214"),
]
# Simpler: just show delta contributions as a bar chart
comp_names  = ["Raw 0.5B\n(floor)", "Corpus Eq.\nGrounding", "Best-of-N\nSelection",
               "Physics\nScore Sel.", "Dense vs\nBM25 upgrade"]
comp_vals   = [0.433, 0.781, 0.110, 0.013, 0.197]
comp_colors = [CRAW,  CGOOD, CGOOD, CGOOD, CNEU]
comp_style  = ["solid","solid","solid","solid","dashed"]

x = np.arange(len(comp_names))
bars = ax.bar(x, comp_vals, 0.5, color=comp_colors, zorder=3, edgecolor=BG, linewidth=0.5)
for bar, v, c in zip(bars, comp_vals, comp_colors):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.008, f"+{v:.3f}" if v != comp_vals[0] else f"{v:.3f}",
            ha="center", va="bottom", fontsize=9.5, color=c, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(comp_names, color=TEXT, fontsize=9)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score contribution (Δ or absolute)", color=TEXT, fontsize=10)
ax.set_title("Component Contributions to Physics Score\n(Δ = gain from adding this component; Raw floor shown absolute)",
             color=TEXT, fontsize=10, fontweight="bold", pad=8)

# Annotate "optional upgrade"
ax.text(4, 0.22, "Optional:\nDense-only\nupgrade", color=CNEU, fontsize=8,
        ha="center", fontstyle="italic")
ax.annotate("", xy=(4, 0.197), xytext=(4, 0.21),
            arrowprops=dict(arrowstyle="->", color=CNEU, lw=1))
ax.text(1.5, 0.88, "Retrieval grounding\n= dominant factor", color=CGOOD,
        fontsize=9, ha="center", fontweight="bold")

# Stage 5 false-negative warning on Physics Score Sel. bar (index 3)
ax.annotate("⚠ false negative\n(n=2, easy mix)\nStage 5: +0.445",
            xy=(3, 0.013), xytext=(3, 0.18),
            arrowprops=dict(arrowstyle="->", color=CBAD, lw=1.1),
            color=CBAD, fontsize=7.2, ha="center", fontweight="bold")

fig.suptitle(
    "Stage 4 — Ablation Study  (n=100, new_checker v2, seed=42)",
    color=TEXT, fontsize=13, fontweight="bold", y=0.97)
fig.text(0.5, 0.005,
    "Generation ablations use stored answers_dump.jsonl  |  "
    "Retrieval ablations re-run on CPU  |  LoRA ablation: GPU required (not shown)",
    ha="center", color=SUB, fontsize=7.8)

OUT = r"f:\AMRITA ALL SEMESTER\SEMESTER-6\NLP\Physics_Based_RAG_SLM\Physics_Based_RAG_SLM\evaluation_stages\stage4_ablation.png"
fig.savefig(OUT, dpi=160, facecolor=BG, bbox_inches="tight")
print("Saved:", OUT)
