import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STAGE1_JSON = ROOT / "data" / "evaluation_new" / "stage1_rescored.json"
OUT_PNG = ROOT.parent / "final_report" / "evaluation_stages" / "stage1_physics.png"

if not STAGE1_JSON.exists():
    raise FileNotFoundError(f"Expected real data at {STAGE1_JSON}.")

pq = json.loads(STAGE1_JSON.read_text(encoding="utf-8"))["per_question"]

sys_all  = np.array([r["sys"]["total"]    for r in pq])
b70_all  = np.array([r["b70"]["total"]    for r in pq])
raw_all  = np.array([r["raw_mean_total"]  for r in pq])

easy_idx   = [i for i, r in enumerate(pq) if r["difficulty"] == "easy"]
medium_idx = [i for i, r in enumerate(pq) if r["difficulty"] == "medium"]
hard_idx   = [i for i, r in enumerate(pq) if r["difficulty"] == "hard"]

# ── Palette ───────────────────────────────────────────────────────────────────
BG    = "#FFFFFF"; GRID  = "#E2E8F0"; TEXT  = "#0F172A"; SUB   = "#475569"
CSYS  = "#1D4ED8"; C70   = "#D97706"; CRAW  = "#64748B"; CGOOD = "#047857"

fig, ax = plt.subplots(figsize=(9, 6.5), facecolor=BG)
ax.set_facecolor(BG)
ax.tick_params(colors=TEXT, labelsize=10)
for sp in ["top","right"]:  ax.spines[sp].set_visible(False)
for sp in ["bottom","left"]: ax.spines[sp].set_color(TEXT)
ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
ax.set_axisbelow(True)

pair_labels = ["Overall\n(n=100)", "Easy\n(n=40)", "Medium\n(n=40)", "Hard\n(n=20)"]
sys_means = [
    float(np.mean(sys_all)),
    float(np.mean(sys_all[easy_idx])),
    float(np.mean(sys_all[medium_idx])),
    float(np.mean(sys_all[hard_idx])),
]
b70_means = [
    float(np.mean(b70_all)),
    float(np.mean(b70_all[easy_idx])),
    float(np.mean(b70_all[medium_idx])),
    float(np.mean(b70_all[hard_idx])),
]
raw_means = [
    float(np.mean(raw_all)),
    float(np.mean(raw_all[easy_idx])),
    float(np.mean(raw_all[medium_idx])),
    float(np.mean(raw_all[hard_idx])),
]

# 95% CI bootstrap helper
def _ci95(arr, n_boot=2000):
    np.random.seed(42)
    arr = np.array(arr)
    boots = [np.mean(np.random.choice(arr, len(arr), replace=True)) for _ in range(n_boot)]
    return np.percentile(boots, [2.5, 97.5])

sys_ci  = [_ci95(sys_all), _ci95(sys_all[easy_idx]), _ci95(sys_all[medium_idx]), _ci95(sys_all[hard_idx])]
b70_ci  = [_ci95(b70_all), _ci95(b70_all[easy_idx]), _ci95(b70_all[medium_idx]), _ci95(b70_all[hard_idx])]

width = 0.25
x = np.arange(len(pair_labels))

b_sys = ax.bar(x - width, sys_means, width, color=CSYS, zorder=3, edgecolor=BG, linewidth=0.5, label="Complete System (SYS)")
b_b70 = ax.bar(x,         b70_means, width, color=C70,   zorder=3, edgecolor=BG, linewidth=0.5, label="NVIDIA 70B (Baseline)")
b_raw = ax.bar(x + width, raw_means, width, color=CRAW,  zorder=3, edgecolor=BG, linewidth=0.5, label="Raw Qwen-0.5B (RAW)")

# 95% CI error bars
for i, (m, ci) in enumerate(zip(sys_means, sys_ci)):
    ax.errorbar(i - width, m, yerr=[[m-ci[0]], [ci[1]-m]], fmt="none", ecolor=TEXT, elinewidth=1.5, capsize=4, zorder=4)
for i, (m, ci) in enumerate(zip(b70_means, b70_ci)):
    ax.errorbar(i, m, yerr=[[m-ci[0]], [ci[1]-m]], fmt="none", ecolor=TEXT, elinewidth=1.5, capsize=4, zorder=4)

# Significance annotations
# T1 (Overall) = n.s., T3 (Easy) = *, T4 (Medium) = n.s., T5 (Hard) = **
sigs  = ["n.s.", "*", "n.s.", "**"]
y_top = 2.15

for i, sig in enumerate(sigs):
    if sig != "n.s.":
        ax.text(i - width/2, y_top, sig, ha="center", fontsize=13, fontweight="bold",
                color=CGOOD if "**" in sig else C70)
        ax.plot([i-width, i], [y_top-0.03, y_top-0.03], color=TEXT, lw=1.0, alpha=0.6)

# Value labels on top of bars
for bar in list(b_sys) + list(b_b70) + list(b_raw):
    v = bar.get_height()
    ax.text(bar.get_x()+bar.get_width()/2, v+0.02, f"{v:.3f}",
            ha="center", va="bottom", fontsize=8.5,
            color=bar.get_facecolor(), fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(pair_labels, color=TEXT, fontsize=10.5)
ax.set_ylim(0, 2.30)
ax.set_ylabel("Physics Score (0–4 scale)", color=TEXT, fontsize=11)
ax.set_title("Physics Validation Score by Difficulty stratum\n(error bars = 95% CI bootstrap, stars = Wilcoxon significance)",
             color=TEXT, fontsize=12, fontweight="bold", pad=12)
ax.legend(fontsize=9.5, facecolor=BG, edgecolor="none", labelcolor=TEXT, loc="upper right")

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(OUT_PNG), dpi=300, facecolor=BG, bbox_inches="tight")
print(f"Saved standlone plot to: {OUT_PNG}")
