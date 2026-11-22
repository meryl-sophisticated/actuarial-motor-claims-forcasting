import json, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
root = sys.argv[1] if len(sys.argv) > 1 else "."
SURF, TXT, TXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": TXT, "axes.labelcolor": TXT2,
                     "xtick.color": TXT2, "ytick.color": TXT2, "axes.edgecolor": GRID, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF})
def clean(ax, grid_axis="y"):
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)

# 1. claim frequency: improvement over the flat-rate model
f = json.load(open(f"{root}/claim_frequency_results.json"))
m = f["models"]; h = m["Homogeneous"]["out_sample"]
rows = [("Poisson GLM (benchmark)", "Poisson GLM"), ("Regression tree, 1-SE rule", "Regression tree (1-SE rule)"),
        ("Regression tree, min-CV", "Regression tree (min-CV pruned)"),
        ("Poisson boosting, M=50, J=3", "Boosting, assignment setting (M=50, J=3, no shrinkage)"),
        ("Poisson boosting, tuned", "Boosting, tuned (shrinkage + early stopping)")]
gain = [h - m[k]["out_sample"] for _, k in rows]
fig, ax = plt.subplots(figsize=(7.5, 3.4))
y = np.arange(len(rows))[::-1]
ax.barh(y, gain, height=0.55, color=[GRAY] + [BLUE] * 4)
for yi, g in zip(y, gain):
    ax.text(g + 0.04, yi, f"{g:.2f}", va="center", color=TXT, fontsize=9.5)
ax.set_yticks(y, [r[0] for r in rows]); ax.set_xlim(0, 3.4); clean(ax, "x")
ax.set_xlabel("Reduction in out-of-sample Poisson deviance vs. flat-rate model (×10⁻²)")
fig.suptitle("Claim frequency: improvement over a flat-rate model, test set", x=0.02, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig(f"{root}/claim_frequency_gain.png", dpi=160); plt.close(fig)
