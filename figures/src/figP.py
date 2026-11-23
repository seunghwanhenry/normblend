# Figure 3: Proposition 1 in practice - calibration-predicted gain of blending against the test gain, per variable
from jstyle import *
import json, numpy as np
from scipy.stats import spearmanr
setup()
D = json.load(open("figdata2.json"))
q = np.array([r["q"] for r in D["prop"]]); pred = 100 * np.array([r["pred"] for r in D["prop"]]); test = 100 * np.array([r["test"] for r in D["prop"]])
bins = [-np.inf, 0.25, 0.5, 0.75, 0.9, 1.0, np.inf]; labs = ["≤0.25", "0.25–0.5", "0.5–0.75", "0.75–0.9", "0.9–1", ">1"]
k = np.digitize(q, bins[1:-1], right=True)

def draw_a(ax, solo):
    grid(ax)
    for i in range(len(labs)):
        t, p = test[k == i], pred[k == i]
        lo, md, hi = np.percentile(t, [25, 50, 75])
        ax.bar(i, md, width=0.62, color=DUAL, alpha=0.85, zorder=3)
        ax.plot([i, i], [lo, hi], color=INK, lw=0.9, zorder=4)
        ax.scatter([i], [np.median(p)], marker="_", s=120, color=GLOB, linewidths=1.6, zorder=5)
        ax.text(i, 1.0, f"n={len(t)}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.4, color=MUTED, clip_on=False)
    ax.axhline(0, color=INK2, lw=0.7)
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, fontsize=6.9)
    ax.set_xlabel(r"$\rho_d \,/\, \sqrt{E_{\min}/E_{\max}}$ (calibration)")
    ax.set_ylabel("Test MSE reduction (%)")
    ax.set_ylim(-4, 34)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=DUAL, alpha=0.85, label="test, median (IQR)"), Line2D([], [], color=GLOB, lw=1.6, label="predicted on calibration")],
              loc="upper right", fontsize=6.6, borderaxespad=0.2)

def draw_b(ax, solo):
    grid(ax, "both")
    m = (pred > 0)
    ax.hexbin(pred[m], test[m], gridsize=34, extent=(0, 40, -20, 40), cmap="Blues", mincnt=1, bins="log", linewidths=0.1, zorder=2)
    ax.plot([0, 40], [0, 40], color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=3)
    ax.axhline(0, color=INK2, lw=0.6, zorder=3)
    r = spearmanr(pred, test).statistic
    ax.text(0.97, 0.04, f"Spearman {r:.2f}\n{m.sum()} of {len(m)} variable-settings\nwith an interior weight", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.6, color=INK, bbox=dict(fc="white", ec="none", pad=1.5))
    ax.set_xlim(0, 40); ax.set_ylim(-20, 40)
    ax.set_xlabel("Predicted MSE reduction on calibration (%)"); ax.set_ylabel("Test MSE reduction (%)")

panels("fig_3", [draw_a, draw_b], [8.2, 7.4], 6.6, [(1.35, 0.2, 1.25, 0.45), (1.35, 0.2, 1.25, 0.25)], gap=0.4)
