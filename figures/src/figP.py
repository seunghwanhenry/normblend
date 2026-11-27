# Figure 4: Proposition 1 on the benchmark data
from jstyle import *
import json, numpy as np
from scipy.stats import spearmanr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
setup()
D = json.load(open("figdata2.json"))
q = np.array([r["q"] for r in D["prop"]]); pred = 100 * np.array([r["pred"] for r in D["prop"]]); test = 100 * np.array([r["test"] for r in D["prop"]])
bins = [-np.inf, 0.25, 0.5, 0.75, 0.9, 1.0, np.inf]; labs = ["≤0.25", "0.25–0.5", "0.5–0.75", "0.75–0.9", "0.9–1", ">1"]
k = np.digitize(q, bins[1:-1], right=True)

def draw_a(ax, solo):
    ax.axvspan(4.5, 5.5, color=TINTS["neutral"], zorder=0, lw=0)
    grid(ax)
    pm = []
    for i in range(len(labs)):
        t, p = test[k == i], pred[k == i]
        lo, md, hi = np.percentile(t, [25, 50, 75])
        ax.bar(i, md, width=0.6, color=DUAL, alpha=0.22, ec=DUAL, lw=0.8, zorder=3)
        ax.plot([i, i], [lo, hi], color=DUAL, lw=2.2, alpha=0.55, zorder=4, solid_capstyle="round")
        ax.scatter([i], [md], s=16, fc="white", ec=DUAL, lw=1.0, zorder=5)
        pm.append(np.median(p))
        ax.text(i, -0.9, f"n = {len(t):,}", ha="center", va="top", fontsize=6.0, color=MUTED, zorder=6)
    ax.plot(range(len(labs)), pm, color=GLOB, lw=0.8, ls=(0, (2.5, 1.8)), zorder=5)
    ax.scatter(range(len(labs)), pm, marker="D", s=14, fc=GLOB, ec="white", lw=0.4, zorder=6)
    ax.axhline(0, color=INK2, lw=0.7, zorder=4)
    ax.text(5, 30.5, "weight\n0 or 1", ha="center", va="top", fontsize=6.0, color=MUTED)
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, fontsize=6.8); ax.tick_params(axis="x", length=0)
    ax.set_xlabel(r"$\rho_d \,/\, \sqrt{E_{\min}/E_{\max}}$ on calibration data")
    ax.set_ylabel("Test MSE reduction (%)")
    ax.set_ylim(-4.5, 32); ax.set_xlim(-0.55, 5.55)
    ax.legend(handles=[Patch(fc=DUAL, alpha=0.3, ec=DUAL, label="test: median and IQR"),
                       Line2D([], [], color=GLOB, lw=0.8, ls=(0, (2.5, 1.8)), marker="D", ms=3.5, mec="white", mew=0.4, label="predicted on calibration")],
              loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=6.4, borderaxespad=0.2, handlelength=1.8)

def draw_b(ax, solo):
    ax.axis("off")
    m = pred > 0
    main = ax.inset_axes([0, 0, 0.79, 0.79]); top = ax.inset_axes([0, 0.82, 0.79, 0.18], sharex=main); right = ax.inset_axes([0.82, 0, 0.18, 0.79], sharey=main)
    grid(main, "both")
    main.hexbin(pred[m], test[m], gridsize=30, extent=(0, 40, -20, 40), cmap="Blues", mincnt=1, bins="log", linewidths=0.15, edgecolors="white", zorder=2)
    main.plot([0, 40], [0, 40], color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=3)
    main.axhline(0, color=INK2, lw=0.6, zorder=3)
    main.set_xlim(0, 40); main.set_ylim(-20, 40)
    main.set_xlabel("Predicted MSE reduction (%)"); main.set_ylabel("Test MSE reduction (%)")
    r = spearmanr(pred, test).statistic
    pill(main, 38.5, -17.5, f"Spearman {r:.2f} (all {len(m):,} blends)\n{m.sum():,} interior blends shown", fc="white", ec="#d4d4cf", size=6.0, ha="right", va="bottom")
    hb = dict(color=DUAL, alpha=0.35, ec=DUAL, lw=0.5)
    top.hist(pred[m], bins=np.linspace(0, 40, 31), **hb); right.hist(test[m], bins=np.linspace(-20, 40, 31), orientation="horizontal", **hb)
    for a in (top, right):
        a.tick_params(labelbottom=False, labelleft=False, length=0)
        for sp in a.spines.values(): sp.set_visible(False)
        a.set_xticks([]) if a is right else a.set_yticks([])
    top.spines["bottom"].set_visible(True); right.spines["left"].set_visible(True)
    top.tick_params(axis="x", labelbottom=False); right.tick_params(axis="y", labelleft=False)

panels("fig_4", [draw_a, draw_b], [8.2, 7.4], 6.6, [(1.35, 0.2, 1.45, 0.25), (1.35, 0.2, 1.25, 0.2)], gap=0.4)
