# Figure 1 (teaser): (A) the better normalization reverses across datasets and under a shift; (B) NormBlend relative to the better one
from jstyle import *
import json, numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
setup()
D = json.load(open("figdata2.json"))
order = ["CMAPSS", "CMAPSS_M", "MetroPT", "SKAB", "SC2MC"]
LAB = {"CMAPSS": "C-MAPSS-SC", "CMAPSS_M": "C-MAPSS-MC", "MetroPT": "MetroPT-3", "SKAB": "SKAB", "SC2MC": "SC → MC"}
BB = ["KNN", "BRITS", "SAITS", "TimesNet", "ImputeFormer", "Transformer"]
DSM = {"CMAPSS": ("o", "#2a78d6"), "CMAPSS_M": ("D", "#0b3d91"), "MetroPT": ("^", "#7fb2ee"), "SKAB": ("s", "#6b6b6b")}
def fmt(v, _): return f"{v:g}"

def draw_a(ax, solo):
    rng = np.random.default_rng(2); grid(ax)
    for i, ds in enumerate(order):
        y = np.array([q["r"] for q in D["teaserA2"] if q["ds"] == ds]); x = i + rng.uniform(-0.24, 0.24, len(y)); lo = y < 1
        ax.scatter(x[lo], y[lo], s=13, marker="^", facecolor=SEQ, edgecolor="white", linewidth=0.35, zorder=3)
        ax.scatter(x[~lo], y[~lo], s=13, marker="s", facecolor=GLOB, edgecolor="white", linewidth=0.35, zorder=3)
        ax.text(i, 1.015, f"{int(lo.sum())}/{len(y)}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.8, color=INK, clip_on=False)
    ax.axhline(1, color=INK2, lw=0.8, zorder=2); ax.axvline(3.5, color=GRID, lw=0.8, zorder=1)
    ax.set_yscale("log"); ax.set_ylim(0.15, 9)
    ax.yaxis.set_major_locator(FixedLocator([0.2, 0.5, 1, 2, 5])); ax.yaxis.set_major_formatter(FuncFormatter(fmt))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(range(len(order))); ax.set_xticklabels([LAB[d] for d in order], rotation=30, ha="right"); ax.set_xlim(-0.6, len(order) - 0.4)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("RMSE per-sequence / RMSE global")
    ax.text(-0.45, 6.5, "global better", ha="left", va="center", fontsize=6.8, color=INK2)
    ax.text(len(order) - 0.45, 0.17, "per-sequence better", ha="right", va="center", fontsize=6.8, color=INK2)
    ax.text(1.5, 1.075, "in-distribution", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.8, color=MUTED)
    ax.text(4.0, 1.075, "shift", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.8, color=MUTED)

def draw_b(ax, solo):
    rng = np.random.default_rng(3); grid(ax)
    for i, m in enumerate(BB):
        for ds, (mk, col) in DSM.items():
            y = np.array([q["r"] for q in D["teaserB"] if q["model"] == m and q["ds"] == ds])
            ax.scatter(i + rng.uniform(-0.22, 0.22, len(y)), y, s=14, marker=mk, facecolor=col, edgecolor="white", linewidth=0.35, zorder=3)
    ax.axhline(1, color=INK2, lw=0.8, zorder=2); ax.axhline(1.02, color=INK2, lw=0.5, ls=(0, (1, 1.5)), zorder=2)
    ax.set_ylim(0.84, 1.04); ax.set_xticks(range(len(BB))); ax.set_xticklabels(BB, rotation=30, ha="right"); ax.set_xlim(-0.6, len(BB) - 0.4)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("RMSE ratio to better normalization")
    h = [Line2D([], [], ls="", marker=mk, ms=4.2, mfc=col, mec="white", mew=0.35, label=DSN[ds]) for ds, (mk, col) in DSM.items()]
    ax.legend(handles=h, loc="lower right", fontsize=6.6, handletextpad=0.2, borderaxespad=0.2)

panels("fig_1", [draw_a, draw_b], [8.4, 7.4], 6.4, [(1.35, 0.15, 1.45, 0.8), (1.35, 0.15, 1.45, 0.8)], gap=0.4)
