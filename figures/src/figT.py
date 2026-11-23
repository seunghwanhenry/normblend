# Figure 1: (A) the better normalization reverses across datasets and under a shift (raincloud);
#           (B) NormBlend relative to the better normalization, per imputer (horizontal beeswarm)
from jstyle import *
import json, numpy as np
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
setup()
D = json.load(open("figdata2.json"))
order = ["CMAPSS", "CMAPSS_M", "MetroPT", "SKAB", "SC2MC"]
LAB = {"CMAPSS": "C-MAPSS-SC", "CMAPSS_M": "C-MAPSS-MC", "MetroPT": "MetroPT-3", "SKAB": "SKAB", "SC2MC": "SC → MC"}
BB = ["KNN", "BRITS", "SAITS", "TimesNet", "ImputeFormer", "Transformer"]
DSM = {"CMAPSS": ("o", "#2a78d6"), "CMAPSS_M": ("D", "#0b3d91"), "MetroPT": ("^", "#7fb2ee"), "SKAB": ("s", "#8c8c8c")}
fmt = lambda v, _: f"{v:g}"

def half_violin(ax, i, y, w=0.34):
    ly = np.log10(y); kde = gaussian_kde(ly, bw_method=0.45)
    yy = np.linspace(ly.min(), ly.max(), 300); d = kde(yy); d = w * d / d.max()
    yv = 10 ** yy; x0 = i - 0.05
    for sel, col in ((yv >= 1, GLOB), (yv <= 1, SEQ)):
        if sel.any():
            ax.fill_betweenx(yv[sel], x0 - d[sel], x0, color=col, alpha=0.28, lw=0, zorder=2)
    ax.plot(x0 - d, yv, color=INK2, lw=0.45, alpha=0.7, zorder=3)
    md = np.median(y); dm = np.interp(np.log10(md), yy, d)
    ax.plot([x0 - dm, x0], [md, md], color=INK, lw=0.9, zorder=4, solid_capstyle="butt")

def draw_a(ax, solo):
    rng = np.random.default_rng(2)
    ax.axhspan(1, 20, color=TINTS["global"], zorder=0, lw=0); ax.axhspan(0.05, 1, color=TINTS["seq"], zorder=0, lw=0)
    grid(ax)
    for i, ds in enumerate(order):
        y = np.array([q["r"] for q in D["teaserA2"] if q["ds"] == ds]); lo = y < 1
        half_violin(ax, i, y)
        x = i + 0.03 + rng.uniform(0.02, 0.3, len(y))
        ax.scatter(x[lo], y[lo], s=9, marker="o", fc=SEQ, ec="white", lw=0.3, zorder=5)
        ax.scatter(x[~lo], y[~lo], s=9, marker="o", fc=GLOB, ec="white", lw=0.3, zorder=5)
        pill(ax, i, 1.035, f"{int(lo.sum())}/{len(y)}", fc="white", ec="#cfcfcb", size=6.3, transform=ax.get_xaxis_transform())
    ax.axhline(1, color=INK2, lw=0.8, zorder=4); ax.axvline(3.5, color="#bdbdb8", lw=0.6, ls=(0, (3, 2)), zorder=1)
    T = ax.get_xaxis_transform()
    for a, b, s in ((-0.35, 3.35, "in distribution"), (3.65, 4.35, "shift")):
        ax.plot([a, a, b, b], [1.125, 1.145, 1.145, 1.125], transform=T, color=MUTED, lw=0.5, clip_on=False)
        ax.text((a + b) / 2, 1.16, s, transform=T, ha="center", va="bottom", fontsize=6.5, color=MUTED, clip_on=False)
    ax.set_yscale("log"); ax.set_ylim(0.15, 9)
    ax.yaxis.set_major_locator(FixedLocator([0.2, 0.5, 1, 2, 5])); ax.yaxis.set_major_formatter(FuncFormatter(fmt))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(range(len(order))); ax.set_xticklabels([LAB[d] for d in order], rotation=30, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.55, len(order) - 0.45); ax.tick_params(axis="x", length=0)
    ax.set_ylabel("RMSE per-sequence / RMSE global")
    ax.text(-0.45, 7.2, "global better", ha="left", va="center", fontsize=6.6, color=DARK["global"], weight="bold")
    ax.text(-0.45, 0.168, "per-sequence better", ha="left", va="center", fontsize=6.6, color=DARK["seq"], weight="bold")

def draw_b(ax, solo):
    rng = np.random.default_rng(3)
    ax.axvspan(0.80, 1.0, color=TINTS["dual"], zorder=0, lw=0)
    grid(ax, "x")
    for i, m in enumerate(BB):
        allr = []
        for ds, (mk, col) in DSM.items():
            x = np.array([q["r"] for q in D["teaserB"] if q["model"] == m and q["ds"] == ds]); allr += list(x)
            ax.scatter(x, i + rng.uniform(-0.27, 0.27, len(x)), s=10, marker=mk, fc=col, ec="white", lw=0.3, zorder=4)
        g = float(np.exp(np.mean(np.log(allr))))
        ax.plot([g, g], [i - 0.36, i + 0.36], color=INK, lw=1.3, zorder=5, solid_capstyle="butt")
        if i < len(BB) - 1: ax.axhline(i + 0.5, color="#e3e3e0", lw=0.5, zorder=1)
    ax.axvline(1, color=INK2, lw=0.8, zorder=3); ax.axvline(1.02, color=INK2, lw=0.6, ls=(0, (1, 1.5)), zorder=3)
    ax.set_xlim(0.835, 1.035); ax.set_ylim(len(BB) - 0.5, -0.5)
    ax.set_yticks(range(len(BB))); ax.set_yticklabels(BB); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.xaxis.set_major_locator(FixedLocator([0.85, 0.9, 0.95, 1.0])); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.set_xlabel("RMSE ratio to the better normalization")
    ax.text(0.842, -0.95, "NormBlend better", ha="left", va="center", fontsize=6.6, color=DARK["dual"], weight="bold", clip_on=False)
    h = [Line2D([], [], ls="", marker=mk, ms=3.8, mfc=col, mec="white", mew=0.3, label=DSN[ds]) for ds, (mk, col) in DSM.items()]
    h.append(Line2D([], [], color=INK, lw=1.3, label="geometric mean"))
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.42, -0.2), ncol=3, fontsize=6.4, handletextpad=0.3, handlelength=1.4, columnspacing=1.0, borderaxespad=0)

panels("fig_1", [draw_a, draw_b], [8.4, 7.6], 6.6, [(1.35, 0.15, 1.45, 0.95), (1.75, 0.2, 1.95, 0.6)], gap=0.4)
