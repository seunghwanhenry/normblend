# Figure 3: shift magnitude. RMSE relative to per-sequence normalization and guard activation against the offset k
from jstyle import *
import json, numpy as np
from matplotlib.lines import Line2D
setup()
Q = json.load(open("figdata2.json"))["shiftk"]
W, H = 17.0, 11.2
fig = plt.figure(figsize=(W * CM, H * CM))
DS = [("CMAPSS", "C-MAPSS-SC"), ("MetroPT", "MetroPT-3")]
SC = [("one", "offset in one variable per window"), ("all", "offset in every variable")]
X0 = 0.3                                                     # plotting position of k = 0
LINES = [("global", dict(color=GLOB, marker="s", ms=3.2, lw=1.0, label="Global standardization")),
         ("nb_noguard", dict(color=DUAL, marker="o", ms=3.0, mfc="white", lw=1.0, ls=(0, (3, 1.6)), label="NormBlend without guard")),
         ("nb_k0.25", dict(color=DARK["dual"], marker="D", ms=2.8, lw=1.0, ls=(0, (1, 1.2)), label="NormBlend, κ = 0.25")),
         ("normblend", dict(color=DUAL, marker="o", ms=3.4, lw=1.5, label="NormBlend (κ = 1)"))]
pw, ph, gh, l0, b0, gx, gy = 6.55, 2.6, 0.85, 1.6, 1.05, 1.15, 1.05
YMAX = 12


def xs(k):
    k = np.asarray(k, float); return np.where(k == 0, X0, k)


def xaxis(ax, labels):
    ax.set_xscale("log"); ax.set_xlim(0.25, 130)
    t = [X0, 0.5, 1, 2, 5, 10, 20, 50, 100]
    ax.set_xticks(t); ax.set_xticklabels(["0", "0.5", "1", "2", "5", "10", "20", "50", "100"] if labels else []); ax.minorticks_off()
    ax.axvspan(0.36, 0.42, color="white", zorder=6, lw=0)                         # axis break between 0 and 0.5
    for xx in (0.36, 0.42):
        ax.plot([xx, xx], [0, 0.035], transform=ax.get_xaxis_transform(), color=SPINE, lw=0.6, zorder=7, clip_on=False)


for r, (ds, dsn) in enumerate(DS):
    for c, (sc, scn) in enumerate(SC):
        q = sorted([x for x in Q if x["dataset"] == ds and x["scenario"] == sc], key=lambda x: x["k"])
        k = xs([x["k"] for x in q])
        x0 = l0 + c * (pw + gx); yb = b0 + (1 - r) * (ph + gh + gy + 0.25)
        ax = fig.add_axes([x0 / W, yb / H, pw / W, ph / H]); grid(ax)
        ax.axhline(1, color=SEQ, lw=1.4, zorder=2)
        for v, kw in LINES:
            y = np.array([x[v] for x in q]); out = np.flatnonzero(y > YMAX)
            yc = y.copy()
            if len(out): yc[out[0]] = YMAX; yc[out[0] + 1:] = np.nan
            ax.plot(k, yc, mec="white" if kw.get("mfc") != "white" else kw["color"], mew=0.5 if kw.get("mfc") != "white" else 0.9, zorder=5 if v == "normblend" else 4, **{**kw, "label": None})
            if len(out): ax.annotate("", xy=(k[out[0]] * 1.25, YMAX * 1.0), xytext=(k[out[0]], YMAX * 0.97), arrowprops=dict(arrowstyle="-|>", color=kw["color"], lw=0.8, mutation_scale=6), annotation_clip=False)
        ax.set_yscale("log"); ax.set_ylim(0.62, YMAX); ax.set_yticks([0.7, 1, 1.5, 2, 3, 5, 10]); ax.set_yticklabels(["0.7", "1", "1.5", "2", "3", "5", "10"]); ax.minorticks_off()
        xaxis(ax, True)
        if c == 0: ax.set_ylabel("RMSE relative to\nper-sequence", fontsize=7.6, linespacing=1.0)
        ga = fig.add_axes([x0 / W, (yb + ph + 0.32) / H, pw / W, gh / H]); grid(ga)
        ga.plot(k, [x["guard1"] for x in q], color=DUAL, lw=1.3, marker="o", ms=2.4, mec="white", mew=0.4, zorder=4)
        ga.plot(k, [x["guard025"] for x in q], color=DARK["dual"], lw=1.0, ls=(0, (1, 1.2)), marker="D", ms=2.2, zorder=3)
        ga.set_ylim(-0.04, 1.08); ga.set_yticks([0, 1]); ga.set_yticklabels(["0", "1"]); xaxis(ga, False)
        ga.spines["bottom"].set_visible(False); ga.tick_params(axis="x", length=0)
        if c == 0: ga.set_ylabel("Flagged\nshare", fontsize=7.6, linespacing=1.0, labelpad=11.5)
        ga.set_title(f"{dsn}, {scn}", fontsize=7.4, color=INK, pad=3, weight="bold")
fig.text((l0 + pw + gx / 2) / W, 0.12 / H, "Offset k (standard deviations of the observed training values)", ha="center", fontsize=8.2, color=INK)
h = [Line2D([], [], color=SEQ, lw=1.4, label="Per-sequence normalization")]
h += [Line2D([], [], mec="white" if kw.get("mfc") != "white" else kw["color"], mew=0.5 if kw.get("mfc") != "white" else 0.9, **kw) for _, kw in LINES]
fig.legend(handles=h, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.005), fontsize=6.9, frameon=False, handlelength=2.4, columnspacing=1.6)
save(fig, "fig_3")
