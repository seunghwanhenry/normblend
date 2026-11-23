# Figure 5: missing-rate sweep (MCAR 10-90%), RMSE of global, per-sequence and NormBlend (means over seeds)
from jstyle import *
import json, numpy as np
from matplotlib.lines import Line2D
setup()
D = json.load(open("figdata2.json"))
rates = [0.1, 0.3, 0.5, 0.7, 0.9]; DS = ["CMAPSS", "MetroPT"]; MD = ["KNN", "SAITS", "ImputeFormer"]
W, H = 17.0, 8.95
fig = plt.figure(figsize=(W * CM, H * CM))
pw, ph, l0, b0, gx, gy = 4.45, 3.05, 1.75, 1.0, 0.85, 0.75
def get(ds, m, v, f):
    return np.array([next(q[f] for q in D["sweep"] if q["dataset"] == ds and q["model"] == m and q["variant"] == v and q["cond"] == f"MCAR {x:g}") for x in rates])
for r, ds in enumerate(DS):
    for c, m in enumerate(MD):
        ax = fig.add_axes([(l0 + c * (pw + gx)) / W, (b0 + (1 - r) * (ph + gy)) / H, pw / W, ph / H]); grid(ax)
        x = 100 * np.array(rates)
        for v in ("global", "seq", "dual"):
            key = "normblend" if v == "dual" else v
            mu = get(ds, m, key, "mean")
            ax.plot(x, mu, color=COL[v], marker=MK[v], ms=3.6 if v != "dual" else 4.0, lw=1.0 if v != "dual" else 1.5,
                    mec="white", mew=0.5, zorder=4 if v == "dual" else 3)
        ax.set_xticks(x); ax.set_xticklabels([f"{int(t)}" for t in x]); ax.set_xlim(4, 96)
        if r == 0: ax.set_title(m, fontsize=7.6, weight="bold", color=INK, pad=5)
        if c == 0:
            fig.text(0.28 / W, (b0 + (1 - r) * (ph + gy) + ph / 2) / H, DSN[ds], rotation=90, ha="center", va="center", fontsize=7.6, weight="bold", color=INK)
        if r == 1: ax.set_xlabel("Missing rate, MCAR (%)")
        if c == 0: ax.set_ylabel("RMSE")
h = [Line2D([], [], color=COL[v], marker=MK[v], ms=4, mec="white", mew=0.5, lw=1.5 if v == "dual" else 1.0, label=LBL[v]) for v in ("global", "seq", "dual")]
fig.legend(handles=h, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), fontsize=7.0, frameon=False, handlelength=1.8, columnspacing=1.4)
save(fig, "fig_5")
