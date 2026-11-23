# Figure 4: missing-rate sweep (MCAR 10-90%), RMSE of global, per-sequence and NormBlend
from jstyle import *
import json, numpy as np
from matplotlib.lines import Line2D
setup()
D = json.load(open("figdata2.json"))
rates = [0.1, 0.3, 0.5, 0.7, 0.9]; DS = ["CMAPSS", "MetroPT"]; MD = ["KNN", "SAITS", "ImputeFormer"]
W, H = 17.0, 9.6
fig = plt.figure(figsize=(W * CM, H * CM))
pw, ph, l0, b0, gx, gy = 4.55, 3.15, 1.35, 1.05, 0.95, 0.95
for r, ds in enumerate(DS):
    for c, m in enumerate(MD):
        ax = fig.add_axes([(l0 + c * (pw + gx)) / W, (b0 + (1 - r) * (ph + gy)) / H, pw / W, ph / H]); grid(ax)
        for v in ("global", "seq", "dual"):
            key = "normblend" if v == "dual" else v
            y = [next(q["mean"] for q in D["sweep"] if q["dataset"] == ds and q["model"] == m and q["variant"] == key and q["cond"] == f"MCAR {x:g}") for x in rates]
            ax.plot(rates, y, color=COL[v], marker=MK[v], ms=3.6, lw=1.1 if v == "dual" else 0.9, mec="white", mew=0.4, zorder=4 if v == "dual" else 3)
        ax.set_xticks(rates); ax.set_xticklabels([f"{int(100 * x)}" for x in rates])
        ax.set_title(f"{DSN[ds]}, {m}", fontsize=7.6, pad=3, color=INK)
        if r == 1: ax.set_xlabel("Missing rate, MCAR (%)")
        if c == 0: ax.set_ylabel("RMSE")
h = [Line2D([], [], color=COL[v], marker=MK[v], ms=4, mec="white", mew=0.4, lw=1, label=LBL[v]) for v in ("global", "seq", "dual")]
fig.legend(handles=h, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), fontsize=7.4, frameon=False)
save(fig, "fig_5")
