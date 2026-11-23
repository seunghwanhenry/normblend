from jstyle import *
import json, numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
setup()
D = json.load(open('figdata2.json'))
plt.rcParams.update({"font.size": 11, "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.labelsize": 10.5})
W, H = 21.2, 8.48
fig = plt.figure(figsize=(W*CM, H*CM))
def fmt(v, _): return f"{v:g}"
# left: reversal
ax = fig.add_axes([1.6/W, 2.0/H, 5.6/W, 4.4/H])
order = ["CMAPSS", "CMAPSS_M", "MetroPT", "SKAB", "SC2MC"]
rng = np.random.default_rng(2)
for i, ds in enumerate(order):
    y = np.array([q["r"] for q in D["teaserA2"] if q["ds"] == ds]); x = i + rng.uniform(-0.22, 0.22, len(y)); lo = y < 1
    ax.scatter(x[lo], y[lo], s=16, marker="^", fc=SEQ, ec="white", lw=0.4, zorder=3)
    ax.scatter(x[~lo], y[~lo], s=16, marker="s", fc=GLOB, ec="white", lw=0.4, zorder=3)
ax.axhline(1, color=INK2, lw=0.9); grid(ax)
ax.axvline(3.5, color=GRID, lw=0.8, zorder=1)
ax.set_yscale("log"); ax.set_ylim(0.12, 16)
ax.yaxis.set_major_locator(FixedLocator([0.2, 0.5, 1, 2, 5])); ax.yaxis.set_major_formatter(FuncFormatter(fmt)); ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_xticks(range(5)); ax.set_xticklabels(["C-MAPSS-SC", "C-MAPSS-MC", "MetroPT-3", "SKAB", "SC → MC"], rotation=30, ha="right", rotation_mode="anchor")
ax.tick_params(axis="x", length=0)
ax.set_ylabel("Per-sequence / global RMSE")
ax.text(-0.4, 11.0, "global better", ha="left", fontsize=9.5, color=INK2, va="center")
ax.text(4.45, 0.145, "per-sequence better", ha="right", fontsize=9.5, color=INK2, va="center")
fig.text(0.4/W, (H-0.55)/H, "The better normalization reverses", fontsize=12.5, weight="bold", color=INK, va="top")
fig.text(0.4/W, (H-1.25)/H, "four benchmarks and a shift of conditions", fontsize=10.5, color=INK2, va="top")
# middle: method
mx = 8.0
fig.text(mx/W, (H-0.55)/H, "NormBlend", fontsize=12.5, weight="bold", color=DUAL, va="top")
lines = [("Train any imputer twice,", INK, "normal"), ("globally (G) and per sequence (S)", INK2, "normal"),
         ("", INK, "normal"), ("Blend per variable", INK, "normal"), ("with closed-form, error-optimal weights", INK2, "normal")]
for k, (s, c, w) in enumerate(lines):
    fig.text(mx/W, (H-1.6-0.55*k)/H, s, fontsize=10, color=c, weight=w, va="top")
fig.text(mx/W, (H-4.9)/H, r"$\hat{x}_d = a_d\,\hat{x}^{\mathrm{G}}_d + (1-a_d)\,\hat{x}^{\mathrm{S}}_d$", fontsize=13, color=INK, va="top")
fig.text(mx/W, (H-6.0)/H, "Range guard:", fontsize=10, color=INK, va="top")
fig.text(mx/W, (H-6.55)/H, "out-of-range windows use S", fontsize=10, color=INK2, va="top")
# right: confirmation result
ax2 = fig.add_axes([16.7/W, 1.75/H, 4.1/W, 4.3/H])
labs = [("seq", "Per-seq."), ("global", "Global"), ("dual", "NormBlend")]
tb = D["teaserB"]
gms = {"dual": float(np.exp(np.mean(np.log([r["r"] for r in tb])))), "global": float(np.exp(np.mean(np.log([r["g"] for r in tb])))),
       "seq": float(np.exp(np.mean(np.log([r["s"] for r in tb]))))}
for i, (k, lab) in enumerate(labs):
    ax2.plot([1, gms[k]], [i, i], color=COL[k], lw=2.2, zorder=3, solid_capstyle="butt")
    ax2.scatter(gms[k], i, s=70, marker=MK[k], fc=COL[k], ec="white", lw=0.6, zorder=4)
    ax2.text(max(gms[k], 1) + 0.035, i, f"{gms[k]:.3f}", va="center", ha="left", fontsize=10.5, color=INK)
ax2.set_yticks(range(3)); ax2.set_yticklabels([l for _, l in labs]); ax2.tick_params(axis="y", length=0)
ax2.set_xlim(0.92, 1.66); ax2.set_ylim(2.6, -0.6); ax2.axvline(1, color=INK2, lw=0.9, ls=(0, (3, 2)), zorder=4)
ax2.set_xticks([1.0, 1.2, 1.4]); ax2.xaxis.set_major_formatter(FuncFormatter(fmt)); grid(ax2, "x"); ax2.spines["left"].set_visible(False)
ax2.set_xlabel("RMSE / better normalization\n(geometric mean, 96 settings)", fontsize=9.5)
fig.text(14.3/W, (H-0.55)/H, "Six imputers, four benchmarks", fontsize=12.5, weight="bold", color=INK, va="top")
fig.text(14.3/W, (H-1.25)/H, "below the better one in 91 of 96 settings", fontsize=10.5, color=INK2, va="top")
print(gms)
fig.savefig(os.path.join(OUT, "Graphical_abstract.png"), dpi=300, facecolor="white")
fig.savefig(os.path.join(OUT, "Graphical_abstract.pdf"), facecolor="white")
fig.savefig(os.path.join(OUT, "Graphical_abstract.tif"), dpi=300, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
