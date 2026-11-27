# Graphical abstract: the better normalization reverses; blend weights per variable; result over 96 settings
from jstyle import *
import json, numpy as np
from scipy.stats import gaussian_kde
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
setup()
D = json.load(open('figdata2.json'))
plt.rcParams.update({"font.size": 10.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "axes.labelsize": 10})
W, H = 21.2, 8.48
fig = plt.figure(figsize=(W * CM, H * CM))
fmt = lambda v, _: f"{v:g}"
def head(x, title, sub, color=INK):
    fig.text(x / W, (H - 0.5) / H, title, fontsize=12.5, weight="bold", color=color, va="top")
    fig.text(x / W, (H - 1.2) / H, sub, fontsize=10, color=INK2, va="top")

# left: reversal (raincloud)
head(0.4, "The better normalization reverses", "four benchmarks and a shift of conditions")
ax = fig.add_axes([1.55 / W, 1.95 / H, 5.6 / W, 4.35 / H])
order = ["CMAPSS", "CMAPSS_M", "MetroPT", "SKAB", "SC2MC"]
ax.axhspan(1, 30, color=TINTS["global"], zorder=0, lw=0); ax.axhspan(0.05, 1, color=TINTS["seq"], zorder=0, lw=0)
rng = np.random.default_rng(2)
for i, ds in enumerate(order):
    y = np.array([q["r"] for q in D["teaserA2"] if q["ds"] == ds]); lo = y < 1
    ly = np.log10(y); kde = gaussian_kde(ly, bw_method=0.45); yy = np.linspace(ly.min(), ly.max(), 200); d = 0.34 * kde(yy) / kde(yy).max(); yv = 10 ** yy
    for sel, col in ((yv >= 1, GLOB), (yv <= 1, SEQ)):
        if sel.any(): ax.fill_betweenx(yv[sel], i - 0.05 - d[sel], i - 0.05, color=col, alpha=0.3, lw=0, zorder=2)
    ax.plot(i - 0.05 - d, yv, color=INK2, lw=0.5, alpha=0.7, zorder=3)
    x = i + 0.05 + rng.uniform(0.02, 0.3, len(y))
    ax.scatter(x[lo], y[lo], s=11, fc=SEQ, ec="white", lw=0.35, zorder=5); ax.scatter(x[~lo], y[~lo], s=11, fc=GLOB, ec="white", lw=0.35, zorder=5)
ax.axhline(1, color=INK2, lw=0.9, zorder=4); ax.axvline(3.5, color="#bdbdb8", lw=0.7, ls=(0, (3, 2)), zorder=1)
ax.set_yscale("log"); ax.set_ylim(0.15, 12)
ax.yaxis.set_major_locator(FixedLocator([0.2, 0.5, 1, 2, 5])); ax.yaxis.set_major_formatter(FuncFormatter(fmt)); ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_xticks(range(5)); ax.set_xticklabels(["C-MAPSS-SC", "C-MAPSS-MC", "MetroPT-3", "SKAB", "SC → MC"], rotation=30, ha="right", rotation_mode="anchor")
ax.tick_params(axis="x", length=0); ax.set_xlim(-0.55, 4.55)
ax.set_ylabel("Per-sequence / global RMSE")
ax.text(-0.45, 8.6, "global better", ha="left", va="center", fontsize=9.5, color=DARK["global"], weight="bold")
ax.text(-0.45, 0.18, "per-sequence better", ha="left", va="center", fontsize=9.5, color=DARK["seq"], weight="bold")

# middle: per-variable weights (MetroPT-3)
from matplotlib.colors import LinearSegmentedColormap
head(8.0, "Blend weights per variable", "learned on held-out units, MetroPT-3")
cmap = LinearSegmentedColormap.from_list("ws", [SEQ, "#f6f6f4", GLOB])
BB = ["KNN", "BRITS", "SAITS", "TimesNet", "ImputeFormer", "Transformer"]
VN = ["TP2", "TP3", "H1", "DV press.", "Reservoirs", "Oil temp.", "Motor curr."]
M = np.array([[next(q["weight"] for q in D["weights"] if q["dataset"] == "MetroPT" and q["model"] == m and q["var"] == k) for k in range(7)] for m in BB])
axm = fig.add_axes([9.75 / W, 2.55 / H, 3.55 / W, 3.55 / H]); axm.set_xlim(0, 7); axm.set_ylim(6, 0); axm.axis("off")
for r in range(6):
    for c in range(7):
        axm.add_patch(FancyBboxPatch((c + 0.06, r + 0.06), 0.88, 0.88, boxstyle="round,pad=0,rounding_size=0.14", fc=cmap(M[r, c]), ec="none"))
    axm.text(-0.25, r + 0.5, BB[r], ha="right", va="center", fontsize=9.0, color=INK2)
for c, nm in enumerate(VN):
    axm.text(c + 0.5, 6.25, nm, rotation=90, ha="center", va="top", fontsize=8.6, color=INK2)
cax = fig.add_axes([13.5 / W, 2.55 / H, 0.2 / W, 3.55 / H])
cax.imshow(np.linspace(0, 1, 256)[:, None], aspect="auto", cmap=cmap, extent=(0, 1, 0, 1), origin="lower"); cax.set_xticks([])
cax.yaxis.tick_right(); cax.set_yticks([0, 1]); cax.set_yticklabels(["per-seq.", "global"], fontsize=8.6); cax.tick_params(length=0, pad=2)
for sp in cax.spines.values(): sp.set_visible(False)

# right: result
head(15.05, "Blending beats selection", "lower error in 91 of 96 settings")
ax2 = fig.add_axes([17.45 / W, 1.9 / H, 3.1 / W, 4.4 / H])
labs = [("dual", "NormBlend", "normblend"), ("dual", "Per-var. selection", "sel_var"), ("neutral", "Val. selection", "valsel"),
        ("global", "Global", "global"), ("seq", "Per-seq.", "seq")]
for i, (c, lab, k) in enumerate(labs):
    v = D["ga"][k]; col = COL.get(c, MUTED); mk = MK.get(c, "o")
    hollow = k == "sel_var"
    ax2.plot([1, v], [i, i], color=col, lw=2.2, zorder=3, solid_capstyle="round", alpha=0.55 if hollow else 1)
    ax2.scatter(v, i, s=70, marker=mk, fc="white" if hollow else col, ec=col if hollow else "white", lw=1.2 if hollow else 0.8, zorder=4)
    ax2.text(max(v, 1) + 0.035, i, f"{v:.3f}", va="center", ha="left", fontsize=10, color=INK, weight="bold" if k == "normblend" else "normal")
ax2.set_yticks(range(len(labs))); ax2.set_yticklabels([l for _, l, _ in labs], fontsize=9.2); ax2.tick_params(axis="y", length=0)
ax2.set_xlim(0.9, 1.5); ax2.set_ylim(len(labs) - 0.4, -0.6); ax2.axvline(1, color=INK2, lw=0.9, zorder=2)
ax2.set_xticks([1.0, 1.2, 1.4]); ax2.xaxis.set_major_formatter(FuncFormatter(fmt)); grid(ax2, "x"); ax2.spines["left"].set_visible(False)
ax2.set_xlabel("RMSE / better normalization\n(geometric mean, 96 settings)", fontsize=9.5)
gms = D["ga"]
print(gms)
fig.savefig(os.path.join(OUT, "Graphical_abstract.png"), dpi=300, facecolor="white")
fig.savefig(os.path.join(OUT, "Graphical_abstract.pdf"), facecolor="white")
fig.savefig(os.path.join(OUT, "Graphical_abstract.tif"), dpi=300, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
