# Graphical abstract: the better normalization reverses; NormBlend; result over 96 settings
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

# middle: method sketch
head(8.0, "NormBlend", "two views, one blend per variable", color=DARK["dual"])
m = fig.add_axes([7.7 / W, 0.15 / H, 7.0 / W, 6.5 / H]); m.set_xlim(0, 70); m.set_ylim(-2, 63); m.axis("off")
def rb(x, y, w, h, fc, ec, lines, fs=9.2):
    m.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=2.2", fc=fc, ec=ec, lw=1.0, zorder=2))
    for k, s_ in enumerate(lines):
        m.text(x + w / 2, y + h / 2 + (len(lines) - 1) * 1.9 - k * 3.8, s_, ha="center", va="center", fontsize=fs,
               color=INK, weight="bold" if k == 0 else "normal", zorder=3)
def ar(p, q, col=INK2, lw=1.0):
    m.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, color=col, lw=lw, shrinkA=0, shrinkB=0, zorder=1))
rng = np.random.default_rng(4); miss = rng.random((4, 8)) < 0.3; miss[1, 3:6] = True; miss[3, 5:8] = True; miss[0, 0] = False
def win(x0, yc, cell, gap, fill=None):
    h = 4 * (cell + gap) - gap
    for d in range(4):
        for t in range(8):
            fc, ec = ((fill, fill) if fill else ("white", "#a9a9a9")) if miss[d, t] else ("#c4c4c1", "#c4c4c1")
            m.add_patch(FancyBboxPatch((x0 + t * (cell + gap), yc - h / 2 + (3 - d) * (cell + gap)), cell, cell,
                                       boxstyle=f"round,pad=0,rounding_size={cell * 0.2}", fc=fc, ec=ec, lw=0.5, zorder=3))
    return 8 * (cell + gap) - gap
ww = win(0, 33, 1.4, 0.35)
rb(21, 45, 20, 12, TINTS["global"], GLOB, ["Imputer", "G view"])
rb(21, 9, 20, 12, TINTS["seq"], SEQ, ["Imputer", "S view"])
ar((ww + 0.8, 34.5), (21, 50.5)); ar((ww + 0.8, 31.5), (21, 15.5))
gx, gy, r = 51.5, 33, 4.0
m.add_patch(Polygon([(gx - r, gy), (gx, gy + r), (gx + r, gy), (gx, gy - r)], closed=True, fc="white", ec=INK2, lw=1.0, zorder=3))
m.text(gx, gy, r"$\kappa$", ha="center", va="center", fontsize=9.5, zorder=4)
ar((41, 51), (gx - 1.8, gy + 2.4), col=GLOB, lw=1.3); ar((41, 15), (gx - 1.8, gy - 2.4), col=SEQ, lw=1.3)
ar((gx + r, gy), (gx + r + 2.8, gy), col=DUAL, lw=1.4)
win(gx + r + 3.2, 33, 1.12, 0.28, fill=DUAL)
m.text(gx + 6.5, 26.3, "range guard", ha="center", va="center", fontsize=8.3, color=MUTED)
m.text(35, 4.2, r"$\hat x_d = a_d\,\hat x^{\mathrm{G}}_d + (1-a_d)\,\hat x^{\mathrm{S}}_d$", ha="center", va="center", fontsize=11, color=INK)
m.text(35, -1.2, r"$a_d$: per-variable weight from held-out errors", ha="center", va="center", fontsize=8.3, color=MUTED)

# right: result
head(14.5, "Six imputers, four benchmarks", "below the better one in 91 of 96 settings")
ax2 = fig.add_axes([16.9 / W, 1.9 / H, 3.9 / W, 4.2 / H])
labs = [("seq", "Per-seq."), ("global", "Global"), ("dual", "NormBlend")]
tb = D["teaserB"]
gms = {"dual": float(np.exp(np.mean(np.log([r["r"] for r in tb])))), "global": float(np.exp(np.mean(np.log([r["g"] for r in tb])))),
       "seq": float(np.exp(np.mean(np.log([r["s"] for r in tb]))))}
ax2.axvspan(0.9, 1.0, color=TINTS["dual"], zorder=0, lw=0)
for i, (k, lab) in enumerate(labs):
    ax2.plot([1, gms[k]], [i, i], color=COL[k], lw=2.4, zorder=3, solid_capstyle="round")
    ax2.scatter(gms[k], i, s=80, marker=MK[k], fc=COL[k], ec="white", lw=0.8, zorder=4)
    ax2.text(max(gms[k], 1) + 0.035, i, f"{gms[k]:.3f}", va="center", ha="left", fontsize=10.5, color=INK, weight="bold" if k == "dual" else "normal")
ax2.set_yticks(range(3)); ax2.set_yticklabels([l for _, l in labs]); ax2.tick_params(axis="y", length=0)
ax2.set_xlim(0.9, 1.66); ax2.set_ylim(2.6, -0.6); ax2.axvline(1, color=INK2, lw=0.9, zorder=2)
ax2.set_xticks([1.0, 1.2, 1.4]); ax2.xaxis.set_major_formatter(FuncFormatter(fmt)); grid(ax2, "x"); ax2.spines["left"].set_visible(False)
ax2.set_xlabel("RMSE / better normalization\n(geometric mean, 96 settings)", fontsize=9.5)
print(gms)
fig.savefig(os.path.join(OUT, "Graphical_abstract.png"), dpi=300, facecolor="white")
fig.savefig(os.path.join(OUT, "Graphical_abstract.pdf"), facecolor="white")
fig.savefig(os.path.join(OUT, "Graphical_abstract.tif"), dpi=300, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
