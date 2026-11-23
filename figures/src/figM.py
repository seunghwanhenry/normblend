# Figure 2: NormBlend schematic (dual-view training, calibration weights, range guard, per-variable blend)
from jstyle import *
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon
setup()
W, H = 192, 84
fig = plt.figure(figsize=(W / 10 * CM, H / 10 * CM))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
T1, T2, T3 = 8.0, 7.4, 7.0

def rbox(x0, y0, w, h, fc, ec, lw=0.8, r=2.0, z=2, ls="-"):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))

def box(x0, y0, w, h, fc, ec, title, lines=(), eq=None, gap=3.5):
    rbox(x0, y0, w, h, fc, ec)
    titles = [title] if isinstance(title, str) else list(title)
    rows = [(t, True) for t in titles] + [(s, False) for s in lines]
    n = len(rows) - 1 + (1 if eq else 0); top = y0 + h / 2 + gap * n / 2
    for i, (s, bold) in enumerate(rows):
        ax.text(x0 + w / 2, top - gap * i, s, ha="center", va="center", fontsize=T1 if bold else T2,
                weight="bold" if bold else "normal", color=INK if bold else INK2, zorder=4)
    if eq:
        pill(ax, x0 + w / 2, top - gap * len(rows) - 0.4, eq, fc="white", ec="#d4d4cf", size=T3, color=MUTED, pad=0.22)

def arrow(pts, color=INK2, lw=0.8, ls="-"):
    xs, ys = zip(*pts)
    if len(pts) > 2: ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, solid_joinstyle="round", solid_capstyle="round", zorder=1)
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=6.5, color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0, zorder=1))

rng = np.random.default_rng(4)
TT, DD = 8, 4
miss = rng.random((DD, TT)) < 0.3; miss[1, 3:6] = True; miss[3, 5:8] = True; miss[0, 0] = False
def window(x0, yc, cell, gap, fill=None):
    w = TT * (cell + gap) - gap; h = DD * (cell + gap) - gap; y0 = yc - h / 2
    for d in range(DD):
        for t in range(TT):
            x = x0 + t * (cell + gap); y = y0 + (DD - 1 - d) * (cell + gap)
            if miss[d, t]:
                fc, ec = (fill, fill) if fill else ("white", "#a9a9a9")
            else:
                fc, ec = "#c4c4c1", "#c4c4c1"
            ax.add_patch(FancyBboxPatch((x, y), cell, cell, boxstyle=f"round,pad=0,rounding_size={cell * 0.18}", fc=fc, ec=ec, lw=0.4, zorder=3))
    return w, h

# lanes
rbox(64, 56, 77, 25, TINTS["global"], "#f3c9b3", lw=0.6, r=3, z=0)
rbox(64, 3, 77, 25, TINTS["seq"], "#b5e3cf", lw=0.6, r=3, z=0)
pill(ax, 75, 81, "G   global view", fc="white", ec=GLOB, color=DARK["global"], size=T3, weight="bold")
pill(ax, 79, 28, "S   per-sequence view", fc="white", ec=SEQ, color=DARK["seq"], size=T3, weight="bold")

# input window
gw, gh = window(3, 42, 2.6, 0.5)
ax.text(3 + gw / 2, 54, "Incomplete window", ha="center", va="center", fontsize=T1, weight="bold", color=INK)
ax.text(3 + gw / 2, 50.3, r"values $x_{td}$, mask $v_{td}$", ha="center", va="center", fontsize=T2, color=INK2)
ax.annotate("", (3 + gw, 33.6), (3, 33.6), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.5, mutation_scale=5))
ax.text(3 + gw, 31.6, "time $t$", ha="right", va="center", fontsize=T3, color=MUTED)

# global standardization (shared)
box(31, 31, 27, 22, "white", "#9b9b9b", ["Global", "standardization"], [r"$z_{td}=(x_{td}-\mu_d)/\sigma_d$"], eq="Eq. 3", gap=3.7)
arrow([(3 + gw + 0.8, 42), (31, 42)])
ax.plot([58, 62], [42, 42], color=INK2, lw=0.8, zorder=1)
ax.plot([62, 62], [15.5, 68.5], color=INK2, lw=0.8, zorder=1, solid_capstyle="round")
arrow([(62, 68.5), (96, 68.5)]); arrow([(62, 15.5), (67, 15.5)])

# G lane
ax.text(79, 73.0, "same imputer,", ha="center", va="center", fontsize=T3, color=MUTED, style="italic")
ax.text(79, 70.6, "same seed", ha="center", va="center", fontsize=T3, color=MUTED, style="italic")
box(96, 61, 20, 15, "white", GLOB, r"Imputer $f_{\mathrm{G}}$", ["trained on $z$"], gap=3.6)
arrow([(116, 68.5), (120, 68.5)])
window(120.5, 68.5, 1.9, 0.4, fill=GLOB)
ax.text(129.4, 75.5, r"$\hat z^{\mathrm{G}}$", ha="center", va="center", fontsize=T1, color=DARK["global"])

# S lane
box(67, 5, 25, 20, "white", SEQ, ["Per-sequence", "normalization"], [r"$\tilde z_{td}=(z_{td}-m_d)/s_d$"], eq="Eq. 5", gap=3.7)
arrow([(92, 15.5), (96, 15.5)])
box(96, 8, 20, 15, "white", SEQ, r"Imputer $f_{\mathrm{S}}$", [r"trained on $\tilde z$"], gap=3.6)
arrow([(116, 15.5), (120, 15.5)])
window(120.5, 15.5, 1.9, 0.4, fill=SEQ)
ax.text(129.4, 22.6, r"$\hat z^{\mathrm{S}}$", ha="center", va="center", fontsize=T1, color=DARK["seq"])

# calibration weights
box(84, 33, 28, 18, "white", "#9b9b9b", "Calibration", [r"$E_{\mathrm{G}},\ E_{\mathrm{S}},\ C$"], eq="Eq. 10", gap=3.9)
arrow([(106, 61), (106, 51)], ls=(0, (2, 1.4)), lw=0.7); arrow([(106, 23), (106, 33)], ls=(0, (2, 1.4)), lw=0.7)
ax.text(104.7, 53.6, "held-out errors", ha="right", va="center", fontsize=T3, color=MUTED)
ax.text(107.3, 30.4, "held-out errors", ha="left", va="center", fontsize=T3, color=MUTED)
# weight glyph a_d
arrow([(112, 42), (115, 42)])
ax.text(121, 34.6, r"$a_d$", ha="center", va="center", fontsize=T1, color=INK)
for j, a in enumerate([0.82, 0.35, 0.62, 0.18]):
    y = 45.5 - j * 2.6
    ax.add_patch(Rectangle((116, y), 10 * a, 1.8, fc=GLOB, ec="white", lw=0.3, zorder=3))
    ax.add_patch(Rectangle((116 + 10 * a, y), 10 * (1 - a), 1.8, fc=SEQ, ec="white", lw=0.3, zorder=3))
arrow([(126.5, 42), (128.5, 42)])
# range guard
gx, gy, r = 134.5, 42, 6.0
ax.add_patch(Polygon([(gx - r, gy), (gx, gy + r), (gx + r, gy), (gx, gy - r)], closed=True, fc="white", ec=INK2, lw=0.8, zorder=3))
ax.text(gx, gy, r"$\kappa$", ha="center", va="center", fontsize=T1, color=INK, zorder=4)
ax.text(gx - 3.0, 53.2, "range guard", ha="center", va="center", fontsize=T3, weight="bold", color=INK2)
ax.text(gx - 3.0, 50.4, r"out of range: $a_d=0$", ha="center", va="center", fontsize=T3, color=MUTED)
arrow([(gx + r, 42), (145, 42)])

# blend
rbox(145, 31, 27, 22, TINTS["dual"], DUAL, lw=1.0)
ax.text(158.5, 48.5, "Per-variable", ha="center", va="center", fontsize=T1, weight="bold", color=INK, zorder=4)
ax.text(158.5, 45.0, "blend", ha="center", va="center", fontsize=T1, weight="bold", color=INK, zorder=4)
ax.text(158.5, 40.4, r"$a_d\,\hat z^{\mathrm{G}}+(1-a_d)\,\hat z^{\mathrm{S}}$", ha="center", va="center", fontsize=T2, color=INK2, zorder=4)
pill(ax, 158.5, 35.4, "Eq. 7", fc="white", ec="#d4d4cf", size=T3, color=MUTED, pad=0.22)
arrow([(139.2, 68.5), (158.5, 68.5), (158.5, 53)], color=GLOB)
arrow([(139.2, 15.5), (158.5, 15.5), (158.5, 31)], color=SEQ)

# output
arrow([(172, 42), (175.5, 42)], color=DUAL, lw=0.9)
ow, oh = window(176, 42, 1.5, 0.3, fill=DUAL)
ax.text(176 + ow / 2, 50.5, "Imputed", ha="center", va="center", fontsize=T1, weight="bold", color=INK)
ax.text(176 + ow / 2, 47.2, "window", ha="center", va="center", fontsize=T1, weight="bold", color=INK)
save(fig, "fig_2")
