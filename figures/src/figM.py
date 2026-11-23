from jstyle import *
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
setup()
W, H = 175, 82
fig = plt.figure(figsize=(W/10*CM, H/10*CM))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
TINT = {GLOB: "#fdf1ea", SEQ: "#e8f7f0", DUAL: "#eaf2fc", NEUT: "#f3f3f3"}
LW = 0.8

def box(x0, y0, w, h, edge, title, lines, lw=LW, tsize=7.4, bsize=6.9, gap=3.7):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=1.5",
                                fc=TINT.get(edge, "white"), ec=edge, lw=lw, zorder=2))
    n = len(lines); top = y0 + h/2 + gap*n/2
    ax.text(x0 + w/2, top, title, ha="center", va="center", fontsize=tsize, weight="bold", color=INK, zorder=3)
    for i, s in enumerate(lines):
        ax.text(x0 + w/2, top - gap*(i+1), s, ha="center", va="center", fontsize=bsize, color=INK2, zorder=3)

def path(pts, color=INK2, lw=0.8, ls="-", head=True):
    xs, ys = zip(*pts)
    if len(pts) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, solid_capstyle="butt", zorder=1)
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>" if head else "-", mutation_scale=6.5,
                                 color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0, zorder=1))

rng = np.random.default_rng(4)
T, D = 9, 4
miss = rng.random((D, T)) < 0.3
miss[1, 3:6] = True; miss[3, 6:9] = True; miss[0, 0] = False
def grid(x0, y0, cell, filled):
    g = 0.32
    for d in range(D):
        for t in range(T):
            x = x0 + t*(cell + g); y = y0 + (D-1-d)*(cell + g)
            if miss[d, t] and not filled:
                ax.add_patch(Rectangle((x, y), cell, cell, fc="white", ec="#a9a9a9", lw=0.45, zorder=2))
            elif miss[d, t] and filled:
                ax.add_patch(Rectangle((x, y), cell, cell, fc=DUAL, ec=DUAL, lw=0.45, zorder=2))
            else:
                ax.add_patch(Rectangle((x, y), cell, cell, fc="#b9b9b9", ec="#b9b9b9", lw=0.45, zorder=2))
    return T*(cell+g) - g, D*(cell+g) - g

# input window
gw, gh = grid(4.5, 34.5, 2.2, False)
ax.text(4.5 + gw/2, 54.5, "Incomplete window", ha="center", va="center", fontsize=7.4, weight="bold", color=INK)
ax.text(4.5 + gw/2, 50.6, r"values $x_{td}$, mask $v_{td}$", ha="center", va="center", fontsize=6.9, color=INK2)
path([(4.5, 32.2), (4.5 + gw, 32.2)], color=MUTED, lw=0.5)
ax.text(4.5 + gw, 30.0, "time $t$", ha="right", va="center", fontsize=6.4, color=MUTED)
ax.text(2.6, 34.5 + gh/2, "variable $d$", ha="center", va="center", fontsize=6.4, color=MUTED, rotation=90)

# global standardization (shared)
box(31, 31, 30, 18, NEUT, "Global standardization", [r"$z_{td}=(x_{td}-\mu_d)/\sigma_d$", r"$\mu_d,\sigma_d$: training windows", "(Equation 3)"], gap=3.5)
path([(4.5 + gw + 1.2, 40), (31, 40)])
# fork
ax.plot([61, 65], [40, 40], color=INK2, lw=0.8, zorder=1)
ax.plot([65, 65], [14, 66], color=INK2, lw=0.8, zorder=1)
path([(65, 66), (83, 66)])
path([(65, 14), (67.5, 14)])
ax.text(63.6, 66, "G", ha="right", va="center", fontsize=7.6, weight="bold", color=GLOB)
ax.text(63.6, 14, "S", ha="right", va="center", fontsize=7.6, weight="bold", color=SEQ)

# lanes
box(83, 58, 25, 16, GLOB, r"Imputer $f_{\mathrm{G}}$", ["trained on G"])
box(67.5, 5, 35, 18, SEQ, "Per-sequence normalization", [r"$\tilde z_{td}=(z_{td}-m_d)/s_d$", r"$m_d,s_d$: visible values", "(Equation 5)"], gap=3.5, tsize=6.9)
box(106, 6, 22, 16, SEQ, r"Imputer $f_{\mathrm{S}}$", ["trained on S"])
path([(102.5, 14), (106, 14)])
ax.text(73, 58.0, "same imputer", ha="center", va="center", fontsize=6.5, color=MUTED, style="italic")
ax.text(73, 54.6, "and seed", ha="center", va="center", fontsize=6.5, color=MUTED, style="italic")

# calibration weights between the imputers
box(88, 31, 32, 18, NEUT, "Calibration weights", [r"$a_d$ for each variable", "held-out units", "(Equation 10)"], gap=3.5)
path([(95.5, 58), (95.5, 49)], ls=(0, (2, 1.4)), lw=0.7)
path([(108.5, 22), (108.5, 31)], ls=(0, (2, 1.4)), lw=0.7)
ax.text(97, 53.5, "calibration errors", ha="left", va="center", fontsize=6.2, color=MUTED)
ax.text(110.0, 26.5, "calibration errors", ha="left", va="center", fontsize=6.2, color=MUTED)

# outputs to the combination
path([(108, 66), (133, 66), (133, 46), (137, 46)], color=GLOB)
path([(128, 14), (133, 14), (133, 34), (137, 34)], color=SEQ)
path([(120, 40), (137, 40)])
ax.text(120.5, 67.8, r"$\hat z^{\mathrm{G}}$", ha="center", va="bottom", fontsize=7.4, color=GLOB)
ax.text(130.5, 15.8, r"$\hat z^{\mathrm{S}}$", ha="center", va="bottom", fontsize=7.4, color=SEQ)

# combination, guard, output
box(137, 30, 36, 20, DUAL, "Per-variable blending", [r"$\hat z_{td}=a_d\,\hat z^{\mathrm{G}}_{td}+(1-a_d)\,\hat z^{\mathrm{S}}_{td}$", r"missing positions only ($v_{td}=0$)", "(Equation 7)"], lw=1.2, gap=3.6)
box(137, 2, 36, 22, NEUT, "Range guard", ["window level or spread outside", r"the training range by $>\kappa$", r"$\rightarrow a_d=0$ for every $d$", "(Equations 14–15)"], gap=3.4)
path([(155, 24), (155, 30)])
ow, oh = grid(145, 59, 2.0, True)
path([(155, 50), (155, 57.3)], color=DUAL, lw=1.0)
ax.text(145 + ow/2, 74.5, "Imputed window", ha="center", va="center", fontsize=7.4, weight="bold", color=INK)
ax.text(145 + ow/2, 70.8, "imputed values in blue", ha="center", va="center", fontsize=6.4, color=MUTED)
save(fig, "fig_2")
