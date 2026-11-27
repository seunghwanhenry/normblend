# Figure 7: case study, air leaks of the MetroPT-3 compressor. Subfigures fig_7_A-D (no labels inside, JCDE 6.7):
# (A) reservoir pressure at a leak onset, (B) oil temperature during the leak, (C, D) RMSE per imputer, normal vs leak windows
from jstyle import *
import json, numpy as np
from matplotlib.lines import Line2D
setup()
C = json.load(open("figdata2.json"))["case"]
T = C["trace"]; t = np.array(T["minute"]) / 60 + T["hour0"]
leak = T["leak_hour"]
VIEWS = [("global", "G"), ("seq", "S"), ("dual", "NB")]
MODELS = ["KNN", "BRITS", "SAITS", "TimesNet", "ImputeFormer", "Transformer"]
R = {(q["sig"], q["model"], q["failure"]): q for q in C["rmse"]}


def trace(ax, var, unit, ylim, yt, x0=None):
    grid(ax)
    ax.axvspan(leak, t.max() + 0.06, color=TINTS["neutral"], zorder=0, lw=0)
    y = np.array(T["truth"][var], float)
    brk = np.r_[np.diff(np.array(T["minute"])) > 1, False]                   # break the line between windows
    ax.plot(t, np.where(brk, np.nan, y), color=INK, lw=0.75, zorder=3)
    m = np.array(T["mask"][var], bool)
    for v, k in VIEWS:
        p = np.array([np.nan if q is None else q for q in T[k][var]], float)
        ax.plot(t, np.where(m, p, np.nan), color=COL[v], lw=1.5 if v == "dual" else 1.1, zorder=5 if v == "dual" else 4, solid_capstyle="butt")
    for a, b in T["gaps"][var]:
        ax.axvspan(t[a] - 0.5 / 60, t[b] + 0.5 / 60, ymin=0, ymax=0.035, color="#bdbdb8", lw=0, zorder=2)
    ax.axvline(leak, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=2)
    lo = t.min() if x0 is None else x0
    ax.set_xlim(lo - 0.05, t.max() + 0.05); ax.set_ylim(*ylim); ax.set_yticks(yt)
    hs = range(int(np.ceil(lo)), int(np.ceil(t.max())) + 1)
    ax.set_xticks(list(hs)); ax.set_xticklabels([f"{h:02d}:00" for h in hs])
    ax.set_ylabel(unit)


def legend(fig, y):
    h = [Line2D([], [], color=INK, lw=0.9, label="Observed record"), Line2D([], [], color="#bdbdb8", lw=4, label="Removed span")]
    h += [Line2D([], [], color=COL[v], lw=1.5 if v == "dual" else 1.1, marker=MK[v], ms=4, mec="white", mew=0.5, label=LBL[v]) for v, _ in VIEWS]
    fig.legend(handles=h, loc="upper center", ncol=5, bbox_to_anchor=(0.5, y), fontsize=7.0, frameon=False, handlelength=1.8, columnspacing=1.3)


def panel_a(fig, W, H, x, y):                                                  # axes box 14.2 x 3.3 cm at (x + 1.5, y + 0.85)
    ax = fig.add_axes([(x + 1.5) / W, (y + 0.85) / H, 13.85 / W, 3.3 / H])
    trace(ax, "Reservoirs", "Reservoir pressure (bar)", (7.4, 10.45), [7.5, 8.5, 9.5, 10])
    ax.set_xlabel(f"Time on {T['day']}")
    pill(ax, leak + 0.06, 10.17, "air leak documented from 10:00", ha="left", size=6.4, fc="white", ec=SPINE)
    ax.text(t.min() + 0.02, 10.17, "normal operation", ha="left", va="center", fontsize=6.4, color=INK2)


def panel_b(fig, W, H, x, y):
    ax = fig.add_axes([(x + 1.5) / W, (y + 0.85) / H, 13.85 / W, 2.9 / H])
    trace(ax, "Oil_temperature", "Oil temperature (°C)", (73.6, 78.2), [74, 75, 76, 77, 78], x0=leak)
    ax.set_xlabel(f"Time on {T['day']}, during the leak")


def dots(ax, sig, fail, xlim, xt, title, ylabels):
    grid(ax, "x")
    for i, mdl in enumerate(MODELS):
        q = R[(sig, mdl, fail)]; y = len(MODELS) - 1 - i
        xs = [q[k] for k in ("global", "seq", "normblend")]
        ax.plot([min(xs), max(xs)], [y, y], color="#d6d6d2", lw=1.6, zorder=1, solid_capstyle="round")
        for v, k in (("global", "global"), ("seq", "seq"), ("dual", "normblend")):
            ax.plot(q[k], y, marker=MK[v], ms=4.4 if v == "dual" else 4.0, color=COL[v], mec="white", mew=0.5, zorder=4 if v == "dual" else 3, ls="none")
    ax.set_yticks(range(len(MODELS))); ax.set_yticklabels(MODELS[::-1] if ylabels else [])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(MODELS) - 0.4); ax.set_xscale("log"); ax.set_xlim(*xlim)
    ax.set_xticks(xt); ax.set_xticklabels([f"{x:g}" for x in xt]); ax.minorticks_off()
    ax.set_title(title, fontsize=7.2, color=INK, pad=3)


CD = {"C": ("Reservoirs", "Reservoir pressure", "RMSE (bar)", (0.07, 0.75), [0.1, 0.2, 0.4]),
      "D": ("Oil_temperature", "Oil temperature", "RMSE (°C)", (0.45, 9.5), [0.5, 1, 2, 4, 8])}


def panel_cd(fig, W, H, x, y, key):                                           # box 7.8 x 5.0 cm
    sig, name, unit, xlim, xt = CD[key]; sw, g1, x0 = 2.65, 0.3, x + 1.75
    for k, fail in enumerate((0, 1)):
        ax = fig.add_axes([(x0 + k * (sw + g1)) / W, (y + 0.85) / H, sw / W, 3.45 / H])
        dots(ax, sig, fail, xlim, xt, "normal windows" if fail == 0 else "leak windows", ylabels=(k == 0))
        if k == 0: ax.set_xlabel(unit, x=1.0 + g1 / sw / 2, ha="center")
    fig.text((x0 + sw + g1 / 2) / W, (y + 4.72) / H, name, ha="center", fontsize=7.6, weight="bold", color=INK)


# subfigures
for key, (w, h, draw) in {"A": (15.9, 4.75, panel_a), "B": (15.9, 3.95, panel_b),
                          "C": (7.8, 5.0, lambda f, W, H, x, y: panel_cd(f, W, H, x, y, "C")),
                          "D": (7.8, 5.0, lambda f, W, H, x, y: panel_cd(f, W, H, x, y, "D"))}.items():
    fig = plt.figure(figsize=(w * CM, h * CM)); draw(fig, w, h, 0, 0)
    if key == "A": legend(fig, 1.0)
    save(fig, f"fig_7_{key}")
# combined preview with (A)-(D) centred below each subfigure (draft only)
W, H, lab = 16.6, 4.75 + 3.95 + 5.0 + 3 * 0.55, 0.55
fig = plt.figure(figsize=(W * CM, H * CM)); y = H
for key, (x, w, h, draw) in (("A", (0.35, 15.9, 4.75, panel_a)), ("B", (0.35, 15.9, 3.95, panel_b))):
    y -= h; draw(fig, W, H, x, y); y -= lab
    fig.text((x + 1.5 + 6.93) / W, (y + 0.12) / H, f"({key})", ha="center", fontsize=8.5, color=INK)
y -= 5.0
for key, x in (("C", 0.0), ("D", 8.4)):
    panel_cd(fig, W, H, x, y, key); fig.text((x + 1.75 + 2.9) / W, (y - lab + 0.12) / H, f"({key})", ha="center", fontsize=8.5, color=INK)
legend(fig, 1.0)
os.makedirs(os.path.join(OUT, "preview"), exist_ok=True)
fig.savefig(os.path.join(OUT, "preview", "fig_7.png"), dpi=300, facecolor="white")
fig.savefig(os.path.join(OUT, "preview", "fig_7.pdf"), facecolor="white")
