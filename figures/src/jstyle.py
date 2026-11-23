import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os
OUT = os.environ.get("FIG_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # default: figures/
CM = 1/2.54
DUAL, GLOB, SEQ = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#1f1f1f", "#4d4d4d", "#8a8a8a", "#e4e4e4"
NEUT = "#6b6b6b"
MK = {"dual": "o", "global": "s", "seq": "^"}
COL = {"dual": DUAL, "global": GLOB, "seq": SEQ}
LBL = {"dual": "NormBlend", "global": "Global standardization", "seq": "Per-sequence normalization"}
COND = {"MCAR 0.3": "MCAR 30%", "MCAR 0.7": "MCAR 70%", "MNAR 0.3": "MNAR 30%", "GAP 0.4": "Gaps 40%"}
CONDS = ["MCAR 0.3", "MCAR 0.7", "MNAR 0.3", "GAP 0.4"]
DSN = {"Siemens": "Siemens", "ECG200": "ECG200", "Motor": "Motors", "Hydraulic": "Hydraulic rig", "CMAPSS": "C-MAPSS-SC", "CMAPSS_M": "C-MAPSS-MC", "MetroPT": "MetroPT-3", "SKAB": "SKAB"}

SPINE = "#9b9b9b"
TINTS = {"global": "#fdf0e9", "seq": "#e6f6ef", "dual": "#e9f1fb", "neutral": "#f4f4f2"}
DARK = {"global": "#b8461a", "seq": "#0f7a54", "dual": "#1d5aa6"}

def setup():
    plt.rcParams.update({
        "font.family": "Liberation Sans", "font.size": 8, "axes.labelsize": 8.2,
        "xtick.labelsize": 7.3, "ytick.labelsize": 7.3, "legend.fontsize": 7.2,
        "axes.edgecolor": SPINE, "axes.linewidth": 0.6, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.major.size": 2.6, "ytick.major.size": 2.6, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.dpi": 600, "figure.dpi": 150, "lines.linewidth": 1.0, "lines.solid_capstyle": "round",
        "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
        "mathtext.bf": "Liberation Sans:bold",
    })

def pill(ax, x, y, s, fc="#f1f1ef", ec="none", color=None, size=6.6, transform=None, ha="center", va="center", weight="normal", pad=0.28):
    return ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=color or INK, weight=weight, zorder=6, clip_on=False,
                   transform=transform or ax.transData, bbox=dict(boxstyle=f"round,pad={pad},rounding_size=0.8", fc=fc, ec=ec, lw=0.5))

def grid(ax, axis="y"):
    ax.grid(True, axis=axis, color="#e3e3e0", lw=0.5, ls=(0, (2.5, 2)), zorder=0)
    ax.set_axisbelow(True)

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=600, facecolor="white")
    fig.savefig(os.path.join(OUT, f"{name}.tif"), dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

def save_preview(fig, name):
    fig.savefig(os.path.join(OUT, "preview", f"{name}.png"), dpi=300, facecolor="white")
    plt.close(fig)
os.makedirs(os.path.join(OUT, "preview"), exist_ok=True)

def panels(name, draws, pw, ph, margins, gap=0.6, label_h=0.75, legend=None, legend_h=0.0, solo_top=None):
    """draws: list of functions f(ax, standalone). pw, ph: panel size in cm; margins: (l, r, b, t) in cm.
    Saves each panel as {name}_{A,B,..} (no labels) and a combined preview with (A), (B) centred below."""
    letters = "ABCDEF"
    n = len(draws)
    ms = margins if isinstance(margins[0], (list, tuple)) else [margins]*n
    pws = pw if isinstance(pw, (list, tuple)) else [pw]*n
    for i, f in enumerate(draws):
        w = pws[i]; l, r, b, t = ms[i]
        extra = (solo_top or [0]*n)[i]
        hh = ph + extra
        fig = plt.figure(figsize=(w*CM, hh*CM))
        ax = fig.add_axes([l/w, b/hh, (w-l-r)/w, (ph-b-t)/hh])
        f(ax, True)
        save(fig, f"{name}_{letters[i]}")
    W = sum(pws) + (n-1)*gap
    H = ph + label_h + legend_h
    fig = plt.figure(figsize=(W*CM, H*CM))
    x0 = 0
    for i, f in enumerate(draws):
        w = pws[i]; l, r, b, t = ms[i]
        ax = fig.add_axes([(x0+l)/W, (label_h+b)/H, (w-l-r)/W, (ph-b-t)/H])
        f(ax, False)
        fig.text((x0 + l + (w-l-r)/2)/W, 0.18/H, f"({letters[i]})", ha="center", va="bottom", fontsize=8.5, color=INK)
        x0 += w + gap
    if legend is not None:
        legend(fig, W, H)
    fig.savefig(os.path.join(OUT, "preview", f"{name}.png"), dpi=300, facecolor="white")
    fig.savefig(os.path.join(OUT, "preview", f"{name}.pdf"), facecolor="white")
    plt.close(fig)
