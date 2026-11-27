# Figure 5: NormBlend weights a_d on the globally standardized imputation, per variable and imputer, with imputer means
from jstyle import *
import json, numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
setup()
D = json.load(open("figdata2.json"))
BB = ["KNN", "BRITS", "SAITS", "TimesNet", "ImputeFormer", "Transformer"]
VN = {"CMAPSS": ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"],
      "CMAPSS_M": ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"],
      "MetroPT": ["TP2", "TP3", "H1", "DV press.", "Reservoirs", "Oil temp.", "Motor curr."],
      "SKAB": ["Accel. 1", "Accel. 2", "Current", "Pressure", "Temp.", "Thermocouple", "Voltage", "Flow"]}
cmap = LinearSegmentedColormap.from_list("ws", [SEQ, "#f6f6f4", GLOB])
W, H = 17.0, 3.75
fig = plt.figure(figsize=(W * CM, H * CM))
widths = [14, 14, 7, 8]; unit = 0.235; l0, b0, gap, mgap = 1.7, 1.55, 0.36, 0.1; ph = 6 * unit; MW = 2.0
x = l0
for ds, w in zip(["CMAPSS", "CMAPSS_M", "MetroPT", "SKAB"], widths):
    ax = fig.add_axes([x / W, b0 / H, (w + MW) * unit / W + mgap / W, ph / H]); ax.axis("off")
    M = np.array([[next(q["weight"] for q in D["weights"] if q["dataset"] == ds and q["model"] == m and q["var"] == k) for k in range(w)] for m in BB])
    mean = M.mean(axis=1)
    sx = mgap / unit
    for r in range(6):
        for c in range(w + 1):
            v = M[r, c] if c < w else mean[r]; xc = c + (sx if c == w else 0); cw = MW if c == w else 1
            ax.add_patch(FancyBboxPatch((xc + 0.06, r + 0.06), cw - 0.12, 0.88, boxstyle="round,pad=0,rounding_size=0.14",
                                        fc=cmap(v), ec="none", zorder=2))
            if c == w:
                ax.text(xc + cw / 2, r + 0.53, f"{v:.2f}", ha="center", va="center", fontsize=5.8,
                        color="white" if abs(v - 0.5) > 0.32 else INK, zorder=3)
    ax.set_xlim(0, w + MW + sx); ax.set_ylim(6, 0)
    for c, nm in enumerate(VN[ds]):
        ax.text(c + 0.5, 6.25, nm, rotation=90, ha="center", va="top", fontsize=6.0, color=INK2)
    ax.text(w + sx + MW / 2, 6.25, "mean", rotation=90, ha="center", va="top", fontsize=6.0, color=INK, weight="bold")
    ax.text((w + MW + sx) / 2, -0.45, DSN[ds], ha="center", va="bottom", fontsize=7.4, weight="bold", color=INK)
    if x == l0:
        for r, m in enumerate(BB):
            ax.text(-0.35, r + 0.5, m, ha="right", va="center", fontsize=6.8, color=INK2)
    x += (w + MW) * unit + mgap + gap
# colour bar (vertical, right)
cx = x - gap + 0.3
cax = fig.add_axes([cx / W, b0 / H, 0.16 / W, ph / H])
g = np.linspace(0, 1, 256)[:, None]
cax.imshow(g, aspect="auto", cmap=cmap, extent=(0, 1, 0, 1), origin="lower"); cax.set_xticks([])
cax.yaxis.tick_right(); cax.set_yticks([0, 0.5, 1]); cax.set_yticklabels(["0  per-seq.", "0.5", "1  global"], fontsize=5.8)
cax.tick_params(length=1.5, pad=1.5)
for sp in cax.spines.values(): sp.set_visible(False)
fig.text((cx + 0.08) / W, (b0 + ph + 0.18) / H, r"$a_d$", ha="center", va="bottom", fontsize=7.0, color=INK)
save(fig, "fig_5")
