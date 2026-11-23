# Figure 4: NormBlend weights a_d on the globally standardized imputation, per variable and imputer
from jstyle import *
import json, numpy as np
from matplotlib.colors import LinearSegmentedColormap
setup()
D = json.load(open("figdata2.json"))
BB = ["KNN", "BRITS", "SAITS", "TimesNet", "ImputeFormer", "Transformer"]
VN = {"CMAPSS": ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"],
      "CMAPSS_M": ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"],
      "MetroPT": ["TP2", "TP3", "H1", "DV press.", "Reservoirs", "Oil temp.", "Motor curr."],
      "SKAB": ["Accel. 1", "Accel. 2", "Current", "Pressure", "Temp.", "Thermocouple", "Voltage", "Flow"]}
cmap = LinearSegmentedColormap.from_list("ws", [SEQ, "#f4f4f4", GLOB])
W, H = 17.0, 4.4
fig = plt.figure(figsize=(W * CM, H * CM))
widths = [14, 14, 7, 8]; unit = 0.28; l0, b0, gap = 2.05, 1.6, 0.30; ph = 6 * unit * 1.0
x = l0
for ds, w in zip(["CMAPSS", "CMAPSS_M", "MetroPT", "SKAB"], widths):
    ax = fig.add_axes([x / W, b0 / H, w * unit / W, ph / H])
    M = np.array([[next(q["weight"] for q in D["weights"] if q["dataset"] == ds and q["model"] == m and q["var"] == k) for k in range(w)] for m in BB])
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(w)); ax.set_xticklabels(VN[ds], rotation=90, fontsize=6.2)
    ax.set_yticks(range(len(BB))); ax.set_yticklabels(BB if x == l0 else [], fontsize=6.8)
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_title(DSN[ds], fontsize=7.6, pad=3, color=INK)
    x += w * unit + gap
cax = fig.add_axes([(x + 0.1) / W, b0 / H, 0.22 / W, ph / H])
cb = fig.colorbar(im, cax=cax); cb.set_ticks([0, 0.5, 1]); cb.ax.tick_params(labelsize=6.6, length=2)
cb.set_label("weight on global imputation", fontsize=6.8)
save(fig, "fig_4")
