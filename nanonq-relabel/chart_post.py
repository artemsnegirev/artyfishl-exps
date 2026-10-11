# Post image: slope panels, same visual language. Left: Rank1 vs RankLLaMA on TREC DL19 before/after relabel (Rank1 paper, Table 5).
# Right: NanoNQ dense models: original qrels, binary relabel by both judges (results.json), and the same minus the 2 queries
# that most favour LateOn over e5-large (drop2.json). BM25 omitted: far below (48.2 -> 49.1), it only stretches the scale.
import json, os
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
NAVY, CREAM, CORAL, MUTED = "#0C1028", "#FFE4C8", "#FF6E32", "#8A8FA8"
for f in fm.findSystemFonts():
    if "Inter" in f: fm.fontManager.addfont(f)
plt.rcParams["font.family"] = "Inter"
W = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(f"{W}/results.json")); D = json.load(open(f"{W}/drop2.json"))
dense = ["lateon", "e5-large", "e5-small", "e5-base"]
fig, axes = plt.subplots(1, 2, figsize=(16, 8.6), dpi=180, gridspec_kw=dict(width_ratios=[1, 1.45]))
fig.patch.set_facecolor(NAVY)

def panel(ax, cols, heads, names, colors, lo, hi, dys, head, sub, foot):
    ax.set_facecolor(NAVY); n = len(cols)
    for x in range(n): ax.plot([x, x], [lo + .3, hi - 2.4], color=MUTED, lw=1.5, alpha=.5)
    for m in cols[0]:
        c = colors.get(m, CREAM); ys = [col[m] for col in cols]
        ax.plot(range(n), ys, color=c, lw=3.5); ax.plot(range(n), ys, "o", color=c, ms=13)
        ax.text(-.08, ys[0] + dys[0].get(m, 0), f"{names[m]}  {ys[0]:.1f}", color=c, fontsize=16, ha="right", va="center")
        ax.text(n - 1 + .08, ys[-1] + dys[-1].get(m, 0), f"{ys[-1]:.1f}  {names[m]}", color=c, fontsize=16, ha="left", va="center")
        for x in range(1, n - 1):
            ax.text(x + .07, ys[x] + dys[x].get(m, 0), f"{ys[x]:.1f}", color=c, fontsize=14, ha="left", va="center", bbox=dict(facecolor=NAVY, edgecolor="none", pad=1.5))
    for x, (t, col) in enumerate(heads):
        ax.text(x, hi - 1.3, t, color=col, fontsize=14, ha="center", va="center", linespacing=1.1)
    ax.set_xlim(-1.15, n - 1 + 1.15); ax.set_ylim(lo, hi); ax.axis("off")
    ax.set_title(head, color=CREAM, fontsize=21, fontweight="bold", loc="left", x=-0.02, pad=34)
    ax.text(-0.02, 1.025, sub, color=MUTED, fontsize=14, transform=ax.transAxes)
    ax.text(-0.02, -0.04, foot, color=CORAL, fontsize=15, fontweight="bold", transform=ax.transAxes)

NAME = {"e5-small": "e5-small", "e5-base": "e5-base", "e5-large": "e5-large", "lateon": "LateOn"}
panel(axes[0], [{"rank1": 66.1, "rankllama": 76.2}, {"rank1": 78.6, "rankllama": 76.9}],
      [("исходная\nразметка", MUTED), ("после\nдоразметки", CREAM)],
      {"rank1": "Rank1", "rankllama": "RankLLaMA"}, {"rank1": CORAL, "rankllama": CREAM}, 62, 84,
      [{}, {"rank1": .5, "rankllama": -.5}],
      "TREC DL19: победитель сменился", "реранкеры 7B, nDCG@10 ×100, 43 запроса", "Rank1 обогнала RankLLaMA")
o, n, d2 = R["old"]["scores"], R["both"]["scores"], D["scores"]
panel(axes[1], [{m: o[m] for m in dense}, {m: n[m] for m in dense}, {m: d2[m] for m in dense}],
      [("исходная\nразметка", MUTED), ("после\nдоразметки", CREAM), ("и без 2 запросов\nиз 50", CORAL)],
      NAME, {"lateon": CORAL}, 62, 84,
      [{"e5-small": .45, "e5-base": -.45}, {"lateon": .9, "e5-large": -.15, "e5-small": -.9, "e5-base": -.9}, {"e5-large": .75, "lateon": 0, "e5-small": -.75}],
      "NanoNQ: уверенность исчезла", "эмбеддеры, nDCG@10 ×100, 50 запросов",
      "значимых пар было 3 из 6, стало 0; без 2 запросов лидер сменился")
fig.subplots_adjust(left=0.04, right=0.985, top=0.86, bottom=0.08, wspace=0.08)
fig.savefig(f"{W}/chart.png", facecolor=NAVY)
