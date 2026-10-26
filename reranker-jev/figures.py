"""Post figures from results.json (300 queries, original vs relabelled qrels): figures/*.png.

    .venv/Scripts/python reranker-jev/figures.py

rank_table.png  place and nDCG@10 before / after, colour = change of place (not of the metric)
rank_slope.png  the same as a slope chart
Places are among the arms shown.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).parent
NAMES = {"jev-pw": "Jev pointwise", "jev-list": "Jev listwise", "fd": "FRIDA-Decisions",
         "bge": "BGE-reranker-v2-m3", "kalm-nano-r2": "KaLM-Nano-R2",
         "qwen3-0.6b": "Qwen3-Reranker-0.6B"}
TITLE = "Места реранкеров по nDCG@10, 300 запросов MIRACL-ru"
INK, MUT, LINE, BG = "#1f1f1d", "#6b6a63", "#e6e5e0", "#fcfcfb"
UP, DOWN, FLAT = "#1f7a4a", "#c4502a", "#8a897f"
JEV, BGE, OTHER = "#2a78d6", "#eb6834", "#b9b8b0"   # categorical slots 1-2, validated (light surface)


def load():
    r = json.load(open(ROOT / "results.json", encoding="utf-8"))
    o = {a: r["miracl"]["original"]["models"][a]["nDCG@10"] for a in NAMES}
    n = {a: r["miracl"]["relabelled"]["models"][a]["nDCG@10"] for a in NAMES}
    return o, n, sorted(NAMES, key=lambda a: -o[a]), sorted(NAMES, key=lambda a: -n[a])


def table(o, n, ro, rn):
    fig, ax = plt.subplots(figsize=(8.4, 5.0), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.axis("off")
    X = [0.02, 0.50, 0.70, 0.90]
    for x, t, ha in zip(X, ["Модель", "Исходная\nразметка", "После\nдоразметки", "Место"],
                        ["left", "center", "center", "center"]):
        ax.text(x, 1.0, t, color=MUT, fontsize=11, ha=ha, va="center")
    step = 0.87 / len(rn)
    for i, a in enumerate(rn):
        y = 0.87 - i * step
        ax.plot([0, 1], [y + step / 2] * 2, color=LINE, lw=1)
        d = ro.index(a) - rn.index(a)
        c, s = (UP, f"▲ {d}") if d > 0 else (DOWN, f"▼ {-d}") if d < 0 else (FLAT, "—")
        w = "bold" if a.startswith("jev") else "normal"
        ax.text(X[0], y, NAMES[a], color=INK, va="center", fontweight=w)
        ax.text(X[1], y, f"{ro.index(a) + 1}.  {o[a]:.1f}", color=MUT, ha="center", va="center")
        ax.text(X[2], y, f"{rn.index(a) + 1}.  {n[a]:.1f}", color=INK, ha="center", va="center", fontweight=w)
        ax.text(X[3], y, s, color=c, ha="center", va="center", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.87 - len(rn) * step + step / 2 - 0.02, 1.07)
    fig.suptitle(TITLE, y=0.985, fontsize=13, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "rank_table.png", facecolor=BG)


def slope(o, n, ro, rn):
    fig, ax = plt.subplots(figsize=(8, 5.6), dpi=200)
    fig.patch.set_facecolor(BG)
    col = {"jev-pw": JEV, "jev-list": JEV, "bge": BGE}
    for a in NAMES:
        c, hi = col.get(a, OTHER), a in col
        y0, y1 = -ro.index(a), -rn.index(a)
        ax.plot([0, 1], [y0, y1], color=c, lw=2.6 if hi else 1.6, zorder=3 if hi else 2, solid_capstyle="round")
        ax.scatter([0, 1], [y0, y1], s=46, color=c, zorder=4 if hi else 3, edgecolor=BG, linewidth=1.5)
        kw = dict(va="center", color=INK if hi else MUT, fontweight="bold" if hi else "normal")
        ax.text(-0.04, y0, f"{NAMES[a]}  {o[a]:.1f}", ha="right", **kw)
        ax.text(1.04, y1, f"{n[a]:.1f}  {NAMES[a]}", ha="left", **kw)
    ax.text(0, 0.9, "Исходная разметка MIRACL", ha="center", color=MUT, fontsize=10)
    ax.text(1, 0.9, "После доразметки", ha="center", color=MUT, fontsize=10)
    ax.set_xlim(-1.15, 2.15)
    ax.set_ylim(-len(NAMES) + 0.4, 1.3)
    ax.axis("off")
    fig.suptitle(TITLE, y=0.97, fontsize=13, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "rank_slope.png", facecolor=BG)


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    (ROOT / "figures").mkdir(exist_ok=True)
    args = load()
    table(*args)
    slope(*args)
