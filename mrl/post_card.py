"""Styled chart for the Telegram post: absolute nDCG@10 vs truncation dim (from results.csv)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent
BG, GRID, TEXT, MUTED = "#0d1129", "#262b47", "#f3e6d3", "#8b8fb0"
QWEN, FRIDA = "#f6e3c8", "#ff6b35"

df = pd.read_csv(ROOT / "results.csv")
df[["ndcg10", "recall100"]] *= 100  # show metrics as 0-100
q = df[df.model == "Qwen3-Embedding-0.6B"].sort_values("dim")
f = df[df.model == "FRIDA"].sort_values("dim")

plt.rcParams["font.family"] = ["Inter", "Segoe UI", "DejaVu Sans"]
fig, ax = plt.subplots(figsize=(10, 7.9), dpi=110)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

ax.plot(q.dim, q.ndcg10, color=QWEN, lw=3, marker="o", ms=9, label="Qwen3-Embedding-0.6B, учили с MRL", zorder=3)
ax.plot(f.dim, f.ndcg10, color=FRIDA, lw=3, marker="o", ms=9, label="FRIDA, без MRL", zorder=3)

ax.set_xscale("log", base=2)
ticks = [32, 64, 128, 256, 512, 1024, 1536]
ax.set_xticks(ticks)
ax.set_xticklabels([str(t) for t in ticks])
ax.minorticks_off()
ax.set_ylim(28, 76)
ax.set_yticks([30, 40, 50, 60, 70])
ax.grid(axis="y", color=GRID, lw=1)
ax.tick_params(colors=TEXT, labelsize=15, length=0, pad=12)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xlabel("сколько первых компонент оставили", color=MUTED, fontsize=15, labelpad=14)

# annotations
v = lambda d, m: float(d.loc[d.dim == m, "ndcg10"].iloc[0])
ax.annotate(f"{v(q, 32):.1f}", (32, v(q, 32)), xytext=(36, v(q, 32) - 3.5), color=QWEN, fontsize=17, fontweight="bold")
ax.annotate(f"{v(f, 32):.1f}", (32, v(f, 32)), xytext=(36, v(f, 32) - 3.5), color=FRIDA, fontsize=17, fontweight="bold")
ax.annotate(
    f"256: FRIDA {v(f, 256):.1f}, Qwen3 {v(q, 256):.1f}",
    xy=(256, v(q, 256)), xytext=(256, 56.5), ha="center", color=TEXT, fontsize=15,
    arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.2),
)

leg = ax.legend(loc="lower right", frameon=False, fontsize=15, labelcolor=TEXT, handlelength=2.2)
fig.text(0.06, 0.955, "Сколько nDCG@10 остается после обрезки эмбеддинга", color=TEXT, fontsize=22, fontweight="bold")
fig.text(0.06, 0.915, "RuBQ Retrieval, 56 826 документов, nDCG@10 × 100", color=MUTED, fontsize=15)
fig.subplots_adjust(left=0.1, right=0.96, top=0.87, bottom=0.12)
fig.savefig(ROOT / "post_card.png", facecolor=BG)
