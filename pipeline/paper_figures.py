"""Publication-quality figures for the CharBot-vs-Athena results section.

Reads the finished JSON from the 35k Athena run and the 30k CharBot run and renders three figures
into pipeline/paper/:
  Fig 1  mean evasion across the 7 classifiers: real DGA vs CharBot vs Athena
  Fig 2  per-classifier evasion, grouped: real DGA vs CharBot vs Athena
  Fig 3  Athena vs the 10 most evasive real DGA families (mean over 7 classifiers)
"""
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "paper"
OUT.mkdir(exist_ok=True)

dga = json.loads((HERE / "out_35k" / "dga_vs_athena.json").read_text(encoding="utf-8"))
cb = json.loads((HERE / "charbot_vs_athena" / "charbot_vs_athena.json").read_text(encoding="utf-8"))

# model order from the DGA summary (drop majority-vote row for per-model plots)
summary = [r for r in dga["summary"] if r["model"] != "Majority vote"]
cb_by = {r["model"]: r for r in cb if r["model"] != "Majority vote"}
models = [r["model"] for r in summary]
dga_ev = np.array([r["dga_evasion"] for r in summary]) * 100
ath_ev = np.array([r["athena_evasion"] for r in summary]) * 100
charb_ev = np.array([cb_by[m]["cb_evasion"] for m in models]) * 100

# ---------- palette ----------
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
GREY, SLATE, ORANGE = "#b9b7 af", "#5b6b7b", "#eb6834"
GREY = "#c2c0b8"      # real DGA
SLATE = "#6b7a88"     # CharBot
ORANGE = "#eb6834"    # Athena
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2})


def frame(ax, xgrid=True):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0)
    ax.grid(axis="x" if xgrid else "y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ---------- Fig 1: mean evasion, three attacks ----------
means = [dga_ev.mean(), charb_ev.mean(), ath_ev.mean()]
labs = ["Real DGA\n(50 families)", "CharBot\n(Peck 2019)", "Athena\n(this work)"]
cols = [GREY, SLATE, ORANGE]
fig, ax = plt.subplots(figsize=(6.2, 4.6), facecolor=SURFACE)
frame(ax, xgrid=False)
ax.grid(axis="y", color=GRID, linewidth=0.8)
bars = ax.bar(labs, means, width=0.6, color=cols)
for b, m in zip(bars, means):
    ax.text(b.get_x() + b.get_width() / 2, m + 1.5, f"{m:.1f}%", ha="center", fontsize=11, color=INK)
ax.set_ylim(0, 108)
ax.set_ylabel("Mean evasion rate across 7 classifiers (%)")
ax.set_title("Mean evasion against seven DGA classifiers", loc="left", fontsize=12.5, color=INK, pad=12)
fig.tight_layout()
fig.savefig(OUT / "fig1_mean_evasion.png", dpi=200)
plt.close(fig)

# ---------- Fig 2: per-classifier grouped ----------
y = np.arange(len(models))[::-1]
h = 0.26
fig, ax = plt.subplots(figsize=(9, 6.2), facecolor=SURFACE)
frame(ax)
ax.barh(y + h, dga_ev, h, color=GREY, label="Real DGA")
ax.barh(y, charb_ev, h, color=SLATE, label="CharBot")
ax.barh(y - h, ath_ev, h, color=ORANGE, label="Athena")
for yy, v in zip(y + h, dga_ev):
    ax.text(v + 1, yy, f"{v:.0f}", va="center", fontsize=7.5, color=INK2)
for yy, v in zip(y, charb_ev):
    ax.text(v + 1, yy, f"{v:.0f}", va="center", fontsize=7.5, color=INK2)
for yy, v in zip(y - h, ath_ev):
    ax.text(v + 1, yy, f"{v:.0f}", va="center", fontsize=7.5, color=INK2)
ax.set_yticks(y, models)
ax.set_xlim(0, 112)
ax.set_xlabel("Evasion rate (%): share of generated names scored < 0.5 (not flagged)")
ax.set_title("Per-classifier evasion: real DGA vs CharBot vs Athena", loc="left", fontsize=12.5, color=INK, pad=26)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=9.5)
fig.tight_layout()
fig.savefig(OUT / "fig2_per_classifier.png", dpi=200)
plt.close(fig)

# ---------- Fig 3: Athena vs top-10 evasive families ----------
fam = dga["family_evasion"]
fam_mean = {k: np.mean(list(v.values())) for k, v in fam.items()}
top = sorted(fam_mean.items(), key=lambda kv: -kv[1])[:10]
names = ["Athena"] + [k for k, _ in top]
vals = [dga["athena_mean_evasion"] * 100] + [v * 100 for _, v in top]
colors = [ORANGE] + [GREY] * len(top)
y = np.arange(len(names))[::-1]
fig, ax = plt.subplots(figsize=(8.4, 5.4), facecolor=SURFACE)
frame(ax)
bars = ax.barh(y, vals, 0.62, color=colors)
for b, v in zip(bars, vals):
    ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(y, names)
ax.set_xlim(0, 112)
ax.set_xlabel("Mean evasion rate across 7 classifiers (%)")
ax.set_title("Athena vs the ten most evasive real DGA families", loc="left", fontsize=12.5, color=INK, pad=12)
fig.tight_layout()
fig.savefig(OUT / "fig3_family_ranking.png", dpi=200)
plt.close(fig)

print("means  real-DGA %.1f  CharBot %.1f  Athena %.1f" % tuple(means))
print(f"saved 3 figures to {OUT}")
