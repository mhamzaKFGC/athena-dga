"""Unseen-family experiment report. Reads <model>/results_holdout.json written by
  python compare.py --holdout <families> --tag _holdout
and compares recall on families the model never trained on with recall on families it did see.
Writes experiments/out/holdout_report.md and two charts."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

rows = []
for folder in sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name[:2].isdigit()):
    path = folder / "results_holdout.json"
    if path.exists():
        rows.append(json.loads(path.read_text(encoding="utf-8")))
if not rows:
    sys.exit("No results_holdout.json files found.")
held = rows[0]["holdout"]
for r in rows:
    rec = r["test"]["recall_by_family"]
    r["unseen_recall"] = float(np.mean([rec[f] for f in held if f in rec]))
    r["seen_recall"] = float(np.mean([v for f, v in rec.items() if f not in held]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
names = [r["model"] for r in rows]
y = np.arange(len(rows))[::-1]
fig, ax = plt.subplots(figsize=(9, 4.8))
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(AXIS)
ax.grid(axis="x", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.tick_params(length=0)
b1 = ax.barh(y + 0.2, [r["seen_recall"] * 100 for r in rows], 0.38, color=BLUE, label="Families seen in training")
b2 = ax.barh(y - 0.2, [r["unseen_recall"] * 100 for r in rows], 0.38, color=ORANGE, label="Families never seen (held out)")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, f"{b.get_width():.0f}%", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(y, names)
ax.set_xlim(0, 108)
ax.set_xlabel("Mean detection rate (recall) per family")
ax.set_title("Can the classifiers catch DGA families they never trained on?", loc="left", fontsize=11, pad=24)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT / "chart_holdout_recall.png", dpi=150)
plt.close(fig)

mat = np.array([[r["test"]["recall_by_family"].get(f, np.nan) for f in held] for r in rows])
fig, ax = plt.subplots(figsize=(1.05 * len(held) + 2.8, 0.5 * len(rows) + 1.8))
ax.imshow(mat, cmap=LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#2a78d6", "#0d366b"]), vmin=0, vmax=1, aspect="auto")
for i in range(len(rows)):
    for j in range(len(held)):
        ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                color="#ffffff" if mat[i, j] > 0.6 else INK)
ax.set_xticks(range(len(held)), held, rotation=35, ha="right")
ax.set_yticks(range(len(rows)), names)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("Detection rate on each never-seen family", loc="left", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "chart_holdout_families.png", dpi=150)
plt.close(fig)

L = ["# Unseen-family experiment", "",
     f"Each classifier was retrained on the 10k-list data **without** these {len(held)} families: "
     f"{', '.join(held)}. It was then tested on all families. This shows how well a detector generalises to "
     "new malware it has never seen (zero-day DGAs).", "",
     "| Model | Recall: seen families | Recall: unseen families | Gap | FPR on legit | ROC-AUC (all) |",
     "|---|---|---|---|---|---|"]
for r in rows:
    L.append(f"| {r['model']} | {r['seen_recall'] * 100:.1f}% | {r['unseen_recall'] * 100:.1f}% | "
             f"{(r['seen_recall'] - r['unseen_recall']) * 100:+.1f} pts | {r['test']['false_positive_rate'] * 100:.1f}% | "
             f"{r['test']['roc_auc']:.4f} |")
best = max(rows, key=lambda r: r["unseen_recall"])
L += ["", f"Best on unseen families: **{best['model']}** ({best['unseen_recall'] * 100:.1f}%).", "",
      "![Seen vs unseen](chart_holdout_recall.png)", "", "![Per family](chart_holdout_families.png)", "",
      "Note: trained on the 10k-list data (2,000 names per family), not the full UMUDGA run, to fit the time budget."]
(OUT / "holdout_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
