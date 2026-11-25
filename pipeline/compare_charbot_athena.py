"""CharBot vs Athena: evasion rate against the 7 DGA classifiers.

1. Generate N CharBot names (Peck et al. 2019: change 2 chars of a benign Majestic label; novelty +
   1-edit-lookalike filtered exactly like Athena, via baselines/common_gen).
2. Score them with every saved classifier (no retraining).
3. Read Athena's per-model scores from an existing run's scores.csv (default: out_35k).
4. Evasion = share scored < 0.5 (not flagged). Write report.md, a bar chart and JSON.

  python compare_charbot_athena.py --n 30000 --athena-out out_35k --out charbot_vs_athena
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # AthenaExp
sys.path.insert(0, str(ROOT / "dga_classifier"))
sys.path.insert(0, str(ROOT / "baselines"))
from common.scorers import load_all  # noqa: E402
import charbot  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=30000)
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--athena-out", default="out_35k", help="pipeline/<folder>/scores.csv to read Athena scores from")
ap.add_argument("--out", default="charbot_vs_athena", help="output folder name inside pipeline/")
a = ap.parse_args()
OUT = Path(__file__).resolve().parent / a.out
OUT.mkdir(exist_ok=True)

# ------------------------------------------------------------------ 1. generate CharBot names
t0 = time.time()
charbot_names = charbot.generate(n=a.n, seed=a.seed)
print(f"generated {len(charbot_names)} CharBot names in {time.time() - t0:.0f}s", flush=True)
(OUT / "charbot_names.txt").write_text("\n".join(charbot_names) + "\n", encoding="utf-8")

# ------------------------------------------------------------------ 2. score with every classifier
models = load_all()
if not models:
    raise SystemExit("No saved classifiers found.")
ids = list(models)
names = {f: models[f][0] for f in ids}
quick = {f: models[f][2].get("quick", False) for f in ids}
cb_scores = {}
for f in ids:
    cb_scores[f] = np.asarray(models[f][1](charbot_names), dtype=float)
    print(f"scored {len(charbot_names)} CharBot names with {names[f]}", flush=True)

# ------------------------------------------------------------------ 3. read Athena scores from the 35k run
ath_csv = Path(__file__).resolve().parent / a.athena_out / "scores.csv"
if not ath_csv.exists():
    raise SystemExit(f"Athena scores not found: {ath_csv} (run run_pipeline.py --out {a.athena_out} first)")
ath_scores = {f: [] for f in ids}
with ath_csv.open(encoding="utf-8") as fh:
    r = csv.DictReader(fh)
    for row in r:
        for f in ids:
            ath_scores[f].append(float(row[f"{names[f]}_athena"]))
ath_scores = {f: np.asarray(v, dtype=float) for f, v in ath_scores.items()}
n_ath = len(next(iter(ath_scores.values())))

# ------------------------------------------------------------------ 4. metrics
ev = lambda s: float(np.mean(s < 0.5))
det = lambda s: float(np.mean(s >= 0.5))
rows = []
for f in ids:
    rows.append({"model": names[f], "quick": quick[f],
                 "cb_evasion": ev(cb_scores[f]), "ath_evasion": ev(ath_scores[f]),
                 "cb_detected": det(cb_scores[f]), "ath_detected": det(ath_scores[f]),
                 "cb_mean": float(cb_scores[f].mean()), "ath_mean": float(ath_scores[f].mean())})
# majority vote
cb_votes = sum((cb_scores[f] >= 0.5).astype(int) for f in ids)
ath_votes = sum((ath_scores[f] >= 0.5).astype(int) for f in ids)
maj = len(ids) / 2
rows.append({"model": "Majority vote", "quick": False,
             "cb_evasion": float((cb_votes <= maj).mean()), "ath_evasion": float((ath_votes <= maj).mean()),
             "cb_detected": float((cb_votes > maj).mean()), "ath_detected": float((ath_votes > maj).mean()),
             "cb_mean": float("nan"), "ath_mean": float("nan")})

(OUT / "charbot_vs_athena.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

# ------------------------------------------------------------------ 5. chart
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
ORANGE, GREY = "#eb6834", "#898781"  # Athena, CharBot
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2})
labels = [r["model"] for r in rows]
y = np.arange(len(labels))[::-1]
h = 0.36
fig, ax = plt.subplots(figsize=(9, 6), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(AXIS)
ax.tick_params(length=0)
ax.grid(axis="x", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ab = ax.barh(y + h / 2 + 0.02, [r["ath_evasion"] * 100 for r in rows], h, color=ORANGE, label="Athena")
cb = ax.barh(y - h / 2 - 0.02, [r["cb_evasion"] * 100 for r in rows], h, color=GREY, label="CharBot")
for bars in (ab, cb):
    for b in bars:
        ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, f"{b.get_width():.0f}%",
                va="center", fontsize=8.5, color=INK2)
ax.set_yticks(y, labels)
ax.set_xlim(0, 112)
ax.set_xlabel("Evasion rate: share of generated names scored < 0.5 (not flagged as DGA)")
ax.set_title(f"CharBot vs Athena evasion  (CharBot n={len(charbot_names)}, Athena n={n_ath})",
             loc="left", fontsize=12, color=INK, pad=28)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "chart_charbot_vs_athena.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------ 6. markdown report
def pct(x):
    return "-" if x != x else f"{x * 100:.1f}%"


def num(x):
    return "-" if x != x else f"{x:.3f}"


L = [
    "# CharBot vs Athena: evasion against the 7 DGA classifiers", "",
    f"Run: {datetime.now():%Y-%m-%d %H:%M} | seed {a.seed} | CharBot n={len(charbot_names)} | "
    f"Athena n={n_ath} (from `{a.athena_out}/scores.csv`)", "",
    "**CharBot** (Peck et al., IEEE Access 2019): take a benign Majestic-200k label and change two random "
    "characters; keep it only if it is novel and not a 1-edit lookalike of a known legit name, the exact "
    "same novelty bar Athena's names pass. No training, no knowledge of the target classifier.",
    "",
    "Evasion = share of generated names scored **< 0.5** (treated as legit, i.e. the attack got through). "
    "Higher = better evasion.", "",
    "## Evasion rate", "",
    "| Model | CharBot evasion | Athena evasion | Difference (Athena - CharBot) | CharBot mean score | Athena mean score |",
    "|---|---|---|---|---|---|",
]
for r in rows:
    tag = " *" if r["quick"] else ""
    diff = r["ath_evasion"] - r["cb_evasion"]
    L.append(f"| {r['model']}{tag} | {pct(r['cb_evasion'])} | **{pct(r['ath_evasion'])}** | "
             f"{diff * 100:+.1f} pts | {num(r['cb_mean'])} | {num(r['ath_mean'])} |")
L += ["", "![CharBot vs Athena evasion](chart_charbot_vs_athena.png)", "",
      "## Detection rate (share flagged as DGA, score ≥ 0.5)", "",
      "| Model | CharBot detected | Athena detected |", "|---|---|---|"]
for r in rows:
    L.append(f"| {r['model']} | {pct(r['cb_detected'])} | {pct(r['ath_detected'])} |")

cb_mean_ev = np.mean([r["cb_evasion"] for r in rows[:-1]])
ath_mean_ev = np.mean([r["ath_evasion"] for r in rows[:-1]])
L += ["", "## Summary",
      f"- Mean evasion across the 7 classifiers: **CharBot {cb_mean_ev * 100:.1f}%**, "
      f"**Athena {ath_mean_ev * 100:.1f}%**.",
      f"- Both are strong evaders, but Athena {'edges out' if ath_mean_ev > cb_mean_ev else 'trails'} "
      f"CharBot by {abs(ath_mean_ev - cb_mean_ev) * 100:.1f} pts on average.",
      "- * marks a classifier still on smoke-test weights at run time.",
      "", "CharBot names: `charbot_names.txt`. Per-model raw numbers: `charbot_vs_athena.json`."]
(OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")

print("\n".join(L[:12 + len(rows)]))
print(f"\nsaved {OUT / 'report.md'}, chart_charbot_vs_athena.png, charbot_vs_athena.json, charbot_names.txt")
