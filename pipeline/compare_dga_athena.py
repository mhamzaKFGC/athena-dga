"""Evasion rate of real DGA domains (UMUDGA test split) vs Athena-generated domains, per classifier.
Athena scores are read from <out>/scores.csv written by run_pipeline.py; real DGA names are sampled from
the held-out test split (never used in training) and scored with the same saved models.
  python compare_dga_athena.py --out out_now --per-family 100"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "dga_classifier"))
from common.scorers import load_all  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="out")
ap.add_argument("--per-family", type=int, default=100)
ap.add_argument("--seed", type=int, default=2026)
a = ap.parse_args()
OUT = HERE / a.out
random.seed(a.seed)

# Athena scores from the pipeline run
rows = list(csv.DictReader((OUT / "scores.csv").open(encoding="utf-8")))
models = load_all()
names = [m[0] for m in models.values()]
quick = {m[0]: m[2]["quick"] for m in models.values()}
athena = {n: np.array([float(r[f"{n}_athena"]) for r in rows]) for n in names}
legit = {n: np.array([float(r[f"{n}_legit"]) for r in rows]) for n in names}
full_test = {}
for key, (name, _, _) in models.items():
    p = ROOT / "dga_classifier" / key / "results.json"
    full_test[name] = json.loads(p.read_text(encoding="utf-8"))["test"] if p.exists() else None

# Real DGA from the held-out test split
split = json.loads((ROOT / "dga_classifier" / "data" / "full" / "split.json").read_text(encoding="utf-8"))
by_fam = {}
for label, y, fam in split["test"]:
    if y == 1:
        by_fam.setdefault(fam, []).append(label)
fams = sorted(by_fam)
sample = [(fam, lab) for fam in fams for lab in random.sample(by_fam[fam], min(a.per_family, len(by_fam[fam])))]
labels = [lab for _, lab in sample]
fam_arr = np.array([f for f, _ in sample])
dga = {}
for key, (name, scorer, _) in models.items():
    dga[name] = np.asarray(scorer(labels), dtype=float)
    print(f"scored {len(labels)} DGA names with {name}", flush=True)

ev = lambda s: float(np.mean(s < 0.5))  # noqa: E731
vote_dga = np.sum([dga[n] >= 0.5 for n in names], axis=0) > len(names) / 2
vote_ath = np.sum([athena[n] >= 0.5 for n in names], axis=0) > len(names) / 2
summary = [{"model": n, "quick": quick[n], "dga_evasion": ev(dga[n]), "athena_evasion": ev(athena[n]),
            "dga_mean": float(dga[n].mean()), "athena_mean": float(athena[n].mean())} for n in names]
summary.append({"model": "Majority vote", "quick": False, "dga_evasion": float(1 - vote_dga.mean()),
                "athena_evasion": float(1 - vote_ath.mean()), "dga_mean": None, "athena_mean": None})
from sklearn.metrics import roc_auc_score  # noqa: E402

det = lambda s: float(np.mean(s >= 0.5))  # noqa: E731
valid = []
for n in names:
    ft = full_test[n] or {}
    valid.append({
        "model": n, "quick": quick[n],
        "det_dga": det(dga[n]), "det_athena": det(athena[n]), "fpr_legit": det(legit[n]),
        "auc_dga_vs_legit": float(roc_auc_score(np.r_[np.ones(len(dga[n])), np.zeros(len(legit[n]))],
                                                np.r_[dga[n], legit[n]])),
        "auc_athena_vs_legit": float(roc_auc_score(np.r_[np.ones(len(athena[n])), np.zeros(len(legit[n]))],
                                                   np.r_[athena[n], legit[n]])),
        "full_recall": ft.get("recall"), "full_precision": ft.get("precision"), "full_f1": ft.get("f1"),
        "full_auc": ft.get("roc_auc"), "full_fpr": ft.get("false_positive_rate")})
fam_ev = {f: {n: ev(dga[n][fam_arr == f]) for n in names} for f in fams}
fam_mean = {f: float(np.mean(list(fam_ev[f].values()))) for f in fams}
ath_mean = float(np.mean([ev(athena[n]) for n in names]))
rank = sorted(fams, key=lambda f: -fam_mean[f])

# ------------------------------------------------------------------ charts
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE, INK, INK2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
GREY, ORANGE = "#898781", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def style(ax, axis="x"):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


# 1) evasion per model
labs = [s["model"] + (" *" if s["quick"] else "") for s in summary]
y = np.arange(len(summary))[::-1]
fig, ax = plt.subplots(figsize=(9, 5))
style(ax)
b1 = ax.barh(y + 0.2, [s["dga_evasion"] * 100 for s in summary], 0.38, color=GREY, label="Real DGA domains (UMUDGA, 50 families)")
b2 = ax.barh(y - 0.2, [s["athena_evasion"] * 100 for s in summary], 0.38, color=ORANGE, label="Athena-generated domains")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, f"{b.get_width():.0f}%", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(y, labs)
ax.set_xlim(0, 110)
ax.set_xlabel("Evasion rate: share NOT flagged as DGA (score < 0.5)")
ax.set_title("Evasion rate: real DGA malware vs Athena", loc="left", fontsize=11, pad=24)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT / "chart_evasion_dga_vs_athena.png", dpi=150)
plt.close(fig)

# 2) mean evasion per family vs Athena
order = rank[::-1]
vals = [fam_mean[f] * 100 for f in order] + [ath_mean * 100]
lab2 = order + ["ATHENA"]
fig, ax = plt.subplots(figsize=(8.5, 11))
style(ax)
yy = np.arange(len(lab2))
ax.barh(yy, vals, 0.7, color=[GREY] * len(order) + [ORANGE])
for i, v in enumerate(vals):
    ax.text(v + 0.8, i, f"{v:.0f}%", va="center", fontsize=7.5, color=INK2)
ax.set_yticks(yy, lab2, fontsize=8)
ax.get_yticklabels()[-1].set_color(ORANGE)
ax.get_yticklabels()[-1].set_fontweight("bold")
ax.set_xlim(0, 110)
ax.set_xlabel("Mean evasion rate across the 7 classifiers")
ax.set_title("Every UMUDGA family vs Athena (mean evasion over 7 classifiers)", loc="left", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "chart_evasion_by_family.png", dpi=150)
plt.close(fig)

# 3) heatmap: top 15 most evasive families + Athena, per model
top = rank[:15]
mat = np.array([[ev(athena[n]) for n in names]] + [[fam_ev[f][n] for n in names] for f in top])
fig, ax = plt.subplots(figsize=(1.1 * len(names) + 3, 0.42 * len(mat) + 1.8))
ax.imshow(mat, cmap=LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#2a78d6", "#0d366b"]), vmin=0, vmax=1, aspect="auto")
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j, i, f"{mat[i, j] * 100:.0f}", ha="center", va="center", fontsize=8,
                color="#ffffff" if mat[i, j] > 0.55 else INK)
ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
ax.set_yticks(range(len(mat)), ["ATHENA"] + top)
ax.get_yticklabels()[0].set_color(ORANGE)
ax.get_yticklabels()[0].set_fontweight("bold")
ax.axhline(0.5, color=SURFACE, linewidth=3)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("Evasion rate (%) per classifier: Athena vs the 15 most evasive DGA families", loc="left", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "chart_evasion_heatmap.png", dpi=150)
plt.close(fig)

# 4) detection rate: real DGA vs Athena vs legit false alarms
BLUE = "#2a78d6"
labs4 = [v["model"] + (" *" if v["quick"] else "") for v in valid]
y = np.arange(len(valid))[::-1]
fig, ax = plt.subplots(figsize=(9, 5.4))
style(ax)
groups = [("det_dga", GREY, "Real DGA detected (should be high)"),
          ("det_athena", ORANGE, "Athena detected"),
          ("fpr_legit", BLUE, "Real legit wrongly flagged (false alarm)")]
for k, (key, col, lab) in enumerate(groups):
    bars = ax.barh(y + 0.27 - k * 0.27, [v[key] * 100 for v in valid], 0.25, color=col, label=lab)
    for b in bars:
        ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, f"{b.get_width():.0f}%", va="center", fontsize=7.5, color=INK2)
ax.set_yticks(y, labs4)
ax.set_xlim(0, 110)
ax.set_xlabel("Share flagged as DGA (score >= 0.5)")
ax.set_title("Detection rate: real DGA vs Athena vs legit", loc="left", fontsize=11, pad=24)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "chart_detection_rate.png", dpi=150)
plt.close(fig)

# 5) score distributions: real DGA vs legit vs Athena
fig, axes = plt.subplots(1, len(names), figsize=(2.0 * len(names), 4.2), sharey=True)
for ax, n in zip(axes, names):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0)
    bp = ax.boxplot([dga[n], legit[n], athena[n]], widths=0.6, patch_artist=True, showfliers=False,
                    medianprops={"color": INK, "linewidth": 1.2})
    for patch, col in zip(bp["boxes"], (GREY, BLUE, ORANGE)):
        patch.set_facecolor(col)
        patch.set_edgecolor(col)
        patch.set_alpha(0.85)
    ax.axhline(0.5, color=INK2, linestyle="--", linewidth=0.8)
    ax.set_xticks([1, 2, 3], ["DGA", "Legit", "Athena"], fontsize=8)
    ax.set_title(n + (" *" if quick[n] else ""), fontsize=8.5)
axes[0].set_ylabel("P(DGA)")
fig.suptitle("Score distribution per classifier (dashed line = 0.5 threshold)", x=0.01, ha="left", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "chart_score_boxplot.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------ report
beat = sum(1 for f in fams if ath_mean > fam_mean[f])
L = ["# Evasion rate: real DGA domains vs Athena", "",
     f"Real DGA: {len(labels):,} names ({a.per_family} from each of {len(fams)} UMUDGA families) from the held-out "
     f"test split. Athena: {len(rows)} names from `scores.csv`. Evasion = share scored < 0.5 (not flagged).", "",
     "## Per classifier", "",
     "| Model | Real DGA evasion | Athena evasion | Difference | Mean score: real DGA | Mean score: Athena |",
     "|---|---|---|---|---|---|"]
for s in summary:
    dm = f"{s['dga_mean']:.3f}" if s["dga_mean"] is not None else "-"
    am = f"{s['athena_mean']:.3f}" if s["athena_mean"] is not None else "-"
    L.append(f"| {s['model']}{' *' if s['quick'] else ''} | {s['dga_evasion'] * 100:.1f}% | **{s['athena_evasion'] * 100:.1f}%** | "
             f"{(s['athena_evasion'] - s['dga_evasion']) * 100:+.1f} pts | {dm} | {am} |")
if any(s["quick"] for s in summary):
    L += ["", "\\* still smoke-test weights at run time (full training not finished)."]
L += ["", "![Evasion per model](chart_evasion_dga_vs_athena.png)", "",
      "## Detection rate (share flagged as DGA)", "",
      "| Model | Real DGA detected | Athena detected | Legit wrongly flagged | AUC: real DGA vs legit | AUC: Athena vs legit |",
      "|---|---|---|---|---|---|"]
for v in valid:
    L.append(f"| {v['model']}{' *' if v['quick'] else ''} | {v['det_dga'] * 100:.1f}% | **{v['det_athena'] * 100:.1f}%** | "
             f"{v['fpr_legit'] * 100:.1f}% | {v['auc_dga_vs_legit']:.3f} | **{v['auc_athena_vs_legit']:.3f}** |")
L += ["", "AUC 1.0 = the model fully separates that set from real legit; 0.5 = it cannot tell them apart; "
      "below 0.5 = it scores that set as *more* legit than real legit names.", "",
      "![Detection rate](chart_detection_rate.png)", "", "![Score distributions](chart_score_boxplot.png)", "",
      "## Validation: are the classifiers really good on actual DGA?", "",
      "Full held-out test set (100,000 real DGA + 100,000 legit, from `results.json`) next to this run's 5,000-name "
      "sample. If the sample detection rate matches the full-test recall, the scoring here is consistent.", "",
      "| Model | Full test: precision | Full test: recall (detection) | Full test: F1 | Full test: ROC-AUC | Full test: FPR | "
      "This sample: detection | Gap |", "|---|---|---|---|---|---|---|---|"]
for v in valid:
    if v["full_recall"] is None:
        continue
    L.append(f"| {v['model']}{' *' if v['quick'] else ''} | {v['full_precision']:.4f} | {v['full_recall'] * 100:.1f}% | "
             f"{v['full_f1']:.4f} | {v['full_auc']:.4f} | {v['full_fpr'] * 100:.2f}% | {v['det_dga'] * 100:.1f}% | "
             f"{(v['det_dga'] - v['full_recall']) * 100:+.1f} pts |")
L += ["",
      f"## Athena vs every DGA family", "",
      f"Athena's mean evasion over the 7 classifiers is **{ath_mean * 100:.1f}%**, higher than **{beat} of {len(fams)}** "
      "real DGA families.", "",
      "| Rank | Family | Mean evasion (7 models) |", "|---|---|---|"]
pos = 1
placed = False
for f in rank:
    if not placed and ath_mean >= fam_mean[f]:
        L.append(f"| {pos} | **ATHENA** | **{ath_mean * 100:.1f}%** |")
        pos += 1
        placed = True
    L.append(f"| {pos} | {f} | {fam_mean[f] * 100:.1f}% |")
    pos += 1
if not placed:
    L.append(f"| {pos} | **ATHENA** | **{ath_mean * 100:.1f}%** |")
L += ["", "![By family](chart_evasion_by_family.png)", "", "![Heatmap](chart_evasion_heatmap.png)"]
(OUT / "dga_vs_athena.md").write_text("\n".join(L) + "\n", encoding="utf-8")
(OUT / "dga_vs_athena.json").write_text(json.dumps({"summary": summary, "detection_validation": valid,
                                                    "family_evasion": fam_ev,
                                                    "athena_mean_evasion": ath_mean}, indent=2), encoding="utf-8")
print("\n".join(L[:20]))
print(f"saved {OUT / 'dga_vs_athena.md'} and 5 charts")
