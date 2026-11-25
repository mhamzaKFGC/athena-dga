"""Athena vs the 7 DGA classifiers.

1. Pick N random legit domains the classifiers never trained on (their held-out test split).
2. Use each one as Athena's candidate: its first 3 characters become Athena's prefix, and Athena
   generates one new name from it (novelty, lookalike and 12-character filters stay on).
3. Score the real legit names and the Athena names with every saved classifier (no training).
4. Write out/results.md, out/scores.csv and two charts.

Run from anywhere:  python pipeline/run_pipeline.py            (100 domains, seed 2026)
                    python pipeline/run_pipeline.py --n 200 --seed 7
"""
import argparse
import csv
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent  # AthenaExp
OUT = Path(__file__).resolve().parent / "out"
sys.path.insert(0, str(ROOT / "dga_classifier"))
from common.scorers import load_all  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--prefix-len", type=int, default=3)
ap.add_argument("--max-len", type=int, default=12)
ap.add_argument("--temperature", type=float, default=0.8)
ap.add_argument("--top-k", type=int, default=8)
ap.add_argument("--out", default="out", help="output folder name inside pipeline/")
a = ap.parse_args()
random.seed(a.seed)
torch.manual_seed(a.seed)
OUT = Path(__file__).resolve().parent / a.out
OUT.mkdir(exist_ok=True)

# ------------------------------------------------------------------ Athena (v1, saved weights)
ALPHA = "abcdefghijklmnopqrstuvwxyz0123456789-"
START, STOP = len(ALPHA), len(ALPHA) + 1
VALID = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
TLDS = ["com", "net", "org", "io", "co", "app", "dev", "xyz", "online", "site", "info", "ai"]


class CharacterGraph(nn.Module):  # identical to Athena_GNN.py / generate_only.py
    def __init__(self, width=128, n_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(STOP + 1, width)
        self.layers = nn.ModuleList(nn.Linear(2 * width, width) for _ in range(n_layers))
        self.readout = nn.Linear(width, STOP + 1)

    def forward(self, tokens):
        states = self.embedding(tokens)
        for layer in self.layers:
            previous = F.pad(states[:, :-1], (0, 0, 1, 0))
            states = layer(torch.cat((states, previous), dim=-1)).tanh()
        return self.readout(states)


def athena_known(path):
    known = set()
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        parts = line.strip().lower().removesuffix(".").split(".")
        if len(parts) <= 2 and all(VALID.fullmatch(p) for p in parts):
            known.add(parts[0])
    return known


def near_known(label, known):
    for i in range(len(label) + 1):
        for c in ALPHA:
            if label[:i] + c + label[i:] in known:
                return True
            if i < len(label) and c != label[i] and label[:i] + c + label[i + 1:] in known:
                return True
        if i < len(label) and label[:i] + label[i + 1:] in known:
            return True
    return False


@torch.no_grad()
def athena_generate(model, prefix, known, used, attempts=300, layers=3):
    limit = a.max_len + 1
    for _ in range(attempts):
        label = prefix
        while len(label) < limit:
            tail = ([START] + [ALPHA.index(c) for c in label])[-(layers + 1):]
            scores = model(torch.tensor([tail]))[0, -1]
            scores[START] = -torch.inf
            if label == prefix or label.endswith("-"):
                scores[STOP] = -torch.inf
            if len(label) == limit - 1:
                scores[ALPHA.index("-")] = -torch.inf
            cutoff = scores.topk(a.top_k).values[-1]
            scores[scores < cutoff] = -torch.inf
            token = int(torch.multinomial((scores / a.temperature).softmax(0), 1).item())
            if token == STOP:
                break
            label += ALPHA[token]
        if (len(label) <= a.max_len and label not in known and label not in used
                and VALID.fullmatch(label) and not near_known(label, known)):
            return label
    return None


t0 = time.time()
athena = CharacterGraph()
athena.load_state_dict(torch.load(ROOT / "athena_v1_weights.pt", map_location="cpu"))
athena.eval()
known = athena_known(ROOT / "data" / "majestic" / "domains_200k.txt")

# ------------------------------------------------------------------ 1. sample legit candidates
full_split = ROOT / "dga_classifier" / "data" / "full" / "split.json"  # matches full-UMUDGA models
split = [full_split] if full_split.exists() else sorted((ROOT / "dga_classifier" / "data").glob("split_pf2000_nl100000_s42.json"))
if not split:
    raise SystemExit("Classifier split not found - run dga_classifier/compare.py first.")
test_legit = sorted({r[0] for r in json.loads(split[0].read_text(encoding="utf-8"))["test"] if r[1] == 0})
pool = [s for s in test_legit if len(s) >= a.prefix_len + 1 and VALID.fullmatch(s)]
rng = random.Random(a.seed)
rng.shuffle(pool)

# ------------------------------------------------------------------ 2. Athena generates from each candidate
pairs, used = [], set()
for legit in pool:
    if len(pairs) == a.n:
        break
    prefix = legit[:a.prefix_len]
    name = athena_generate(athena, prefix, known, used)
    if name:
        used.add(name)
        pairs.append({"legit": legit, "prefix": prefix, "athena": name,
                      "athena_domain": f"{name}.{TLDS[len(pairs) % len(TLDS)]}"})
print(f"generated {len(pairs)} Athena names from {len(pairs)} legit candidates in {time.time() - t0:.0f}s")

# ------------------------------------------------------------------ 3. score with every saved classifier
models = load_all()
if not models:
    raise SystemExit("No saved classifiers found in dga_classifier/.")
legit_labels = [p["legit"] for p in pairs]
athena_labels = [p["athena"] for p in pairs]
scores = {}
for folder, (name, scorer, info) in models.items():
    scores[folder] = {"name": name, "quick": info["quick"],
                      "legit": np.asarray(scorer(legit_labels), dtype=float),
                      "athena": np.asarray(scorer(athena_labels), dtype=float)}
ids = list(scores)
votes_legit = sum((scores[f]["legit"] >= 0.5).astype(int) for f in ids)
votes_athena = sum((scores[f]["athena"] >= 0.5).astype(int) for f in ids)
majority = len(ids) / 2

summary = []
for f in ids:
    s = scores[f]
    summary.append({"model": s["name"], "folder": f, "quick": s["quick"],
                    "legit_flagged": float((s["legit"] >= 0.5).mean()), "legit_mean": float(s["legit"].mean()),
                    "athena_flagged": float((s["athena"] >= 0.5).mean()), "athena_mean": float(s["athena"].mean()),
                    "athena_median": float(np.median(s["athena"]))})
summary.append({"model": "Majority vote", "folder": "-", "quick": False,
                "legit_flagged": float((votes_legit > majority).mean()), "legit_mean": float("nan"),
                "athena_flagged": float((votes_athena > majority).mean()), "athena_mean": float("nan"),
                "athena_median": float("nan")})

# ------------------------------------------------------------------ 4a. CSV
with (OUT / "scores.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["legit", "prefix", "athena_domain"] + [f"{scores[f]['name']}_legit" for f in ids]
               + [f"{scores[f]['name']}_athena" for f in ids] + ["votes_legit", "votes_athena"])
    for k, p in enumerate(pairs):
        w.writerow([p["legit"], p["prefix"], p["athena_domain"]]
                   + [round(float(scores[f]["legit"][k]), 4) for f in ids]
                   + [round(float(scores[f]["athena"][k]), 4) for f in ids]
                   + [int(votes_legit[k]), int(votes_athena[k])])

# ------------------------------------------------------------------ 4b. charts (static PNG, light theme)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1 and 2: real legit, Athena
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2})


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


names = [r["model"] for r in summary]
y = np.arange(len(names))[::-1]
h = 0.36
fig, ax = plt.subplots(figsize=(9, 6), facecolor=SURFACE)
style(ax)
lb = ax.barh(y + h / 2 + 0.02, [r["legit_flagged"] * 100 for r in summary], h, color=BLUE, label="Real legit domains")
at = ax.barh(y - h / 2 - 0.02, [r["athena_flagged"] * 100 for r in summary], h, color=ORANGE,
             label="Athena-generated domains")
for bars in (lb, at):
    for b in bars:
        ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, f"{b.get_width():.0f}%",
                va="center", fontsize=8.5, color=INK2)
ax.set_yticks(y, names)
ax.set_xlim(0, 108)
ax.set_xlabel("Share flagged as DGA (score ≥ 0.5)")
ax.set_title(f"How often each classifier calls a name DGA  (n = {len(pairs)} each)", loc="left",
             fontsize=12, color=INK, pad=28)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "chart_flag_rate.png", dpi=150)
plt.close(fig)

model_names = [scores[f]["name"] for f in ids]
y = np.arange(len(ids))[::-1]
jit = np.random.default_rng(a.seed)
fig, ax = plt.subplots(figsize=(9, 6), facecolor=SURFACE)
style(ax)
for i, f in enumerate(ids):
    for vals, off, color, label in ((scores[f]["legit"], 0.18, BLUE, "Real legit domains"),
                                    (scores[f]["athena"], -0.18, ORANGE, "Athena-generated domains")):
        ax.scatter(vals, y[i] + off + jit.uniform(-0.08, 0.08, len(vals)), s=30, color=color, alpha=0.75,
                   edgecolors=SURFACE, linewidths=0.8, label=label if i == 0 else None, zorder=3)
ax.axvline(0.5, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
ax.text(0.505, y[0] + 0.55, "DGA threshold", fontsize=8.5, color=INK2)
ax.set_yticks(y, model_names)
ax.set_xlim(-0.02, 1.02)
ax.set_xlabel("P(DGA) from each classifier  (0 = looks legit, 1 = looks algorithmic)")
ax.set_title("Score of every name, per classifier", loc="left", fontsize=12, color=INK, pad=28)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "chart_score_distribution.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------ 4c. markdown report
def pct(x):
    return "-" if x != x else f"{x * 100:.0f}%"


def num(x):
    return "-" if x != x else f"{x:.3f}"


quick = [scores[f]["name"] for f in ids if scores[f]["quick"]]
mean_athena = np.mean([scores[f]["athena"] for f in ids], axis=0)
order = np.argsort(-mean_athena)
most = max(summary[:-1], key=lambda r: r["athena_flagged"])
least = min(summary[:-1], key=lambda r: r["athena_flagged"])
L = [
    "# Athena vs DGA classifiers", "",
    f"Run: {datetime.now():%Y-%m-%d %H:%M} | seed {a.seed} | {len(pairs)} candidate pairs | {len(ids)} classifiers", "",
    "## What was done",
    f"1. Picked {len(pairs)} random legit domains from the classifiers' **held-out test split** (never seen in training).",
    f"2. Gave each to **Athena** as its candidate: the first {a.prefix_len} characters became the prefix, and Athena "
    f"generated one new name (≤ {a.max_len} chars, temperature {a.temperature}, top-k {a.top_k}; rejects known "
    "names and one-edit lookalikes of the 200k Majestic list).",
    "3. Scored both sets with every saved classifier (no retraining). A score ≥ 0.5 means \"DGA\".", "",
    "## Results", "",
    "| Model | Real legit flagged as DGA (false positives) | Athena flagged as DGA | Athena evasion rate | "
    "Mean score: legit | Mean score: Athena | Median score: Athena |",
    "|---|---|---|---|---|---|---|",
]
for r in summary:
    tag = " * smoke-test weights" if r["quick"] else ""
    L.append(f"| {r['model']}{tag} | {pct(r['legit_flagged'])} | {pct(r['athena_flagged'])} | "
             f"**{pct(1 - r['athena_flagged'])}** | {num(r['legit_mean'])} | {num(r['athena_mean'])} | "
             f"{num(r['athena_median'])} |")
L += ["", "![Share flagged as DGA](chart_flag_rate.png)", "", "![Score distribution](chart_score_distribution.png)", "",
      "## Reading it",
      f"- Real legit names flagged = the classifier's **false-positive rate** on this sample.",
      f"- Athena names flagged = how often a classifier thinks an Athena name looks algorithmically generated.",
      "- Evasion rate = 100% minus Athena flagged: the share of Athena names the classifier treats as legit. "
      "Compare it with the false-positive rate: if a classifier flags Athena names about as often as real legit "
      "names, it cannot tell them apart.",
      f"- Most suspicious of Athena: **{most['model']}** ({pct(most['athena_flagged'])}). "
      f"Least: **{least['model']}** ({pct(least['athena_flagged'])}).",
      f"- Majority vote flags {pct(summary[-1]['athena_flagged'])} of Athena names vs "
      f"{pct(summary[-1]['legit_flagged'])} of real legit names."]
if quick:
    L += [f"- * {', '.join(quick)} still had smoke-test weights at run time; rerun after full training."]
L += ["", "## Athena names the classifiers found most DGA-like (top 10 by mean score)", "",
      "| Athena domain | From legit | Mean score | Votes |", "|---|---|---|---|"]
for k in order[:10]:
    L.append(f"| {pairs[k]['athena_domain']} | {pairs[k]['legit']} | {mean_athena[k]:.3f} | {votes_athena[k]}/{len(ids)} |")
L += ["", "## All pairs", "", "| # | Legit candidate | Prefix | Athena domain | Votes legit | Votes Athena |",
      "|---|---|---|---|---|---|"]
for k, p in enumerate(pairs):
    L.append(f"| {k + 1} | {p['legit']} | {p['prefix']} | {p['athena_domain']} | {votes_legit[k]}/{len(ids)} | "
             f"{votes_athena[k]}/{len(ids)} |")
L += ["", "Per-model scores for every name: `scores.csv`. TLDs on Athena names are added for display only; "
      "classifiers score the name before the TLD."]
(OUT / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")

print("\n".join(L[:14 + len(summary)]))
print(f"\nsaved {OUT / 'results.md'}, scores.csv, chart_flag_rate.png, chart_score_distribution.png")
