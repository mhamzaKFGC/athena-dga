"""XAI experiments on all 7 saved classifiers. Writes xai/out/XAI_report.md, xai_results.json and charts.

Explained names (fixed seed): 4 test names per DGA family, 200 legit test names, and up to 100 Athena names
(from pipeline/out/scores.csv). Methods: character Shapley (all models), Integrated Gradients (neural),
attention maps (03, 05), feature Shapley + permutation importance (02, 04), deletion faithfulness test,
cross-model and cross-method agreement.
Run: python xai/run_xai.py   [--perm 16] [--per-family 4] [--legit 200]"""
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from common import xai  # noqa: E402
from common.scorers import load_all, load_feature_model, load_torch_model, model_folders  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--perm", type=int, default=16)
ap.add_argument("--per-family", type=int, default=4)
ap.add_argument("--legit", type=int, default=200)
ap.add_argument("--seed", type=int, default=11)
a = ap.parse_args()
OUT = HERE / "out"
(OUT / "examples").mkdir(parents=True, exist_ok=True)
t0 = time.time()

# ------------------------------------------------------------------ names to explain
full_split = ROOT / "data" / "full" / "split.json"
small = sorted((ROOT / "data").glob("split_pf2000_nl100000_s42.json"))
split_path = full_split if full_split.exists() else small[0]
test = json.loads(split_path.read_text(encoding="utf-8"))["test"]
rng = random.Random(a.seed)
by_fam = {}
for r in test:
    by_fam.setdefault(r[2], []).append(r[0])
names, groups, fams = [], [], []
for fam in sorted(by_fam):
    pick = rng.sample(by_fam[fam], min(a.legit if fam == "legit" else a.per_family, len(by_fam[fam])))
    names += pick
    groups += ["legit" if fam == "legit" else "dga"] * len(pick)
    fams += [fam] * len(pick)
athena_csv = ROOT.parent / "pipeline" / "out" / "scores.csv"
if athena_csv.exists():
    for row in csv.DictReader(athena_csv.open(encoding="utf-8")):
        names.append(row["athena_domain"].split(".")[0]); groups.append("athena"); fams.append("athena")
names_arr, groups_arr = np.array(names), np.array(groups)
print(f"explaining {len(names)} names ({(groups_arr == 'dga').sum()} DGA, {(groups_arr == 'legit').sum()} legit, "
      f"{(groups_arr == 'athena').sum()} Athena) from {split_path.parent.name}/{split_path.name}", flush=True)

# ------------------------------------------------------------------ per model
models = load_all()
results, shap_by_model = {}, {}
for folder in model_folders():
    if folder.name not in models:
        continue
    name, scorer, info = models[folder.name]
    t1 = time.time()
    phis, base, scores = xai.char_shapley(scorer, names, n_perm=a.perm, seed=a.seed)
    shap_by_model[name] = phis
    scores = np.array(scores)
    pred = scores >= 0.5
    r = {"folder": folder.name, "quick_weights": info["quick"], "base_score_empty_name": base,
         "flagged_share": {g: float(pred[groups_arr == g].mean()) for g in ("legit", "dga", "athena")
                           if (groups_arr == g).any()}}
    for g in ("dga", "legit", "athena"):
        idx = np.where(groups_arr == g)[0]
        if len(idx):
            r[f"summary_{g}"] = xai.summarize([names[i] for i in idx], [phis[i] for i in idx])
    r["deletion"] = xai.deletion_test(scorer, names, phis, k=2, seed=a.seed)

    model = load_torch_model(folder)
    if model is not None:
        ig = xai.integrated_gradients(model, names)
        r["agreement_shapley_vs_ig"] = xai.rank_agreement(phis, ig)
        att = xai.attention_weights(model, names)
        if att is not None:
            r["agreement_abs_shapley_vs_attention"] = xai.rank_agreement([np.abs(p) for p in phis], att)
    fm = load_feature_model(folder)
    if fm is not None:
        clf, ref, fn = fm
        X = fn(names, ref)
        background = X[rng.sample(range(len(X)), min(100, len(X)))]
        fphi = xai.feature_shapley(lambda m: clf.predict_proba(m)[:, 1], X, background, n_perm=a.perm, seed=a.seed)
        feat_names = list(json.loads((folder / "results.json").read_text(encoding="utf-8"))["feature_importance"])
        # feature_importance in results.json is sorted; recover the extraction order from the train script
        order = {"02_MaldomDetector": ["length", "entropy", "vowel_ratio", "consonant_ratio", "digit_ratio",
                                       "hyphen_ratio", "longest_consonant_run", "longest_digit_run",
                                       "unique_char_ratio", "letter_digit_switches", "legit_bigram_share",
                                       "legit_trigram_share"],
                 "04_NGram-KL-Jaccard": ["kl2_legit", "kl2_dga", "jaccard2_legit", "jaccard2_dga", "kl3_legit",
                                         "kl3_dga", "jaccard3_legit", "jaccard3_dga", "length"]}[folder.name]
        assert set(order) == set(feat_names)
        r["feature_shapley_mean_abs"] = dict(sorted(zip(order, map(float, np.abs(fphi).mean(0))), key=lambda kv: -kv[1]))
        r["feature_shapley_mean_athena"] = dict(zip(order, map(float, fphi[groups_arr == "athena"].mean(0))))
        sub = rng.sample(range(len(test)), min(4000, len(test)))
        Xt = fn([test[i][0] for i in sub], ref)
        yt = [test[i][1] for i in sub]
        pi = permutation_importance(clf, Xt, yt, scoring="roc_auc", n_repeats=5, random_state=a.seed, n_jobs=-1)
        r["permutation_importance_auc_drop"] = dict(sorted(zip(order, map(float, pi.importances_mean)), key=lambda kv: -kv[1]))
    results[name] = r
    print(f"  {name}: done in {time.time() - t1:.0f}s", flush=True)

# agreement between models (same names, Shapley)
model_names = list(results)
agree = np.array([[xai.rank_agreement(shap_by_model[m1], shap_by_model[m2]) for m2 in model_names] for m1 in model_names])

# ------------------------------------------------------------------ charts
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
DIVERGING = LinearSegmentedColormap.from_list("legit_dga", ["#184f95", "#f0efec", "#b5332f"])  # blue=legit, red=DGA
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#2a78d6", "#0d366b"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def heat(matrix, rows, cols, title, path, cmap, norm=None, fmt="{:+.3f}", note=""):
    fig, ax = plt.subplots(figsize=(1.3 * len(cols) + 2.6, 0.48 * len(rows) + 1.6))
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = matrix[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(cols)), cols, rotation=0)
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=11, pad=10)
    if note:
        fig.text(0.01, 0.01, note, fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04 if note else 0, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


types = ["consonant", "vowel", "digit", "hyphen"]
m = np.array([[results[n]["summary_dga"]["mean_by_char_type"].get(t, np.nan) for t in types] for n in model_names])
lim = np.nanmax(np.abs(m)) or 1
heat(m, model_names, types, "Mean character contribution to P(DGA), true DGA names", OUT / "chart_char_type.png",
     DIVERGING, TwoSlopeNorm(0, -lim, lim), note="Red = pushes toward DGA, blue = toward legit. Character Shapley values.")
pos = list(results[model_names[0]]["summary_dga"]["mean_by_position_fifth"])
m = np.array([[results[n]["summary_dga"]["mean_by_position_fifth"].get(p, np.nan) for p in pos] for n in model_names])
lim = np.nanmax(np.abs(m)) or 1
heat(m, model_names, pos, "Where in the name the DGA evidence sits (position in the label)", OUT / "chart_position.png",
     DIVERGING, TwoSlopeNorm(0, -lim, lim), note="Mean character Shapley value by relative position, true DGA names.")
heat(agree, model_names, [n.split("-")[0][:10] for n in model_names], "Do models agree on which characters matter?",
     OUT / "chart_agreement.png", SEQ, None, fmt="{:.2f}", note="Mean Spearman correlation of character attributions on the same names.")

fig, ax = plt.subplots(figsize=(9, 4.8))
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(AXIS)
ax.grid(axis="x", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.tick_params(length=0)
y = np.arange(len(model_names))[::-1]
top = [results[n]["deletion"]["mean_drop_top"] for n in model_names]
rnd = [results[n]["deletion"]["mean_drop_random"] for n in model_names]
ax.barh(y + 0.2, top, 0.38, color=BLUE, label="Remove the 2 most important characters")
ax.barh(y - 0.2, rnd, 0.38, color=ORANGE, label="Remove 2 random characters")
ax.set_yticks(y, model_names)
ax.set_xlabel("Mean drop in P(DGA) score")
ax.set_title("Faithfulness: removing the explained characters changes the score far more", loc="left", fontsize=11, pad=24)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT / "chart_deletion.png", dpi=150)
plt.close(fig)

feat_models = [n for n in model_names if "feature_shapley_mean_abs" in results[n]]
if feat_models:
    fig, axes = plt.subplots(1, len(feat_models), figsize=(5.2 * len(feat_models), 4.6))
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, feat_models):
        d = results[n]["feature_shapley_mean_abs"]
        ks = list(d)[::-1]
        ax.barh(ks, [d[k] for k in ks], color=BLUE, height=0.6)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.set_title(n, loc="left", fontsize=10.5)
        ax.set_xlabel("Mean |Shapley value| on P(DGA)")
    fig.suptitle("Which hand-made features drive the forest models", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "chart_features.png", dpi=150)
    plt.close(fig)

# per-name character heatmaps (rows = models)
showcase = []
for g, k in (("legit", 3), ("dga", 6), ("athena", 3)):
    idx = [i for i in np.where(groups_arr == g)[0] if 6 <= len(names[i]) <= 18]
    showcase += rng.sample(idx, min(k, len(idx)))
for i in showcase:
    s = names[i]
    mat = np.array([shap_by_model[n][i] for n in model_names])
    lim = np.abs(mat).max() or 1
    fig, ax = plt.subplots(figsize=(0.42 * len(s) + 3.2, 0.42 * len(model_names) + 1.3))
    ax.imshow(mat, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for r_ in range(len(model_names)):
        for c_ in range(len(s)):
            ax.text(c_, r_, s[c_], ha="center", va="center", fontsize=10, color=INK, family="DejaVu Sans Mono")
    ax.set_yticks(range(len(model_names)), model_names)
    ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    sc = [float(np.sum(shap_by_model[n][i])) for n in model_names]
    ax.set_title(f"{s}   ({fams[i]})   red = toward DGA, blue = toward legit", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "examples" / f"{groups[i]}_{s}.png", dpi=150)
    plt.close(fig)

# ------------------------------------------------------------------ report
(OUT / "xai_results.json").write_text(json.dumps({"names": len(names), "split": str(split_path.name),
                                                   "models": results, "agreement": agree.tolist(),
                                                   "model_order": model_names}, indent=2), encoding="utf-8")
L = ["# XAI report: why the 7 DGA classifiers decide what they decide", "",
     f"{len(names)} names explained ({(groups_arr == 'dga').sum()} DGA from {len(by_fam) - 1} families, "
     f"{(groups_arr == 'legit').sum()} legit, {(groups_arr == 'athena').sum()} Athena). Test split: `{split_path.name}`. "
     f"Character Shapley with {a.perm} permutations per name. Runtime {time.time() - t0:.0f}s.", "",
     "## Methods", "",
     "- **Character Shapley (all 7 models):** each character's share of the P(DGA) score, by removing characters "
     "in random orders. Same method for neural and forest models, so explanations are comparable.",
     "- **Integrated Gradients (5 neural models):** gradient-based attribution on the logit; used to cross-check Shapley.",
     "- **Attention maps (03, 05):** the attention layer's weights, compared with |Shapley|.",
     "- **Feature Shapley + permutation importance (02, 04):** which hand-made features drive the forests.",
     "- **Deletion test:** delete the 2 most important characters vs 2 random ones. A faithful explanation causes a "
     "much larger score drop.", "",
     "## Faithfulness and agreement", "",
     "| Model | Drop: top-2 chars | Drop: random-2 | Shapley vs IG (Spearman) | |Shapley| vs attention |",
     "|---|---|---|---|---|"]
for n in model_names:
    r = results[n]
    d = r["deletion"]
    ig = r.get("agreement_shapley_vs_ig")
    at = r.get("agreement_abs_shapley_vs_attention")
    L.append(f"| {n}{' * smoke-test weights' if r['quick_weights'] else ''} | {d['mean_drop_top']:+.3f} | "
             f"{d['mean_drop_random']:+.3f} | {'-' if ig is None else f'{ig:.2f}'} | {'-' if at is None else f'{at:.2f}'} |")
L += ["", "![Deletion test](chart_deletion.png)", "", "![Agreement between models](chart_agreement.png)", "",
      "## What drives a DGA verdict", "", "![By character type](chart_char_type.png)", "",
      "![By position](chart_position.png)", "", "### Character n-grams pushing hardest toward DGA / toward legit", "",
      "| Model | Toward DGA (true DGA names) | Toward legit (true legit names) |", "|---|---|---|"]
for n in model_names:
    td = ", ".join(f"`{g}`" for g, _ in results[n]["summary_dga"]["ngrams_toward_dga"][:8])
    tl = ", ".join(f"`{g}`" for g, _ in results[n]["summary_legit"]["ngrams_toward_legit"][:8])
    L.append(f"| {n} | {td} | {tl} |")
if feat_models:
    L += ["", "## Forest models: feature importance", "", "![Features](chart_features.png)", ""]
    for n in feat_models:
        r = results[n]
        L.append(f"**{n}** - top features by mean |Shapley|: " + ", ".join(
            f"{k} ({v:.3f})" for k, v in list(r["feature_shapley_mean_abs"].items())[:5]) +
            ". Permutation importance (AUC drop): " + ", ".join(
            f"{k} ({v:.3f})" for k, v in list(r["permutation_importance_auc_drop"].items())[:5]) + ".")
        L.append("")
if (groups_arr == "athena").any():
    L += ["", "## Why Athena names look legit to the classifiers", "",
          "| Model | Athena flagged | Legit flagged | n-grams pushing Athena names toward legit |", "|---|---|---|---|"]
    for n in model_names:
        r = results[n]
        tl = ", ".join(f"`{g}`" for g, _ in r["summary_athena"]["ngrams_toward_legit"][:8])
        L.append(f"| {n} | {r['flagged_share'].get('athena', 0) * 100:.0f}% | {r['flagged_share']['legit'] * 100:.0f}% | {tl} |")
L += ["", "## Example explanations", "", "Each row is one model; red characters push toward DGA, blue toward legit.", ""]
for i in showcase:
    L += [f"**{names[i]}** ({fams[i]})", "", f"![{names[i]}](examples/{groups[i]}_{names[i]}.png)", ""]
L += ["## Limits", "",
      "- Shapley values here remove characters (the name gets shorter); other masking choices give somewhat different "
      "values. Integrated Gradients uses a zero-embedding baseline.",
      "- Monte-Carlo estimates with a fixed seed; small values are noisy.",
      "- Explanations describe these reconstructed models, not the original papers' code."]
(OUT / "XAI_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"saved {OUT / 'XAI_report.md'} in {time.time() - t0:.0f}s")
