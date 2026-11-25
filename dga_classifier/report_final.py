"""Build FINAL_REPORT.md from every experiment's outputs (run after the overnight chain)."""
import json
import re
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
FOLDERS = sorted(p for p in HERE.iterdir() if p.is_dir() and p.name[:2].isdigit())


def load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def section(md_path, start, stop="\n## "):
    if not md_path.exists():
        return "_not available_"
    text = md_path.read_text(encoding="utf-8")
    i = text.find(start)
    if i < 0:
        return "_not available_"
    j = text.find(stop, i + len(start))
    return text[i + len(start): j if j > 0 else None].strip()


def fix_images(text, prefix):
    return re.sub(r"\]\((?!http)([^)]+\.png)\)", lambda m: f"]({prefix}{m.group(1)})", text)


L = ["# DGA classifiers on full UMUDGA, with XAI and Athena: final report", "",
     f"Generated {datetime.now():%Y-%m-%d %H:%M}.", "",
     "## 1. Full-UMUDGA results (7 classifiers)", ""]
rows = []
for f in FOLDERS:
    full, small = load(f / "results.json"), load(HERE / "results_small" / f.name / "results.json")
    if full:
        rows.append((f, full, small))
L += ["| Model | Type | F1 (10k lists) | **F1 (full UMUDGA)** | ROC-AUC | FPR | CharBot det. | Training names seen | Train time |",
      "|---|---|---|---|---|---|---|---|---|"]
for f, full, small in rows:
    t = full["test"]
    seen = full.get("training_samples_seen")
    L.append(f"| {full['model']} | {full['type']} | {small['test']['f1']:.4f} | **{t['f1']:.4f}** | {t['roc_auc']:.4f} | "
             f"{t['false_positive_rate'] * 100:.2f}% | {full['robustness']['charbot_detection_rate'] * 100:.0f}% | "
             f"{seen:,} | {full['train_seconds'] / 60:.0f} min |" if small and seen else
             f"| {full['model']} | {full['type']} | - | **{t['f1']:.4f}** | {t['roc_auc']:.4f} | "
             f"{t['false_positive_rate'] * 100:.2f}% | {full['robustness']['charbot_detection_rate'] * 100:.0f}% | - | "
             f"{full['train_seconds'] / 60:.0f} min |")
if rows:
    best = max(rows, key=lambda x: x[1]["test"]["f1"])[1]
    L += ["", f"Best F1 on full UMUDGA: **{best['model']}** ({best['test']['f1']:.4f}). Test set: 100,000 held-out DGA "
          "names (2,000 from each of 50 families) + 100,000 held-out legit names. Full table: `results.md`."]
    for f, full, _ in rows:
        if "multiclass" in full:
            m = full["multiclass"]
            L += ["", f"**Family classification** ({full['model']}, {m['classes']} classes): accuracy {m['accuracy']:.4f}, "
                  f"macro-F1 {m['macro_f1']:.4f}."]
    L += ["", "Hardest families (lowest recall) per model:", ""]
    for f, full, _ in rows:
        fam = full["test"]["recall_by_family"]
        L.append(f"- {full['model']}: " + ", ".join(f"{k} {fam[k]:.2f}" for k in full["test"]["worst_families"]))

L += ["", "## 2. Unseen-family (zero-day) experiment", "",
      fix_images(section(HERE / "experiments" / "out" / "holdout_report.md", "# Unseen-family experiment", "\n# "),
                 "experiments/out/"),
      "", "## 3. Athena vs the classifiers (evasion)", "",
      fix_images(section(HERE.parent / "pipeline" / "out" / "results.md", "## Results"), "../pipeline/out/"), "",
      section(HERE.parent / "pipeline" / "out" / "results.md", "## Reading it"), "",
      "## 4. Explainability (XAI)", ""]
x = load(HERE / "xai" / "out" / "xai_results.json")
if x:
    L += [fix_images(section(HERE / "xai" / "out" / "XAI_report.md", "## Faithfulness and agreement"), "xai/out/"), "",
          fix_images(section(HERE / "xai" / "out" / "XAI_report.md", "## What drives a DGA verdict"), "xai/out/"), "",
          "Full XAI report with per-name explanations: `xai/out/XAI_report.md`."]
else:
    L.append("_XAI results not available._")
L += ["", "## 5. Limits", "",
      "- The 7 models are reconstructions from the papers' abstracts, not the authors' code.",
      "- 'Full UMUDGA': all 36M names were deduplicated and pooled (up to 300k per family kept as training pool); each "
      "model drew fresh samples from those pools every epoch within a CPU time budget, so no model saw every name.",
      "- One seed per experiment. The unseen-family experiment used the 10k-list data to fit the time budget.",
      "- Athena results use 100 names; differences of a few points are within noise."]
(HERE / "FINAL_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"saved {HERE / 'FINAL_REPORT.md'}")
