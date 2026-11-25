"""Train all 7 classifiers on the same split and write one comparison table.
  python compare.py             full run (CPU, roughly 1-2 hours)
  python compare.py --quick     smoke test of every model
  python compare.py --report    only rebuild the table from existing results.json files
  python compare.py --only 03 07   run just some folders
Logs go to logs/<folder>.log; the table goes to results.md and results.json."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FOLDERS = sorted(p for p in HERE.iterdir() if p.is_dir() and p.name[:2].isdigit())

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--report", action="store_true")
ap.add_argument("--only", nargs="*", default=None, help="folder number prefixes, e.g. 01 05")
ap.add_argument("--tag", default="", help="passed to train.py; also suffixes the log names")
a, extra = ap.parse_known_args()
if a.tag:
    extra += ["--tag", a.tag]

(HERE / "logs").mkdir(exist_ok=True)
if not a.report:
    for folder in FOLDERS:
        if a.only and folder.name[:2] not in a.only:
            continue
        log = HERE / "logs" / f"{folder.name}{a.tag}{'_full' if '--full' in extra else ''}.log"
        cmd = [sys.executable, "-u", str(folder / "train.py")] + (["--quick"] if a.quick else []) + extra
        print(f"== {folder.name}  ->  {log.name}", flush=True)
        t0 = time.time()
        with log.open("w", encoding="utf-8") as f:
            code = subprocess.run(cmd, cwd=folder, stdout=f, stderr=subprocess.STDOUT).returncode
        print(f"   {'ok' if code == 0 else 'FAILED (exit %d)' % code} in {time.time() - t0:.0f}s", flush=True)

if a.tag:  # tagged runs (e.g. _holdout) have their own report; leave the main table alone
    sys.exit(0)
rows = []
for folder in FOLDERS:
    path = folder / "results.json"
    if path.exists():
        r = json.loads(path.read_text(encoding="utf-8"))
        r["folder"] = folder.name
        rows.append(r)

header = ("| # | Model | Type | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | CharBot det. | "
          "Word-concat det. | Train s | ms/1k | Params |")
data_note = rows[0].get("data", "UMUDGA 10k lists") if rows else ""
test_note = ("Test set = 100,000 held-out DGA names (2,000 per family) + 100,000 held-out legit names."
             if data_note == "full UMUDGA" else "Test set = held-out 10%.")
lines = ["# DGA classifier comparison", "", f"Data: {data_note}. Same split, same seed, same metrics. {test_note}",
         "", header, "|" + "---|" * 14]
for r in rows:
    t, rb = r["test"], r["robustness"]
    lines.append(f"| {r['folder'][:2]} | {r['model']} | {r['type']} | {t['accuracy']:.4f} | {t['precision']:.4f} | "
                 f"{t['recall']:.4f} | {t['f1']:.4f} | {t['roc_auc']:.4f} | {t['false_positive_rate']:.4f} | "
                 f"{rb['charbot_detection_rate']:.3f} | {rb['word_concat_detection_rate']:.3f} | "
                 f"{r['train_seconds']} | {r['ms_per_1000']} | {r['params']} |")
lines += ["", "## Hardest DGA families per model (lowest recall)", ""]
for r in rows:
    fams = r["test"]["recall_by_family"]
    lines.append(f"- **{r['model']}**: " + ", ".join(f"{f} {fams[f]:.2f}" for f in r["test"]["worst_families"]))
for r in rows:
    if "multiclass" in r:
        m = r["multiclass"]
        lines += ["", f"## Family classification ({r['model']}, {m['classes']} classes)", "",
                  f"accuracy {m['accuracy']:.4f}, macro-F1 {m['macro_f1']:.4f}, weighted-F1 {m['weighted_f1']:.4f}"]
lines += ["", "CharBot / word-concat det. = share of adversarial names (built from unseen legit test names) "
          "flagged as DGA; higher = more robust.", "Integrated-DL is the only model trained with CharBot-style "
          "adversarial examples, as its paper focuses on adversarial robustness."]
(HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
(HERE / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
print("\n".join(lines))
