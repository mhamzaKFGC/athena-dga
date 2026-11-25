"""Score domains with the SAVED models - no training.
  python predict.py google.com xkqzvtplwmrn.net 8f3a9c1d7e2b.ddns.net
  python predict.py --file domains.txt          (one domain per line, or a CSV with a 'domain' column)
  python predict.py --models 01,03 google.com   (only some models)
Each column is P(DGA) from one model; 'votes' = how many models say DGA (score >= 0.5)."""
import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common.data import VALID, label_of  # noqa: E402
from common.scorers import load_all  # noqa: E402


ap = argparse.ArgumentParser()
ap.add_argument("domains", nargs="*")
ap.add_argument("--file")
ap.add_argument("--models", help="comma-separated folder numbers, e.g. 01,03")
a = ap.parse_args()
# PowerShell may pass "01,03" as "1,3" or "1 3", so normalise to two-digit folder prefixes
a.models = [f"{int(x):02d}" for x in a.models.replace(",", " ").split()] if a.models else None

items = list(a.domains)
if a.file:
    text = Path(a.file).read_text(encoding="utf-8-sig")
    if text.lstrip().lower().startswith("domain"):
        items += [r["domain"] for r in csv.DictReader(text.splitlines())]
    else:
        items += [ln.strip() for ln in text.splitlines() if ln.strip()]
if not items:
    raise SystemExit("Give domains or --file.")

labels = [label_of(d) for d in items]
valid = [bool(VALID.fullmatch(s)) and len(s) >= 3 for s in labels]
safe = [s if ok else "invalid" for s, ok in zip(labels, valid)]

columns = {}
for folder_name, (_, scorer, info) in load_all(a.models).items():
    if info["quick"]:
        print(f"note {folder_name}: saved model is from a --quick smoke test")
    columns[folder_name[:2]] = scorer(safe)

ids = list(columns)
if not ids:
    raise SystemExit("No saved models matched. Train first, or check --models.")
print(f"\n{'domain':34}" + "".join(f"{i:>7}" for i in ids) + "   votes  verdict")
for k, d in enumerate(items):
    if not valid[k]:
        print(f"{d:34}  invalid label")
        continue
    scores = [float(columns[i][k]) for i in ids]
    votes = sum(s >= 0.5 for s in scores)
    verdict = "DGA" if votes > len(ids) / 2 else "legit"
    print(f"{d:34}" + "".join(f"{s:7.3f}" for s in scores) + f"   {votes}/{len(ids)}    {verdict}")
