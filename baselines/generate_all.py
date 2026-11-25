"""Generate N names from each baseline and save them for the Athena-vs-classifiers comparison.

  python generate_all.py --n 1000 --seed 2026

Writes baselines/out/<name>.txt (one label per line) and prints counts + a few samples. CharBot needs
no weights; the other three require train_all.py to have run first.
"""
import argparse
from pathlib import Path

import charbot
import deepdga
import khaos
import maskdga

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

GENS = {"charbot": charbot.generate, "deepdga": deepdga.generate,
        "maskdga": maskdga.generate, "khaos": khaos.generate}

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--only", nargs="*", default=None)
a = ap.parse_args()
want = a.only or list(GENS)

for name in want:
    try:
        names = GENS[name](n=a.n, seed=a.seed)
        (OUT / f"{name}.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
        sample = ", ".join(names[:8])
        print(f"{name:10s} {len(names):5d} names -> out/{name}.txt   e.g. {sample}", flush=True)
    except FileNotFoundError as e:
        print(f"{name:10s} SKIPPED - {e}", flush=True)
