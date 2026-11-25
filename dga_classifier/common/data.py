"""Shared dataset: legit labels vs UMUDGA DGA labels, one fixed split for every model.

Legit  : Majestic top-200k + UMUDGA 'legit' list.
DGA    : UMUDGA 10k-per-family lists (50 families), sampled to `per_family` each.
Label  : the leftmost DNS label (handles DGA names under ddns.net, mooo.com, co.uk, ...).
Split  : 80/10/10, stratified per class/family. A label appears in exactly one class and one split.
The split is cached in data/ so all seven models train and test on identical rows.
"""
import json
import random
import re
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-_"
CHAR2ID = {c: i + 1 for i, c in enumerate(ALPHABET)}  # 0 = padding
VOCAB = len(ALPHABET) + 1
MAXLEN = 48  # 99% of UMUDGA labels are <= 38 characters
VALID = re.compile(r"[a-z0-9_](?:[a-z0-9_-]{0,61}[a-z0-9_])?")


def label_of(domain):
    parts = [p for p in domain.strip().lower().rstrip(".").split(".") if p]
    if parts and parts[0] == "www" and len(parts) > 2:
        parts = parts[1:]
    return parts[0] if parts else ""


def read_labels(path):
    out = []
    for line in Path(path).read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        s = label_of(line)
        if 3 <= len(s) <= 63 and VALID.fullmatch(s):
            out.append(s)
    return out


def load_dataset(per_family=2000, n_legit=100000, seed=42):
    cache = DATA / f"split_pf{per_family}_nl{n_legit}_s{seed}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    rng = random.Random(seed)
    legit = set(read_labels(DATA / "legit_majestic_200k.txt")) | set(read_labels(DATA / "umudga" / "legit.txt"))
    seen = set(legit)  # DGA labels that collide with a legit label are dropped
    groups = {}
    for p in sorted((DATA / "umudga").glob("*.txt")):
        if p.stem == "legit":
            continue
        labs = sorted(set(read_labels(p)) - seen)
        rng.shuffle(labs)
        labs = labs[:per_family]
        seen.update(labs)
        groups[p.stem] = labs
    legit = sorted(legit)
    rng.shuffle(legit)
    groups = {"legit": legit[:n_legit], **groups}

    splits = {"train": [], "val": [], "test": []}
    for fam, labs in groups.items():
        y = 0 if fam == "legit" else 1
        a, b = int(len(labs) * 0.8), int(len(labs) * 0.9)
        for name, part in (("train", labs[:a]), ("val", labs[a:b]), ("test", labs[b:])):
            splits[name] += [[s, y, fam] for s in part]
    for name in splits:
        rng.shuffle(splits[name])
    ds = {"families": [f for f in groups if f != "legit"], **splits}
    cache.write_text(json.dumps(ds), encoding="utf-8")
    return ds


def encode(labels, maxlen=MAXLEN):
    """Character ids, shape (N, maxlen), 0-padded on the right."""
    x = torch.zeros(len(labels), maxlen, dtype=torch.int32)
    for i, s in enumerate(labels):
        ids = [CHAR2ID.get(c, 0) for c in s[:maxlen]]
        x[i, :len(ids)] = torch.tensor(ids, dtype=torch.int32)
    return x
