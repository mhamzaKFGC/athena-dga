"""Full-UMUDGA dataset: reads every domain of every family once, deduplicates, and writes
  data/full/split.json        fixed validation + test rows (never used for training)
  data/full/pool_<name>.txt   training pools per family (and pool_legit.txt)
Training then draws fresh random samples from these pools every epoch (see train_eval.py --full).

Legit = UMUDGA legit list + Majestic top-200k. A label that appears as legit is removed from every DGA
family, and a label shared by two families is kept only in the first, so classes never overlap.
Run once:  python -m common.data_full   (from the dga_classifier folder)"""
import json
import random
import time
from pathlib import Path

from .data import DATA, read_labels

FULL_SRC = DATA / "umudga_full"
FULL = DATA / "full"
TEST_PER_FAMILY, VAL_PER_FAMILY = 2000, 500
TEST_LEGIT, VAL_LEGIT = 100000, 25000
POOL_CAP = 300000  # per family; more than any model can consume in the time budget


def build(seed=42):
    t0 = time.time()
    FULL.mkdir(exist_ok=True)
    rng = random.Random(seed)
    legit = set(read_labels(FULL_SRC / "legit.txt")) | set(read_labels(DATA / "legit_majestic_200k.txt"))
    print(f"legit unique: {len(legit)}  ({time.time() - t0:.0f}s)", flush=True)
    seen = set(legit)
    stats, val, test = {}, [], []
    for p in sorted(FULL_SRC.glob("*.txt")):
        if p.stem == "legit":
            continue
        labs = sorted(set(read_labels(p)) - seen)
        seen.update(labs)
        rng.shuffle(labs)
        n_raw = len(labs)
        te, va, tr = labs[:TEST_PER_FAMILY], labs[TEST_PER_FAMILY:TEST_PER_FAMILY + VAL_PER_FAMILY], \
            labs[TEST_PER_FAMILY + VAL_PER_FAMILY:][:POOL_CAP]
        test += [[s, 1, p.stem] for s in te]
        val += [[s, 1, p.stem] for s in va]
        (FULL / f"pool_{p.stem}.txt").write_text("\n".join(tr), encoding="utf-8")
        stats[p.stem] = {"unique": n_raw, "train_pool": len(tr), "val": len(va), "test": len(te)}
        print(f"{p.stem:14} unique {n_raw:>8}  pool {len(tr):>7}  ({time.time() - t0:.0f}s)", flush=True)
    legit = sorted(legit)
    rng.shuffle(legit)
    test += [[s, 0, "legit"] for s in legit[:TEST_LEGIT]]
    val += [[s, 0, "legit"] for s in legit[TEST_LEGIT:TEST_LEGIT + VAL_LEGIT]]
    pool_legit = legit[TEST_LEGIT + VAL_LEGIT:]
    (FULL / "pool_legit.txt").write_text("\n".join(pool_legit), encoding="utf-8")
    stats["legit"] = {"unique": len(legit), "train_pool": len(pool_legit), "val": VAL_LEGIT, "test": TEST_LEGIT}
    rng.shuffle(val)
    rng.shuffle(test)
    families = sorted(k for k in stats if k != "legit")
    (FULL / "split.json").write_text(json.dumps({"families": families, "val": val, "test": test}), encoding="utf-8")
    (FULL / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s: val {len(val)}, test {len(test)}, "
          f"DGA unique total {sum(v['unique'] for k, v in stats.items() if k != 'legit')}")


def load_full():
    split = json.loads((FULL / "split.json").read_text(encoding="utf-8"))
    pools = {p.stem[5:]: p.read_text(encoding="utf-8").split("\n") for p in FULL.glob("pool_*.txt")}
    return split, pools


class EpochSampler:
    """Fresh random training rows each epoch: equal share per DGA family, the same total of legit."""

    def __init__(self, pools, families, per_family, seed):
        self.pools, self.families, self.per_family = pools, families, per_family
        self.rng = random.Random(seed)

    def epoch(self):
        rows = []
        for fam in self.families:
            pool = self.pools[fam]
            k = self.per_family
            picks = self.rng.sample(pool, k) if len(pool) >= k else [self.rng.choice(pool) for _ in range(k)]
            rows += [[s, 1, fam] for s in picks]
        n_legit = len(rows)
        rows += [[s, 0, "legit"] for s in self.rng.sample(self.pools["legit"], min(n_legit, len(self.pools["legit"])))]
        self.rng.shuffle(rows)
        return rows


if __name__ == "__main__":
    build()
