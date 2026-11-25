"""One training loop and one evaluation for every model, so results are directly comparable."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.nn import functional as F

from .attacks import charbot, word_concat
from .data import load_dataset
from .data_full import EpochSampler, load_full


# ---------------------------------------------------------------- setup
def parse_args(description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--quick", action="store_true", help="tiny smoke test (1 epoch, small data)")
    ap.add_argument("--full", action="store_true",
                    help="full UMUDGA: fresh samples from the full pools every epoch (run common.data_full first)")
    ap.add_argument("--epochs", type=int, default=None, help="default 6 (small) or 4 (full)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--per-family", type=int, default=2000, help="small mode: names per family")
    ap.add_argument("--n-legit", type=int, default=100000, help="small mode: legit names")
    ap.add_argument("--epoch-per-family", type=int, default=12000,
                    help="full mode: names drawn per family per epoch (legit: the same total)")
    ap.add_argument("--rf-per-family", type=int, default=10000, help="full mode: training rows per family for the forests")
    ap.add_argument("--holdout", default="", help="small mode: comma-separated families removed from training")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _holdout")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.epochs is None:
        a.epochs = 4 if a.full else 6
    if a.quick:
        a.epochs, a.per_family, a.n_legit, a.epoch_per_family, a.rf_per_family = 1, 100, 5000, 100, 100
    a.holdout = [f for f in a.holdout.split(",") if f]
    return a


def setup(a):
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if a.full:
        split, pools = load_full()
        ds = {"families": split["families"], "val": split["val"], "test": split["test"], "full": True,
              "sampler": EpochSampler(pools, split["families"], a.epoch_per_family, a.seed)}
        if a.quick:  # keep the smoke test small
            ds["val"], ds["test"] = ds["val"][:3000], ds["test"][:6000]
        # one fixed draw for the feature models and for reference statistics
        ds["train"] = EpochSampler(pools, split["families"], a.rf_per_family, a.seed + 1).epoch()
        print(f"FULL mode: {len(ds['families'])} families, {a.epoch_per_family} per family per epoch "
              f"(+ same number of legit), {a.epochs} epochs")
    else:
        ds = load_dataset(a.per_family, a.n_legit, a.seed)
        if a.holdout:
            ds = dict(ds)
            ds["train"] = [r for r in ds["train"] if r[2] not in a.holdout]
            ds["val"] = [r for r in ds["val"] if r[2] not in a.holdout]
            print(f"held-out families (never trained on): {', '.join(a.holdout)}")
    for k in ("train", "val", "test"):
        n_dga = sum(r[1] for r in ds[k])
        print(f"{k}: {len(ds[k])} labels ({n_dga} DGA, {len(ds[k]) - n_dga} legit)")
    return ds


def column(rows, i):
    return [r[i] for r in rows]


# ---------------------------------------------------------------- torch training
def _batches(tensors, idx, batch):
    for i in range(0, len(idx), batch):
        b = idx[i:i + batch]
        parts = [t[b] for t in tensors]
        length = max(int((parts[0] != 0).sum(1).max()), 1)  # trim padding columns
        yield b, [p[:, :length].long() for p in parts]


def _targets(rows, classes):
    if classes is None:
        return torch.tensor([float(r[1]) for r in rows])
    index = {c: i for i, c in enumerate(classes)}
    return torch.tensor([index[r[2]] for r in rows])


def _loss(out, y, multiclass):
    return F.cross_entropy(out, y) if multiclass else F.binary_cross_entropy_with_logits(out, y)


def train_torch(model, ds, a, classes=None, extra_train=None):
    """classes=None -> binary (logit output); classes=list -> multiclass.
    extra_train: rows added to training only - a list, or a function(epoch_rows) -> rows.
    Full mode draws fresh training rows from the full pools every epoch."""
    multiclass = classes is not None

    def training_rows():
        rows = ds["sampler"].epoch() if ds.get("full") else ds["train"]
        extra = extra_train(rows) if callable(extra_train) else (extra_train or [])
        return rows + extra

    train_rows = training_rows()
    xtr, ytr = model.prepare(column(train_rows, 0)), _targets(train_rows, classes)
    xva, yva = model.prepare(column(ds["val"], 0)), _targets(ds["val"], classes)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    best, ckpt, bad, t0 = float("inf"), None, 0, time.time()
    seen = 0
    for ep in range(a.epochs):
        if ds.get("full") and ep > 0:
            train_rows = training_rows()
            xtr, ytr = model.prepare(column(train_rows, 0)), _targets(train_rows, classes)
        seen += len(train_rows)
        model.train()
        for b, parts in _batches(xtr, torch.randperm(len(ytr)), a.batch):
            opt.zero_grad()
            _loss(model(*parts), ytr[b], multiclass).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        total = 0.0
        with torch.no_grad():
            for b, parts in _batches(xva, torch.arange(len(yva)), 1024):
                total += _loss(model(*parts), yva[b], multiclass).item() * len(b)
        vloss = total / len(yva)
        print(f"  epoch {ep + 1}: val loss {vloss:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if vloss < best:
            best, bad = vloss, 0
            ckpt = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 2:
                print("  early stop")
                break
    model.load_state_dict(ckpt)
    model.samples_seen = seen
    return time.time() - t0


@torch.no_grad()
def predict_torch(model, labels, multiclass=False):
    """Binary: P(DGA) per label. Multiclass: class-probability matrix."""
    model.eval()
    x = model.prepare(labels)
    res = None
    for b, parts in _batches(x, torch.arange(len(labels)), 1024):
        out = model(*parts)
        p = out.softmax(-1) if multiclass else torch.sigmoid(out)
        if res is None:
            res = torch.zeros((len(labels),) + tuple(p.shape[1:]))
        res[b] = p
    return res.numpy()


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------- evaluation
def binary_metrics(scores, rows, threshold=0.5):
    y = np.array(column(rows, 1))
    fams = np.array(column(rows, 2))
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    per_family = {f: float(pred[fams == f].mean()) for f in sorted(set(fams)) if f != "legit"}
    return {
        "accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / max(prec + rec, 1e-9), "roc_auc": float(roc_auc_score(y, scores)),
        "false_positive_rate": fp / max(fp + tn, 1),
        "recall_by_family": per_family,
        "worst_families": sorted(per_family, key=per_family.get)[:5],
    }


def multiclass_metrics(probs, rows, classes):
    index = {c: i for i, c in enumerate(classes)}
    y = np.array([index[r[2]] for r in rows])
    pred = probs.argmax(1)
    return {"accuracy": float((pred == y).mean()),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "weighted_f1": float(f1_score(y, pred, average="weighted"))}


def robustness(predict_fn, ds):
    legit_test = [r[0] for r in ds["test"] if r[1] == 0]
    return {"charbot_detection_rate": float((predict_fn(charbot(legit_test)) >= 0.5).mean()),
            "word_concat_detection_rate": float((predict_fn(word_concat(legit_test)) >= 0.5).mean())}


def timed_predict(predict_fn, labels):
    t0 = time.time()
    scores = predict_fn(labels)
    return scores, (time.time() - t0) / len(labels) * 1000 * 1000  # ms per 1000 labels


def save_results(folder, info):
    path = Path(folder) / f"results{info.get('tag', '')}.json"
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    m = info["test"]
    print(f"\n[{info['model']}] test acc {m['accuracy']:.4f}  F1 {m['f1']:.4f}  AUC {m['roc_auc']:.4f}  "
          f"FPR {m['false_positive_rate']:.4f}  | charbot {info['robustness']['charbot_detection_rate']:.3f}  "
          f"word-concat {info['robustness']['word_concat_detection_rate']:.3f}")
    print(f"saved {path}")


def run_torch_binary(model, ds, a, folder, info, extra_train=None):
    """Train, evaluate, test robustness and save - the standard path for neural binary models."""
    train_s = train_torch(model, ds, a, extra_train=extra_train)
    predict = lambda labels: predict_torch(model, labels)
    scores, ms = timed_predict(predict, column(ds["test"], 0))
    info.update(params=n_params(model), train_seconds=round(train_s, 1), ms_per_1000=round(ms, 2),
                training_samples_seen=getattr(model, "samples_seen", None),
                data="full UMUDGA" if ds.get("full") else "UMUDGA 10k lists",
                test=binary_metrics(scores, ds["test"]), robustness=robustness(predict, ds),
                settings=vars(a), tag=a.tag, holdout=a.holdout)
    torch.save(model.state_dict(), Path(folder) / f"weights{a.tag}.pt")
    save_results(folder, info)
    return model
