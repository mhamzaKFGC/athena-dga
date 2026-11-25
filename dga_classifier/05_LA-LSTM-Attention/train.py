"""Train and evaluate LA_Bin07 (binary) and LA_Mul07 (family classification).
Run: python train.py  (add --quick for a smoke test, --full for full UMUDGA)"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

from common.train_eval import (column, multiclass_metrics, parse_args, predict_torch,  # noqa: E402
                               run_torch_binary, setup, train_torch)
from model import LALSTM  # noqa: E402

a = parse_args(__doc__)
ds = setup(a)
info = {"model": "LA-LSTM-Attention", "paper": "Tuan, Long, Taniar - Computers & Security 113 (2022)",
        "type": "LSTM + attention"}

print("\n== LA_Bin07 (binary)")
run_torch_binary(LALSTM(1), ds, a, HERE, info)

if not a.holdout:  # family classification is meaningless when families are hidden from training
    print("\n== LA_Mul07 (legit + each DGA family)")
    classes = ["legit"] + ds["families"]
    torch.manual_seed(a.seed)
    mul = LALSTM(len(classes))
    secs = train_torch(mul, ds, a, classes=classes)
    probs = predict_torch(mul, column(ds["test"], 0), multiclass=True)
    info["multiclass"] = {"classes": len(classes), "train_seconds": round(secs, 1),
                          **multiclass_metrics(probs, ds["test"], classes)}
    torch.save(mul.state_dict(), HERE / f"weights_multiclass{a.tag}.pt")
    print(f"[LA_Mul07] accuracy {info['multiclass']['accuracy']:.4f}  macro-F1 {info['multiclass']['macro_f1']:.4f}")
    (HERE / f"results{a.tag}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
