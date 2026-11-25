"""Train and evaluate ATT-CNN-BiLSTM. Run: python train.py  (add --quick for a smoke test)"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

from common.train_eval import parse_args, run_torch_binary, setup  # noqa: E402
from model import ATTCNNBiLSTM  # noqa: E402

a = parse_args(__doc__)
ds = setup(a)
run_torch_binary(ATTCNNBiLSTM(), ds, a, HERE, {
    "model": "ATT-CNN-BiLSTM", "paper": "Ren, Jiang, Wang, Liu - Cybersecurity 3 (2020)",
    "type": "CNN + BiLSTM + attention"})
