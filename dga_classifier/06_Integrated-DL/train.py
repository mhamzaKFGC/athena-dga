"""Train and evaluate the integrated CNN+BiLSTM detector with adversarial training.
Run: python train.py  (add --quick for a smoke test, --full for full UMUDGA)"""
import itertools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

from common.attacks import charbot  # noqa: E402
from common.train_eval import parse_args, run_torch_binary, setup  # noqa: E402
from model import IntegratedDL  # noqa: E402

a = parse_args(__doc__)
ds = setup(a)
seeds = itertools.count(123)


def adversarial(rows):
    """CharBot-style perturbations of this epoch's TRAINING legit names (10% of the epoch size)."""
    legit = [r[0] for r in rows if r[1] == 0]
    return [[s, 1, "adv_charbot"] for s in charbot(legit, n=len(rows) // 10, seed=next(seeds))]


run_torch_binary(IntegratedDL(), ds, a, HERE, {
    "model": "Integrated-DL", "type": "CNN || BiLSTM fusion + adversarial training",
    "paper": "Ravi et al. - IEEE Transactions on Engineering Management 70(1) (2023)"}, extra_train=adversarial)
