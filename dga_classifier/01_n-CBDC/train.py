"""Train and evaluate n-CBDC. Run: python train.py  (add --quick for a smoke test)"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

from common.train_eval import parse_args, run_torch_binary, setup  # noqa: E402
from model import NCBDC  # noqa: E402

a = parse_args(__doc__)
ds = setup(a)
run_torch_binary(NCBDC(), ds, a, HERE, {
    "model": "n-CBDC", "paper": "Xu, Shen, Du - Computers & Security 85 (2019)",
    "type": "n-gram + CNN"})
