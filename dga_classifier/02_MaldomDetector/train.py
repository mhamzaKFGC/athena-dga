"""MaldomDetector (Almashhadani, Kaiiali, Carlin, Sezer - Computers & Security 93, 2020).

Reconstruction: 12 easy-to-compute, language-independent lexical features of the domain name,
classified by a random forest (fixed seed, so it is deterministic).
Run: python train.py  (add --quick for a smoke test)"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

from common.features import lexical_features  # noqa: E402
from common.sklearn_runner import run_feature_model  # noqa: E402
from common.train_eval import parse_args, setup  # noqa: E402

FEATURES = ["length", "entropy", "vowel_ratio", "consonant_ratio", "digit_ratio", "hyphen_ratio",
            "longest_consonant_run", "longest_digit_run", "unique_char_ratio", "letter_digit_switches",
            "legit_bigram_share", "legit_trigram_share"]

a = parse_args(__doc__)
ds = setup(a)
run_feature_model(lexical_features, ds, a, HERE, {
    "model": "MaldomDetector", "paper": "Almashhadani et al. - Computers & Security 93 (2020)",
    "type": "12 lexical features + random forest"}, FEATURES)
