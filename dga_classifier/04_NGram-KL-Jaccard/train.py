"""N-gram KL/Jaccard detector (Cucchiarelli, Morbidoni, Spalazzi, Baldi -
Expert Systems with Applications 170, 2021).

Reconstruction: for each label, its bigram and trigram distributions are compared with the legit and
DGA reference profiles from the training split using Kullback-Leibler divergence and the Jaccard index
(against each profile's most frequent n-grams). The 9 resulting features feed a random forest.
Run: python train.py  (add --quick for a smoke test)"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

from common.features import ngram_kl_jaccard_features  # noqa: E402
from common.sklearn_runner import run_feature_model  # noqa: E402
from common.train_eval import parse_args, setup  # noqa: E402

FEATURES = ["kl2_legit", "kl2_dga", "jaccard2_legit", "jaccard2_dga",
            "kl3_legit", "kl3_dga", "jaccard3_legit", "jaccard3_dga", "length"]

a = parse_args(__doc__)
ds = setup(a)
run_feature_model(ngram_kl_jaccard_features, ds, a, HERE, {
    "model": "NGram-KL-Jaccard", "paper": "Cucchiarelli et al. - Expert Systems with Applications 170 (2021)",
    "type": "2/3-gram KL divergence + Jaccard + random forest"}, FEATURES)
