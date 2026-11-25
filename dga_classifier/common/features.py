"""Hand-made features for the two non-neural models (02 MaldomDetector, 04 n-gram KL/Jaccard).
Reference n-gram statistics are always built from the TRAINING split only."""
import math
from collections import Counter

import numpy as np

VOWELS = set("aeiou")
CONSONANTS = set("bcdfghjklmnpqrstvwxyz")


def ngrams(s, n):
    return [s[i:i + n] for i in range(len(s) - n + 1)]


class NgramReference:
    """Bigram/trigram statistics of legit and DGA training labels."""

    def __init__(self, legit, dga, top=(500, 2000)):
        self.n_values = (2, 3)
        self.counts, self.totals, self.top, self.all = {}, {}, {}, {}
        for cls, labels in (("legit", legit), ("dga", dga)):
            for n, k in zip(self.n_values, top):
                c = Counter(g for s in labels for g in ngrams(s, n))
                self.counts[cls, n] = c
                self.totals[cls, n] = sum(c.values())
                self.top[cls, n] = {g for g, _ in c.most_common(k)}
                self.all[cls, n] = set(c)

    def prob(self, cls, n, g):  # add-one smoothed over the full alphabet space
        return (self.counts[cls, n][g] + 1) / (self.totals[cls, n] + 38 ** n)


def _entropy(s):
    if not s:
        return 0.0
    c = Counter(s)
    return -sum(v / len(s) * math.log2(v / len(s)) for v in c.values())


def _longest_run(s, chars):
    best = cur = 0
    for ch in s:
        cur = cur + 1 if ch in chars else 0
        best = max(best, cur)
    return best


def lexical_features(labels, ref):
    """12 cheap, language-independent features (MaldomDetector-style reconstruction)."""
    digits = set("0123456789")
    rows = []
    for s in labels:
        n = max(len(s), 1)  # XAI may score an empty string (every character removed)
        switches = sum(1 for a, b in zip(s, s[1:]) if (a in digits) != (b in digits))
        bi, tri = ngrams(s, 2), ngrams(s, 3)
        rows.append([
            n,
            _entropy(s),
            sum(ch in VOWELS for ch in s) / n,
            sum(ch in CONSONANTS for ch in s) / n,
            sum(ch in digits for ch in s) / n,
            sum(ch in "-_" for ch in s) / n,
            _longest_run(s, CONSONANTS),
            _longest_run(s, digits),
            len(set(s)) / n,
            switches / n,
            sum(g in ref.all["legit", 2] for g in bi) / max(len(bi), 1),
            sum(g in ref.all["legit", 3] for g in tri) / max(len(tri), 1),
        ])
    return np.asarray(rows, dtype=np.float32)


def ngram_kl_jaccard_features(labels, ref):
    """KL divergence and Jaccard index of each label's 2/3-gram profile vs legit and DGA profiles."""
    rows = []
    for s in labels:
        f = []
        for n in ref.n_values:
            grams = ngrams(s, n)
            if not grams:
                f += [0.0, 0.0, 0.0, 0.0]
                continue
            c = Counter(grams)
            total = len(grams)
            for cls in ("legit", "dga"):
                f.append(sum(v / total * math.log((v / total) / ref.prob(cls, n, g)) for g, v in c.items()))
            gs = set(grams)
            for cls in ("legit", "dga"):
                top = ref.top[cls, n]
                f.append(len(gs & top) / len(gs | top))
        f.append(len(s))
        rows.append(f)
    return np.asarray(rows, dtype=np.float32)
