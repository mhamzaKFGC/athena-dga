"""Simple adversarial DGA tests, generated from TEST-split legit labels (never seen in training).

charbot     : CharBot-style - replace 2 random characters of a real (legit) name.
word_concat : dictionary-DGA style - glue two short real names together.
A robust detector should flag these as DGA. Rates are reported, not used for training."""
import random
import string

POOL = string.ascii_lowercase + string.digits


def charbot(labels, n=2000, k=2, seed=7):
    rng = random.Random(seed)
    base = [s for s in labels if len(s) >= 4]
    out = []
    for _ in range(n):
        s = list(rng.choice(base))
        for i in rng.sample(range(len(s)), k):
            s[i] = rng.choice([c for c in POOL if c != s[i]])
        out.append("".join(s))
    return out


def word_concat(labels, n=2000, seed=7):
    rng = random.Random(seed)
    base = [s for s in labels if 3 <= len(s) <= 8 and s.isalpha()]
    return [rng.choice(base) + rng.choice(base) for _ in range(n)]
