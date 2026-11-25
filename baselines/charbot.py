"""CharBot (Peck, Nie, De Cock et al., IEEE Access 2019 / arXiv:1905.01078).

"CharBot: A Simple and Effective Method for Evading DGA Classifiers."

Method, verbatim from the paper: take a benign domain and randomly modify TWO characters of it,
without touching the TLD, keeping only names that are not already registered. No model, no knowledge
of the target classifier. The paper's point is that this trivial attack already defeats
state-of-the-art lexical detectors and that retraining on CharBot output is not a viable defense.

Here the "benign domain" is a Majestic-200k label; the two changed positions are in that label (we
work on labels, not full domains, so the TLD is untouched by construction). Novelty is enforced by
common_gen.collect (not registered / not a 1-edit lookalike of a known legit name).
"""
from common_gen import LETTERS, collect, legit_labels


def _make(rng, base_pool, k=2):
    base = rng.choice(base_pool)
    s = list(base)
    positions = rng.sample(range(len(s)), min(k, len(s)))
    for i in positions:
        s[i] = rng.choice([c for c in LETTERS if c != s[i]])
    return "".join(s)


def generate(n=1000, seed=2026, k=2, min_len=4):
    """n CharBot names: each is a Majestic label with `k` random characters changed."""
    pool = [s for s in legit_labels() if len(s) >= min_len]
    return collect(lambda rng: _make(rng, pool, k), n, seed)


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    for name in generate(n):
        print(name)
