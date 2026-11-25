# 02. MaldomDetector

**Paper:** Almashhadani, Kaiiali, Carlin, Sezer. *MaldomDetector: A system for detecting algorithmically
generated domain names with machine learning.* Computers & Security 93, 101787 (2020).
**Paper's idea (abstract):** easy-to-compute, language-independent features from the domain name only and
a deterministic algorithm. One record reports 12 features and 97.82% accuracy.

**This implementation (`train.py` + `common/features.py::lexical_features`):** 12 features:
length, Shannon entropy, vowel / consonant / digit / hyphen ratios, longest consonant run, longest digit
run, unique-character ratio, letter↔digit switches, and the share of the name's bigrams and trigrams seen
in legit training names. They feed a random forest (300 trees, fixed seed, so it is deterministic).
The n-gram reference uses one half of the training split and the forest the other half, to avoid leakage.

**Approximated:** the exact 12 features and the paper's classifier are not known from the abstract.

`python train.py` writes `results.json` (including feature importance) and `model.pkl`.
