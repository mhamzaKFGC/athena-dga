# 04. N-gram KL / Jaccard

**Paper:** Cucchiarelli, Morbidoni, Spalazzi, Baldi. *Algorithmically generated malicious domain names
detection based on n-grams features.* Expert Systems with Applications 170, 114551 (2021).
**Paper's idea (abstract):** lexical features from 2-grams and 3-grams, measured with Kullback-Leibler
divergence and the Jaccard index against benign and malicious names, fed to a classifier. It reports
promising zero-day results.

**This implementation (`train.py` + `common/features.py::ngram_kl_jaccard_features`):** for n = 2 and 3,
the KL divergence from the name's n-gram distribution to the legit and DGA reference distributions
(add-one smoothing), and the Jaccard index with each profile's top n-grams (500 bigrams, 2,000 trigrams),
plus length. That makes 9 features, fed to a random forest. The references come from one half of the
training split and the forest is fitted on the other half.

**Approximated:** the classifier choice, profile sizes, smoothing, and the use of one DGA profile rather
than per-family profiles.

`python train.py` writes `results.json` and `model.pkl`.
