# 07. HDDN (Hybrid DGA DefenseNet)

**Paper:** *A hybrid DGA DefenseNet for detecting DGA domain names based on FastText and deep learning
techniques.* Computers & Security (2024). Authors not confirmed.
**Paper's idea (abstract):** FastText word embeddings with CNN and LSTM layers, for detection and family
classification.

**This implementation (`model.py`):** FastText-style input built in PyTorch. Each character position gets
its character embedding plus the average embedding of the hashed character n-grams (n = 3, 4, 5, using
FastText's `<` `>` boundary markers) starting there. Then a 1-D CNN (128 filters), an LSTM (128, packed),
max pooling, Dense 64 and one logit.

**Approximated:** the paper uses pretrained FastText embeddings; here the subword embeddings are learned
during training. Layer sizes are also my choice. Family classification is not included (model 05 covers it).

`python train.py` writes `results.json` and `weights.pt`.
