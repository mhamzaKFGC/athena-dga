# 03. ATT-CNN-BiLSTM

**Paper:** Ren, Jiang, Wang, Liu. *A DGA domain names detection modeling method based on integrating an
attention mechanism and deep neural network.* Cybersecurity 3 (2020), DOI 10.1186/s42400-020-00046-6.
**Paper's idea (abstract):** a CNN and a BiLSTM extract features and an attention layer weights them; aimed
at wordlist-based DGAs. It reports F1 98.79% for detection.

**This implementation (`model.py`):** character embedding (64), then a 1-D CNN (128 filters, width 3,
ReLU), then a BiLSTM (64 per direction, packed so padding is ignored), then additive attention pooling,
then Dropout, Dense 64 and one logit.

**Approximated:** layer sizes and the attention form (Bahdanau-style additive attention).

`python train.py` writes `results.json` and `weights.pt`.
