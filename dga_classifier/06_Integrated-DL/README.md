# 06. Integrated deep learning (adversarial defense)

**Paper:** Ravi, Alazab, Srinivasan, Arunachalam, Soman. *Adversarial Defense: DGA-Based Botnets and DNS
Homographs Detection Through Integrated Deep Learning.* IEEE Transactions on Engineering Management 70(1),
249–266 (2023).
**Paper's idea (abstract):** an integrated deep-learning detector that needs no reverse engineering of the
DGA, tested for robustness against the DeepDGA, CharBot and MaskDGA attacks. It reports accuracy 0.99.

**This implementation (`model.py`):** one character embedding feeds two parallel branches:
- a CNN branch with widths 2–5 (64 filters each, max-pooled)
- a BiLSTM branch (64 per direction, max and mean pooled)

They are fused, then go through Dense 128, Dropout and one logit.
**Adversarial training:** `train.py` adds CharBot-style perturbed *training* legit names (10% of the
training size) as DGA examples. Test-time robustness uses names built from unseen *test* legit names.

**Approximated:** the paper's exact integrated architecture is not described in the abstract. CNN plus
BiLSTM fusion with CharBot-style adversarial training is my reconstruction. DNS homograph detection is not
included.

`python train.py` writes `results.json` and `weights.pt`.
