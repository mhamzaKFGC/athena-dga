# 05. LA_Bin07 / LA_Mul07

**Paper:** Tuan, Long, Taniar. *On detecting and classifying DGA botnets and their families.* Computers &
Security 113, 102549 (2022).
**Paper's idea (abstract):** an LSTM with an attention layer, in a binary model (LA_Bin07) and a
multiclass family model (LA_Mul07), evaluated on UMUDGA (50 families), the same dataset used here.

**This implementation (`model.py`):** character embedding (64), then an LSTM (128, packed), then additive
attention, then Dropout and the output. `train.py` trains both:
- **LA_Bin07:** one logit (DGA or legit). This is the comparison-table row.
- **LA_Mul07:** 51 outputs (legit plus 50 families). Its accuracy and macro-F1 are in `results.json`
  under `multiclass`.

**Approximated:** layer sizes and attention form.

`python train.py` writes `results.json`, `weights.pt` and `weights_multiclass.pt`.
