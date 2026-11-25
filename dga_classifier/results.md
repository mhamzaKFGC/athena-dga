# DGA classifier comparison

Data: full UMUDGA. Same split, same seed, same metrics. Test set = 100,000 held-out DGA names (2,000 per family) + 100,000 held-out legit names.

| # | Model | Type | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | CharBot det. | Word-concat det. | Train s | ms/1k | Params |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 01 | n-CBDC | n-gram + CNN | 0.9577 | 0.9677 | 0.9469 | 0.9572 | 0.9919 | 0.0316 | 0.212 | 0.254 | 3511.4 | 345.28 | 974209 |
| 02 | MaldomDetector | 12 lexical features + random forest | 0.8083 | 0.8825 | 0.7114 | 0.7878 | 0.8894 | 0.0947 | 0.239 | 0.213 | 91.7 | 38.47 | 2693478 |
| 03 | ATT-CNN-BiLSTM | CNN + BiLSTM + attention | 0.9568 | 0.9639 | 0.9490 | 0.9564 | 0.9914 | 0.0355 | 0.201 | 0.242 | 3672.7 | 138.14 | 151489 |
| 04 | NGram-KL-Jaccard | 2/3-gram KL divergence + Jaccard + random forest | 0.9036 | 0.9326 | 0.8701 | 0.9003 | 0.9652 | 0.0629 | 0.263 | 0.278 | 250.3 | 240.92 | 7354596 |
| 05 | LA-LSTM-Attention | LSTM + attention | 0.9520 | 0.9522 | 0.9518 | 0.9520 | 0.9900 | 0.0478 | 0.235 | 0.293 | 2902.9 | 100.36 | 118593 |
| 06 | Integrated-DL | CNN || BiLSTM fusion + adversarial training | 0.9509 | 0.9558 | 0.9456 | 0.9507 | 0.9882 | 0.0437 | 0.611 | 0.297 | 4984.0 | 184.44 | 192449 |
| 07 | HDDN | FastText-style subwords + CNN + LSTM | 0.9663 | 0.9724 | 0.9599 | 0.9661 | 0.9948 | 0.0273 | 0.211 | 0.239 | 5526.5 | 443.48 | 1447681 |

## Hardest DGA families per model (lowest recall)

- **n-CBDC**: nymaim 0.47, symmi 0.81, matsnu 0.83, suppobox_2 0.86, proslikefan 0.87
- **MaldomDetector**: vawtrak_v2 0.00, vawtrak_v3 0.02, suppobox_2 0.05, suppobox_1 0.09, pizd 0.10
- **ATT-CNN-BiLSTM**: nymaim 0.54, symmi 0.78, matsnu 0.86, gozi_rfc4343 0.88, simda 0.88
- **NGram-KL-Jaccard**: suppobox_2 0.12, nymaim 0.21, pizd 0.44, suppobox_1 0.49, matsnu 0.51
- **LA-LSTM-Attention**: nymaim 0.57, symmi 0.81, matsnu 0.84, proslikefan 0.88, gozi_rfc4343 0.89
- **Integrated-DL**: nymaim 0.46, symmi 0.76, matsnu 0.81, gozi_rfc4343 0.87, suppobox_2 0.88
- **HDDN**: nymaim 0.58, symmi 0.81, matsnu 0.87, proslikefan 0.91, kraken_v1 0.92

## Family classification (LA-LSTM-Attention, 51 classes)

accuracy 0.8229, macro-F1 0.6705, weighted-F1 0.8048

CharBot / word-concat det. = share of adversarial names (built from unseen legit test names) flagged as DGA; higher = more robust.
Integrated-DL is the only model trained with CharBot-style adversarial examples, as its paper focuses on adversarial robustness.
