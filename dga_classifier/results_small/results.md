# DGA classifier comparison

Same data split, same seed, same metrics. Test set = held-out 10%.

| # | Model | Type | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | CharBot det. | Word-concat det. | Train s | ms/1k | Params |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 01 | n-CBDC | n-gram + CNN | 0.9312 | 0.9354 | 0.9264 | 0.9309 | 0.9813 | 0.0640 | 0.298 | 0.392 | 492.4 | 390.73 | 974209 |
| 02 | MaldomDetector | 12 lexical features + random forest | 0.8425 | 0.9022 | 0.7683 | 0.8299 | 0.9144 | 0.0833 | 0.295 | 0.294 | 15.7 | 57.7 | 2214402 |
| 03 | ATT-CNN-BiLSTM | CNN + BiLSTM + attention | 0.9385 | 0.9504 | 0.9254 | 0.9377 | 0.9852 | 0.0483 | 0.258 | 0.375 | 862.3 | 148.43 | 151489 |
| 04 | NGram-KL-Jaccard | 2/3-gram KL divergence + Jaccard + random forest | 0.9073 | 0.9287 | 0.8823 | 0.9049 | 0.9674 | 0.0677 | 0.261 | 0.344 | 48.2 | 389.99 | 2974012 |
| 05 | LA-LSTM-Attention | LSTM + attention | 0.9307 | 0.9472 | 0.9124 | 0.9295 | 0.9819 | 0.0509 | 0.252 | 0.361 | 685.3 | 113.33 | 118593 |
| 06 | Integrated-DL | CNN || BiLSTM fusion + adversarial training | 0.9229 | 0.9005 | 0.9508 | 0.9250 | 0.9807 | 0.1050 | 0.731 | 0.520 | 1044.3 | 198.0 | 192449 |
| 07 | HDDN | FastText-style subwords + CNN + LSTM | 0.9397 | 0.9309 | 0.9499 | 0.9403 | 0.9852 | 0.0705 | 0.344 | 0.437 | 668.5 | 422.58 | 1447681 |

## Hardest DGA families per model (lowest recall)

- **n-CBDC**: nymaim 0.46, suppobox_2 0.62, matsnu 0.73, simda 0.74, pizd 0.80
- **MaldomDetector**: suppobox_2 0.03, suppobox_1 0.05, pizd 0.06, vawtrak_v2 0.15, nymaim 0.17
- **ATT-CNN-BiLSTM**: nymaim 0.46, matsnu 0.68, gozi_gpl 0.77, gozi_nasa 0.77, suppobox_2 0.78
- **NGram-KL-Jaccard**: suppobox_2 0.11, nymaim 0.26, matsnu 0.36, suppobox_1 0.51, pizd 0.55
- **LA-LSTM-Attention**: nymaim 0.46, suppobox_2 0.66, matsnu 0.72, gozi_nasa 0.74, symmi 0.75
- **Integrated-DL**: nymaim 0.59, suppobox_2 0.76, matsnu 0.82, simda 0.85, symmi 0.88
- **HDDN**: nymaim 0.58, matsnu 0.79, symmi 0.83, simda 0.87, proslikefan 0.88

## Family classification (LA-LSTM-Attention, 51 classes)

accuracy 0.7833, macro-F1 0.5946, weighted-F1 0.7548

CharBot / word-concat det. = share of adversarial names (built from unseen legit test names) flagged as DGA; higher = more robust.
Integrated-DL is the only model trained with CharBot-style adversarial examples, as its paper focuses on adversarial robustness.
