# DGA classifier benchmark: 7 published detectors, one test bench

Seven DGA (domain generation algorithm) detectors from journal papers (2019–2024), re-implemented and
evaluated on the same data, split, seed and metrics.

| Folder | Model | Paper | Type |
|---|---|---|---|
| `01_n-CBDC` | n-CBDC | Xu, Shen, Du, *Computers & Security* 85 (2019) | n-gram + CNN |
| `02_MaldomDetector` | MaldomDetector | Almashhadani et al., *Computers & Security* 93 (2020) | 12 lexical features + random forest |
| `03_ATT-CNN-BiLSTM` | ATT-CNN-BiLSTM | Ren et al., *Cybersecurity* 3 (2020) | CNN + BiLSTM + attention |
| `04_NGram-KL-Jaccard` | n-gram KL/Jaccard | Cucchiarelli et al., *Expert Systems with Applications* 170 (2021) | n-gram statistics + random forest |
| `05_LA-LSTM-Attention` | LA_Bin07 / LA_Mul07 | Tuan, Long, Taniar, *Computers & Security* 113 (2022) | LSTM + attention (binary and family) |
| `06_Integrated-DL` | Integrated DL | Ravi et al., *IEEE Trans. Engineering Management* 70(1) (2023) | CNN ‖ BiLSTM + adversarial training |
| `07_HDDN` | HDDN | Hybrid DGA DefenseNet, *Computers & Security* (2024) | FastText-style subwords + CNN + LSTM |

## Run
```
python compare.py            # train all 7 (CPU, about 1-2 hours) and write results.md
python compare.py --quick    # smoke test, about 2 minutes
python compare.py --report   # rebuild the table from existing results
python 03_ATT-CNN-BiLSTM/train.py   # train one model
```
Needs Python 3.9+, PyTorch, scikit-learn and NumPy.

## Data (`data/`)
- **DGA:** UMUDGA (Zago, Gil Pérez, Martínez Pérez, University of Murcia, 2020; Mendeley Data
  10.17632/y8ph45msv8.1): the 10,000-per-family lists for 50 malware families. 2,000 per family are used.
- **Legit:** Majestic Million top-200k plus the UMUDGA legit list; 100,000 are used.
- **Split:** 80/10/10 stratified by class and family; each name is in exactly one split. It is cached as
  `data/split_*.json`, so every model sees identical rows.
- The classified string is the leftmost DNS label (DGA names often sit under `ddns.net`, `mooo.com`,
  `co.uk`).

## Shared code (`common/`)
`data.py` (dataset), `features.py` (lexical and n-gram features), `layers.py` (attention, CNN, packed LSTM),
`train_eval.py` (training loop, metrics), `sklearn_runner.py` (feature models), `attacks.py` (robustness tests).

## Metrics
Accuracy, precision, recall, F1, ROC-AUC, false-positive rate, recall per DGA family, training time,
inference speed (ms per 1,000 names, including feature extraction) and size. Two robustness tests use
unseen legit test names: **CharBot-style** (2 characters replaced) and **word-concat** (two real names
glued). The share flagged as DGA is reported; higher means more robust.

## Honest limits
- These are **reconstructions from the papers' abstracts**, not the authors' code. Each follows the
  published architecture idea; layer sizes, the exact 12 MaldomDetector features, classifier choices and
  similar details are approximations. Results will not match the papers' reported numbers.
- One dataset and one seed. The papers used different data, so their numbers cannot be compared with these.
- Word-concat names built from real words are genuinely ambiguous, so low detection there is expected.
