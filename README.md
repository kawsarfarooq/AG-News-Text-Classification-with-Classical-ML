# AG News Text Classification

## Overview

Multi-class classification of 127,600 news articles into 4 categories (World, Sports, Business, Sci/Tech) using TF-IDF features and three classical ML algorithms, with parameter sweeps and per-class error analysis.

- **Dataset:** AG News — https://huggingface.co/datasets/sh0416/ag_news
- **Reference paper:** Zhang, X., Zhao, J., & LeCun, Y. (2015). *Character-level Convolutional Networks for Text Classification.* NeurIPS 2015.

## Course requirement check

| Requirement | This project |
|---|---|
| N ≥ 20,000 | N = 127,600 (120,000 train + 7,600 test) |
| p ≥ 4 | p ≈ 200,000 TF-IDF features (7 raw columns incl. derived length features) |
| n_A ≥ 3 | Multinomial Naive Bayes, Logistic Regression, Linear SVM |
| N·p ≥ 1,000,000 | ≈ 2.5 × 10¹⁰ |

## Repository structure

```
.
├── ag_news_classification.ipynb   # main (and only) notebook — self-contained
├── requirements.txt
└── README.md
```

## How to run

The notebook is **self-contained**: it downloads the dataset itself (HuggingFace primary, GitHub CSV mirror as automatic fallback) and requires no auxiliary files.

```powershell
# Windows / PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook ag_news_classification.ipynb
```

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook ag_news_classification.ipynb
```

Then *Run All*. Full end-to-end runtime is roughly 5–10 minutes on a modern laptop (the Logistic Regression sweep is the slowest part).

## Results (test set, after parameter tuning on a validation split)

| Model | Accuracy | Macro-F1 | Fit time |
|---|---|---|---|
| Linear SVM (C=0.5) | **0.9280** | 0.9279 | ~2 s |
| Logistic Regression (C=5) | 0.9243 | 0.9242 | ~10 s |
| Multinomial NB (α=0.01) | 0.9136 | 0.9132 | <1 s |

Dominant confusion: Business ↔ Sci/Tech; Sports is nearly perfectly separated.

## Notebook contents

The notebook opens with an abstract, a table of contents, a requirements/setup section
(pinned library versions + how to run), and a methodology overview, followed by the
numbered analysis:

1. Data loading (auto-download) + derived raw features
2. Exploratory analysis: class distribution, length statistics by class, top TF-IDF terms per class
3. TF-IDF feature extraction (unigrams + bigrams, 200k features)
4. Baseline runs of the 3 algorithms
5. Hyperparameter tuning: n-gram range, NB smoothing α, regularization C (validation split, test set untouched)
6. Final test-set evaluation: summary table, confusion matrices, per-class report
7. Discussion of results
8. Conclusion

It closes with a References section.
