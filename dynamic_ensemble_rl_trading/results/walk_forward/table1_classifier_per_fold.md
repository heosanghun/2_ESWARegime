# Table 1 — Regime Classifier Performance (Walk-Forward)

_Generated: 2026-05-14T21:24:43_

Per-fold metrics on the **forward-looking Trend Scanning ground truth** evaluated on the out-of-sample test window of each fold.

| Fold | Test window | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|-----:|-------------|---------:|------------------:|---------------:|-----------:|
| 1 | 2022-04-19..2022-08-19 | 0.4688 | 0.3302 | 0.3350 | 0.3262 |
| 2 | 2022-08-19..2022-12-19 | 0.4851 | 0.3391 | 0.3507 | 0.3387 |
| 3 | 2022-12-19..2023-04-19 | 0.4700 | 0.3635 | 0.3550 | 0.3499 |
| 4 | 2023-04-19..2023-08-19 | 0.4595 | 0.3770 | 0.3559 | 0.3490 |
| 5 | 2023-08-19..2023-12-19 | 0.4199 | 0.3118 | 0.3170 | 0.3082 |
| **Mean** |  | **0.4607** | **0.3443** | **0.3427** | **0.3344** |

## Per-fold confusion matrices (rows = true, cols = predicted, 0=Bear 1=Sideways 2=Bull)

### Fold 1 — 2022-04-19..2022-08-19

| true \ pred | Bear | Sideways | Bull |
|-----|-----|-----|-----|
| **Bear** | 762 | 5 | 665 |
| **Sideways** | 118 | 1 | 74 |
| **Bull** | 680 | 14 | 610 |

### Fold 2 — 2022-08-19..2022-12-19

| true \ pred | Bear | Sideways | Bull |
|-----|-----|-----|-----|
| **Bear** | 813 | 11 | 642 |
| **Sideways** | 129 | 1 | 102 |
| **Bull** | 614 | 10 | 607 |

### Fold 3 — 2022-12-19..2023-04-19

| true \ pred | Bear | Sideways | Bull |
|-----|-----|-----|-----|
| **Bear** | 699 | 34 | 535 |
| **Sideways** | 129 | 10 | 114 |
| **Bull** | 693 | 34 | 656 |

### Fold 4 — 2023-04-19..2023-08-19

| true \ pred | Bear | Sideways | Bull |
|-----|-----|-----|-----|
| **Bear** | 812 | 26 | 484 |
| **Sideways** | 163 | 17 | 130 |
| **Bull** | 738 | 42 | 517 |

### Fold 5 — 2023-08-19..2023-12-19

| true \ pred | Bear | Sideways | Bull |
|-----|-----|-----|-----|
| **Bear** | 605 | 47 | 564 |
| **Sideways** | 153 | 6 | 122 |
| **Bull** | 776 | 37 | 619 |