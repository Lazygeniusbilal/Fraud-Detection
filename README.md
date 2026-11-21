# Fraud Detection on Card Transactions

This project builds a model that flags likely fraud in card transactions. It uses the
labelled training files from the Kaggle IEEE-CIS Fraud Detection competition: 590,540
transactions over about 182 days, of which about 3.5% are fraud. Everything is evaluated
the way a real fraud team would use the model: train on the past, score the future, and
flag only as many transactions as analysts can review. The final model is LightGBM on the
base features. On a held-out window of the latest 30% of transactions (177,162 rows), it
caught 48% of fraud with 54% precision, while staying under the alert budget (5,491
alerts against a limit of 6,000).

---

## Contents

1. [Key results](#key-results)
2. [Problem and why recall matters](#problem-and-why-recall-matters)
3. [Approach](#approach)
4. [What I tried and what did not work](#what-i-tried-and-what-did-not-work)
5. [Project structure](#project-structure)
6. [How to run it](#how-to-run-it)
7. [Leakage safeguards](#leakage-safeguards)
8. [Limitations](#limitations)
9. [Next steps](#next-steps)
10. [Credits and data source](#credits-and-data-source)
11. [Open questions](#open-questions)

---

## Key results

Final model: **LightGBM, base features only** (no card-history features). Scored once
on the holdout at a threshold frozen before the holdout run.

| Metric | Holdout value |
|---|---|
| Holdout rows (latest 30% by time) | 177,162 |
| Fraud cases in holdout | 6,125 (3.46%) |
| PR-AUC | 0.518 |
| ROC-AUC | 0.886 |
| Threshold (frozen) | 0.30 |
| Recall (fraud caught) | 0.480 (2,941 of 6,125) |
| Precision | 0.536 |
| False alarms | 2,550 |
| Alerts vs budget | 5,491 / 6,000 (within budget) |

All four models on the same holdout, each at its own frozen threshold:

| Model | PR-AUC | ROC-AUC | Threshold | Recall | Precision | Alerts | Within 6,000? |
|---|---|---|---|---|---|---|---|
| **LightGBM** | **0.518** | 0.886 | 0.30 | 0.480 | 0.536 | 5,491 | Yes |
| XGBoost | 0.490 | 0.879 | 0.17 | 0.797 | 0.124 | 39,516 | No |
| HistGradientBoosting | 0.476 | 0.855 | 0.37 | 0.459 | 0.476 | 5,914 | Yes |
| Random Forest | 0.489 | 0.892 | 0.49 | 0.480 | 0.445 | 6,613 | No |

Source: `results/final_holdout/summary.csv`.

**A note on honesty.** The holdout was scored once, for all four models, before the
final model choice was made. So the final choice was not completely blind to the holdout.
LightGBM also ranked first on mean PR-AUC in time-based cross-validation (table below),
so the choice agrees with the CV ranking. The card-history features were tested only in
cross-validation and were never run on the holdout.

Time-based cross-validation on the oldest 70% (4 folds, mean ± sample std):

| Model | PR-AUC | ROC-AUC | Recall | Precision | Folds within budget |
|---|---|---|---|---|---|
| LightGBM | 0.511 ± 0.010 | 0.872 ± 0.014 | 0.464 ± 0.052 | 0.529 ± 0.077 | 2 / 4 |
| XGBoost | 0.506 ± 0.009 | 0.862 ± 0.015 | 0.460 ± 0.049 | 0.551 ± 0.084 | 3 / 4 |
| HistGradientBoosting | 0.476 ± 0.010 | 0.841 ± 0.013 | 0.445 ± 0.049 | 0.513 ± 0.082 | 3 / 4 |
| Random Forest | 0.457 ± 0.035 | 0.867 ± 0.007 | 0.420 ± 0.043 | 0.496 ± 0.112 | 3 / 4 |

Source: `results/time_cv/summary.csv`.

Plots:

- `results/final_holdout/recall_vs_false_alarms_holdout.png`: recall against false
  alarms for each model on the holdout, with the frozen threshold marked.
- `results/time_cv/pr_auc_boxplot.png`: PR-AUC per fold for each model.

---

## Problem and why recall matters

The task is binary classification: given a transaction, estimate the chance that it is
fraud. Only about 1 in 29 transactions is fraud, so accuracy is useless here. A model that
says "not fraud" every time is 96.5% accurate and catches nothing.

**Missed fraud costs more than a false alarm.** A missed fraud (false negative) means a
lost payment, a chargeback and an unhappy customer. A false alarm (false positive) costs an
analyst a few minutes, and maybe a short delay for the customer. So the goal is to catch as
much fraud as possible (high recall), not to balance the two error types evenly.

**Why F1 is not the main metric.** F1 weighs precision and recall equally. Picking the
threshold that gives the best F1 ignores that the two errors have different costs. It
also ignores the one limit that matters in practice: how many alerts the team can
review. Early versions of this project used a best-F1 threshold (see
[What I tried](#what-i-tried-and-what-did-not-work)); it gave lower recall than the
alert-budget rule.

**Why the threshold is set by an alert budget.** A fraud team can only review a fixed
number of cases. So the question is not "what is the best threshold in theory?" but
"given that we can review N alerts, how much fraud can we catch?". This project assumes a
budget of **3,000 alerts per 88,581 transactions** (about 3.4% of traffic). The budget is
scaled by row count to whatever window it is applied to. For the 177,162-row holdout that
is 6,000 alerts.

**Main ranking metric: PR-AUC.** PR-AUC (average precision) summarises the
precision-recall trade-off across all thresholds and is sensitive to performance on the
rare class. ROC-AUC is reported as a secondary check.

---

## Approach

### Data

- `train_transaction.csv` (590,540 rows) left-joined with `train_identity.csv`
  (144,233 rows) on `TransactionID`.
- `TransactionDT` is a time offset in seconds. The data covers roughly day 1 to day 183.
- The Kaggle test files (`test_transaction.csv`, `test_identity.csv`) have no labels and
  are **not used** anywhere in the evaluation.
- Memory is reduced by downcasting numeric types (about 1.78 GB to 0.86 GB for the
  transaction table, from the run logs).

### Preprocessing and features

- Columns with more than 80% missing values are dropped (75 columns), decided on the
  training data only.
- Exact duplicate rows (ignoring `TransactionID`) are dropped from training data (1 row).
- Identity fields are split into parts: device brand and version, OS, browser, screen
  width and height. Rare device brands are grouped into "Others". A `had_id` flag marks
  rows with an identity record.
- Categorical encoding: T/F columns to 1/0, one-hot for `ProductCD`, `card4`, `card6`,
  `M4`, and frequency encoding for `P_emaildomain`.
- **No imputation** for the final models. All four tree models handle missing values
  natively, and missingness itself carries signal (33.3% of all cells in the training
  matrix are missing).
- The final feature matrix has 360 columns.

### Time-based split

The labelled data is sorted by `TransactionDT` and split once:

| Window | Rows | Fraud rate | Days | Used for |
|---|---|---|---|---|
| Oldest 70% | 413,377 (after dedup) | 3.52% | 1.0 to 120.8 | Cross-validation, threshold choice, final training |
| Latest 30% | 177,162 | 3.46% | 120.8 to 183.0 | Holdout, scored once |

Source: `results/final_holdout/fraud_rates.csv`, `final_training_info.json`.

### Models compared

| Model | Class imbalance handling | Early stopping |
|---|---|---|
| LightGBM | `scale_pos_weight` (legit/fraud ratio) | Yes, on PR-AUC, 200 rounds patience |
| XGBoost (GPU) | `scale_pos_weight` | Yes, on PR-AUC, 200 rounds patience |
| HistGradientBoosting | `class_weight='balanced'` | Yes, on a time-based slice |
| Random Forest | `class_weight='balanced_subsample'` | No |

Logistic Regression was also tried in an earlier comparison (`compare_model.py`) and was
far behind (test PR-AUC 0.115), so it was dropped.

### Threshold rule

1. Sweep thresholds from 0.01 to 0.99 in steps of 0.01.
2. Keep only thresholds where alerts (true positives + false positives) fit the budget:
   `floor(3000 × rows / 88,581)`.
3. Among those, take the one with the highest recall. Ties go to fewer false alarms.
4. The threshold is chosen on validation data (never the data it is evaluated on),
   then frozen and applied once.

For the final holdout, the threshold for each model came from pooling the threshold
slices of all four CV folds (116,784 rows, 4,448 frauds). It was written to
`results/final_holdout/frozen_thresholds.json` before the holdout was scored. The script
refuses to overwrite that file.

### Time-based cross-validation

The oldest 70% is cut into 5 consecutive blocks (A to E) of about 82,675 rows each.
Cuts fall on `TransactionDT` values, so rows from the same second are never split.

| Fold | Train on | Validate on | Validation days |
|---|---|---|---|
| 1 | A | B | 20.6 to 40.8 |
| 2 | A+B | C | 40.8 to 68.2 |
| 3 | A+B+C | D | 68.2 to 94.0 |
| 4 | A+B+C+D | E | 94.0 to 120.8 |

Within each fold:

- A **3-day gap** is removed from the end of the training period, so the model never
  trains on transactions right next to the validation block.
- The training period is split by time into **fit (70%)**, **early stopping (15%)** and
  **threshold (latest 15%)** slices.
- Preprocessing (dropped columns, encoders) is fit on the training period only.
- The validation block is scored once at the frozen threshold. Nothing is fit, stopped
  or tuned on it.

Source: `results/time_cv/folds.csv`, `results/time_cv/blocks.csv`.

### Final training

Each model is retrained on the full oldest 70%: fit on the oldest 85% of it (351,370
rows) with early stopping on the newest 15% (62,007 rows). Random Forest has no early
stopping and is fit on all 413,377 rows. The holdout is then scored once.

---

## What I tried and what did not work

This section is the most useful part of the project. Most of the work went into making
the evaluation honest, not into tuning models.

### 1. Random splits vs time-based splits

The first version of this work (`notebooks/fraudclassification.ipynb`, from an earlier
course project) used shuffled 4-fold `KFold` and `train_test_split` on a randomly
undersampled, balanced dataset, with weighted accuracy, precision, recall and F1. Those
numbers look good (for example, weighted F1 around 0.83 for Random Forest). But they
do not measure what matters:

- Random splits put future transactions in the training set. The same cards, devices and
  fraud campaigns appear on both sides, so the model is partly recognising, not
  predicting.
- Balanced undersampling and weighted metrics hide how the model behaves at the real 3.5%
  fraud rate.

All later work uses time-ordered splits only. The notebook numbers are **not comparable**
to the results above, and no same-model random-vs-time experiment was run in the current
code (see [Open questions](#open-questions)).

### 2. F1-based threshold

`train_xgboost.py` and `compare_model.py` choose the threshold that maximises F1 on a
validation window, then score a later test window (time split 70/15/15). Using the same
XGBoost model and the same test window (88,581 rows, 3,083 frauds), the two threshold
rules compare like this:

| Threshold rule | Threshold | Test recall | Test precision | Frauds caught | False alarms |
|---|---|---|---|---|---|
| Best F1 on validation | 0.446 | 0.450 | 0.667 | 1,388 | 692 |
| Highest recall within 3,000 alerts | 0.22 | 0.519 | 0.505 | 1,600 | 1,567 |

Sources: `models/boosting_comparison_no_impute.csv` and
`models/boosting_comparison_alert_budget.csv` (both local, git-ignored). The budget rule
catches 212 more frauds for 875 more false alarms, and still fits the review budget.
Given that a missed fraud costs more than a false alarm, that is the better trade.

An earlier bug is also worth noting (recorded in `notes.md`): the first version chose the
F1 threshold on the validation set and reported precision and recall on that same set,
which inflated them. That was fixed by adding a separate test window.

### 3. Imputation and PCA

- Median imputation plus scaling was replaced by keeping missing values as NaN. On the
  same 70/15/15 split with F1 thresholds, the effect on test PR-AUC was small (XGBoost
  0.529 to 0.533, HistGradientBoosting 0.502 to 0.504, Random Forest 0.496 to 0.493).
  NaN was kept because it is simpler and keeps the "this was missing" signal.
- PCA on the numerical columns was tried and then disabled. The code comment records
  that it reduced 211 columns to 8 and threw away fraud signal. No saved metrics for the
  PCA run were found (see [Open questions](#open-questions)).

### 4. XGBoost score shift between CV and final models

XGBoost did well in CV (PR-AUC 0.506, second to LightGBM). On the holdout its ranking
quality was still reasonable (PR-AUC 0.490), but its frozen threshold of 0.17 flagged
**39,516 alerts against a budget of 6,000**, with precision of only 0.124.

The cause is that the final XGBoost model produced scores on a different scale from the
CV models the threshold was chosen on. I checked this from the saved probabilities:

| Score on legitimate transactions | CV threshold slices (pooled) | Holdout (final model) |
|---|---|---|
| Median | 0.00034 | 0.053 |
| 99th percentile | 0.24 | 0.72 |
| Share above threshold 0.17 | 1.3% | 20.3% |

LightGBM did not have this problem: its legitimate-score median moved from 0.0025 to
0.0027, and the share above its threshold stayed at about 1.5%.

One visible difference is the number of trees. The CV XGBoost models stopped at 1,508 to
2,997 trees. The final model stopped at 443. With `scale_pos_weight` near 28.6, fewer
trees can leave legitimate scores much higher. I have not confirmed this is the full
cause.

Lesson: a threshold is tied to a specific model's score scale. If the model is retrained,
the threshold has to be re-checked on data the new model has not seen, or the scores
need calibrating. Comparing models by PR-AUC alone would have hidden this failure.

### 5. Card-history features (no gain)

I built leakage-safe features that describe each card's past behaviour
(`src/data_processing/history_features.py`), keyed on `card1`:

- **Velocity:** number of transactions in the last 1 hour, 24 hours and 7 days; time
  since the previous transaction; 24-hour spend.
- **Amount vs normal:** amount divided by the card's past mean and past max; z-score
  against the card's past amounts.
- **New-for-card flags:** first time this card is seen with this `addr1`, email domain,
  product code or raw device string.
- **Other:** distinct past email domains and addresses per card; counts for card+address
  and card+email pairs; hour of day; day of week; log amount; cents; number of missing
  fields in the row.

Rules: only rows with a strictly earlier `TransactionDT` count as history, no labels are
used, and nothing is fitted.

**The validation-history issue (v2).** In the first version, validation rows could only
use training-period rows as history. Because of the 3-day gap, every validation row's
recent history was empty. The velocity features looked completely different on
validation than on training data:

| Fold 1 | Threshold slice | Validation (v2) | Validation (v3) |
|---|---|---|---|
| Share with ≥1 transaction in past 1 h | 44.8% | 0.0% | 46.5% |
| Share with ≥1 transaction in past 24 h | 80.2% | 0.0% | 79.4% |
| Median seconds since previous transaction | 4,397 | 1,034,369 | 4,164 |

Source: `results/cv_features_v2/feature_shift.csv`, `results/cv_features_v3/feature_shift.csv`.
The same pattern holds in all four folds.

**The fix (v3).** Validation rows may also use **earlier rows from their own validation
block** as history (transaction fields only, never labels). This matches production,
where you know a card's earlier transactions but not yet whether they were fraud. With
this, the feature distributions on validation match the training data.

The first v3 run failed its own leakage checks (160 future-truncation mismatches and 98
mismatches against the reference implementation, all on `amt_card_zscore`) and stopped.
After fixing the z-score computation, which now uses per-card centred running sums and
treats near-constant past amounts as missing, all checks passed with zero mismatches on
every fold.

**Result: no gain.** Mean CV PR-AUC:

| Model | Base features (v1) | History v2 | History v3 (leakage-safe) |
|---|---|---|---|
| LightGBM | 0.511 | 0.505 | 0.504 |
| XGBoost | 0.506 | 0.492 | 0.504 |

Sources: `results/time_cv/summary.csv`, `results/cv_features_v2/summary.csv`,
`results/cv_features_v3/summary.csv`.

The models did use the new features. In fold 4, LightGBM ranked `amt_card_zscore`,
`card1_email_count`, `amt_to_card_max`, `card1_cnt_7d` and `amt_to_card_mean` in its top
15 by gain. But they did not improve PR-AUC, probably because the existing anonymised
`C` and `D` columns already encode similar counts and time-since information (this is
a guess, not tested). The final model uses the base features only.

### 6. Alert budget overshoot on some folds

The threshold is chosen on one slice and applied to another, so the alert count on the
new data can drift. In CV, every model went over budget on fold 1:

| Model | Fold 1 alerts | Budget | Fold 2 alerts |
|---|---|---|---|
| LightGBM | 3,058 | 2,799 | 2,820 (also over) |
| XGBoost | 2,939 | 2,799 | 2,705 |
| HistGradientBoosting | 3,114 | 2,799 | 2,631 |
| Random Forest | 3,450 | 2,799 | 2,493 |

Reasons visible in the data:

- **Small threshold slice.** In fold 1 the threshold is chosen on only 10,019 rows,
  with a budget of 339 alerts. A threshold step of 0.01 can move the alert count a lot,
  and the estimate is noisy.
- **Different fraud rate.** The fold 1 threshold slice had a 2.49% fraud rate; the
  validation block had 2.84%. More fraud means more true alerts at the same threshold.
  Fold 2 is similar (3.60% vs 4.22%).
- **Score drift.** The share of legitimate transactions above the threshold also rose
  between slices in fold 1 (for example, 1.9% to 2.4% for LightGBM in the v2 run,
  `results/cv_features_v2/score_scale.csv`).

The thresholds were not re-tuned on the validation blocks to hide this. For the final
holdout, thresholds were chosen on the pooled slices from all folds (116,784 rows), which
gives a more stable estimate. LightGBM and HistGradientBoosting then stayed within
budget; Random Forest went over (6,613 vs 6,000).

---

## Project structure

```
fraud_detection/
├── data.zip                        Kaggle IEEE-CIS CSV files, zipped
├── main.py                         Placeholder entry point (prints a greeting only)
├── pyproject.toml                  Project metadata and dependencies (uv)
├── uv.lock                         Locked dependency versions
├── .python-version                 Python 3.12
├── notes.md                        Working notes (threshold leakage fix, HistGB early stopping)
├── notebooks/
│   └── fraudclassification.ipynb   Earlier exploratory notebook (random splits, many models)
├── src/
│   ├── __init__.py                 Paths for data.zip and the unzip folder
│   ├── config.py                   Paths to the unzipped CSV files
│   ├── logger.py                   Copies all printed output to logs/<name>_<time>.log
│   ├── data_processing/
│   │   ├── unzip_data.py           Extracts data.zip into unzip_data/
│   │   ├── data_size_reduction.py  Downcasts dtypes to save memory
│   │   ├── data_loading.py         Loads and joins transaction and identity tables
│   │   ├── data_cleaning.py        Drops high-missing columns and duplicate rows
│   │   ├── feature_engineering.py  Identity feature splits, encoding, fit/transform pipeline
│   │   ├── history_features.py     Leakage-safe card-history features
│   │   └── leakage_checks.py       Reference implementation and leakage tests for history features
│   └── model_training/
│       ├── evaluation.py           Metrics, threshold sweep, alert-budget threshold rule, plots
│       ├── train_xgboost.py        70/15/15 time split, single XGBoost model (F1 threshold)
│       ├── compare_model.py        Early comparison incl. Logistic Regression (imputed features)
│       ├── boosting_models.py      Four tree models, no imputation, alert-budget threshold
│       ├── time_cv.py              4-fold expanding-window time CV on the oldest 70%
│       ├── cv_features_v2.py       Time CV with card-history features (v2 and v3)
│       └── final_holdout.py        Frozen thresholds, final training, one-shot holdout
└── results/
    ├── time_cv/                    CV on the oldest 70% (blocks, folds, per-fold, summary, box plot)
    ├── time_cv_full590k/           Earlier CV run on all 590,540 rows (before the holdout was set aside)
    ├── cv_features_v2/             History features, validation sees training history only
    ├── cv_features_v3/             History features, leakage-safe version, plus leakage check results
    └── final_holdout/              Frozen thresholds, holdout summary, sweep, plot, run output
```

Created when scripts run, and git-ignored:

- `unzip_data/`: extracted CSV files.
- `models/`: saved XGBoost models, preprocessors and early comparison CSVs.
- `logs/`: one log file per script run.

---

## How to run it

### Requirements

- Python 3.12
- An NVIDIA GPU with CUDA for XGBoost. The XGBoost model is hard-coded to
  `device='cuda'`, and there is no CPU switch. LightGBM, HistGradientBoosting and Random
  Forest run on CPU.
- Enough RAM to hold the joined table (about 0.9 GB after downcasting, more during
  feature building).

### Setup with uv (recommended)

```bash
uv sync
```

### Setup with pip

There is no `requirements.txt`. Install the dependencies listed in `pyproject.toml`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install "lightgbm>=4.7.0" "matplotlib>=3.11.2" "numpy>=2.5.4" "pandas>=3.1.1" \
            "scikit-learn>=1.9.1" "xgboost>=3.2,<3.3"
```

With pip, drop the `uv run` prefix from the commands below.

### Get the data

The scripts expect the CSV files in `unzip_data/`. Either extract `data.zip`:

```bash
uv run python -m src.data_processing.unzip_data
```

or download the competition files from Kaggle and place `train_transaction.csv`,
`train_identity.csv`, `test_transaction.csv` and `test_identity.csv` in `unzip_data/`.

### Run the main pipeline

All commands run from the project root. Every script writes a log to `logs/`.

**1. Time-based cross-validation** (four models, oldest 70% only):

```bash
uv run python -m src.model_training.time_cv
```

Rebuild the tables and plot from cached fold probabilities, without retraining:

```bash
uv run python -m src.model_training.time_cv --reuse-probas
```

**2. Final holdout.** Run in order. Step 1 needs the cached CV probabilities from
`time_cv`.

```bash
# Step 1: choose and freeze one threshold per model from pooled CV threshold slices
uv run python -m src.model_training.final_holdout --thresholds

# Step 2+3: train final models on the oldest 70% and score the holdout once
uv run python -m src.model_training.final_holdout --final

# Rebuild the holdout tables from saved holdout probabilities (no retraining)
uv run python -m src.model_training.final_holdout --report
```

`--thresholds` refuses to run if `frozen_thresholds.json` already exists, and `--final`
refuses to run if the holdout has already been scored. This is on purpose. To rerun from
scratch, move `results/final_holdout/` aside first.

**3. Card-history feature experiments** (LightGBM and XGBoost only):

```bash
# v2: validation rows see training-period history only
uv run python -m src.model_training.cv_features_v2

# v3: leakage-safe version, validation rows also see earlier validation rows
uv run python -m src.model_training.cv_features_v2 --history train_and_valid
```

### Earlier scripts (single 70/15/15 time split)

These are kept for reference. They write to `models/`.

```bash
uv run python -m src.model_training.train_xgboost      # one XGBoost model, F1 threshold
uv run python -m src.model_training.compare_model      # adds Logistic Regression, imputed features
uv run python -m src.model_training.boosting_models    # four tree models, alert-budget threshold
uv run python -m src.model_training.boosting_models --reuse-probas
```

### Training time

From `results/final_holdout/final_training_info.json` (final models, one machine):
LightGBM 1.9 min, XGBoost (GPU) 3.4 min, HistGradientBoosting 4.0 min, Random Forest
2.9 min. Data loading and feature building add to this.

---

## Leakage safeguards

| Safeguard | Where | What it does |
|---|---|---|
| Time-ordered splits | `train_xgboost.time_split`, `time_cv.make_blocks` | Training data is always older than evaluation data. |
| Gap before validation | `time_cv.fold_indices` | Drops 3 days of training rows before each validation block. |
| Order asserts | `time_cv.py`, `cv_features_v2.py`, `final_holdout.py` | Assert fit ≤ early-stopping ≤ threshold slice < gap < validation, and training max time ≤ holdout min time. |
| Holdout isolation check | `time_cv.block_table` | Asserts no holdout `TransactionID` appears in any CV block. |
| Fit on training data only | `data_cleaning.clean_data`, `feature_engineering.fit_features` | Dropped columns, encoders and frequency maps come from training rows only. |
| Threshold never chosen on evaluation data | `evaluation.choose_threshold` | Chosen on a separate slice, then applied once. |
| **Label independence** | `leakage_checks.check_label_independence` | Recomputes history features with shuffled labels and with no label column. Output must be identical. |
| **Future truncation** | `leakage_checks.check_future_truncation` | For cut times at the 10th, 50th and 90th percentile of validation time, removes all later rows. Features of earlier rows must not change. |
| **Second implementation** | `leakage_checks.reference_history_features` | A slower, independent version using pandas rolling windows with `closed='left'`. It is compared with the fast version on every row and every history feature. |

The three history-feature checks run inside every fold of the v3 experiment, and the
run stops if any fails. Result: zero mismatches in all four folds (149,467 to 398,592
rows checked per fold, `results/cv_features_v3/leakage_checks.csv`).

**Holdout rule.** The latest 30% is never used for training, early stopping, threshold
choice or feature decisions. Thresholds are frozen to a file before the holdout is
scored, and the script refuses to score the holdout twice. As noted above, the holdout
was scored once for all four models before the final model was chosen.

---

## Limitations

- **Recall is modest.** The final model catches about 48% of fraud on the holdout. More
  than half of fraud (3,184 cases) is missed at this alert budget.
- **The alert budget is a trade-off, not a free lunch.** Raising recall means more
  alerts. The 3,000 per 88,581 rows budget is an assumption, not a number from a real
  fraud team. Different budgets give different thresholds and results.
- **Alert counts drift.** Even with a frozen threshold, alert volume moved over budget in
  some CV folds and for two of four models on the holdout. A deployed model would need
  monitoring and threshold adjustment.
- **Drift risk.** Fraud rates vary by period (2.7% to 4.2% across CV blocks), and fraud
  patterns change. The model was tested on about 2 months of future data only.
- **Threshold depends on score scale.** As the XGBoost case shows, retraining can shift
  scores enough to break a frozen threshold.
- **No deployment.** There is no serving code, API, model registry or monitoring. The
  final LightGBM model is not saved to disk by `final_holdout.py`; only its holdout
  probabilities are.
- **One data source.** All data comes from one anonymised Kaggle dataset. Many columns
  (`C`, `D`, `M`, `V`) have hidden meanings, which limits feature design and
  explanation.
- **Limited tuning.** Model settings were fixed early and not tuned systematically.

---

## Next steps

**Deployment plan**

1. **Save the final model.** Save the trained LightGBM model, the fitted preprocessing
   objects and the frozen threshold together, with a version tag.
2. **Shadow mode.** Score live transactions without acting on the scores. Compare alerts
   with the current process and with confirmed fraud labels as they arrive.
3. **Monitoring.** Track daily alert volume against budget, the score distribution on
   legitimate transactions, input feature drift, and recall and precision once labels
   arrive. Alert when any of these move outside set limits.
4. **Rollback.** Keep the previous model and threshold ready. Roll back if alert volume
   or precision breaks agreed limits.
5. **Threshold re-check on retrain.** Every retrained model gets its threshold re-checked
   on recent unseen data before it goes live (lesson from the XGBoost score shift).

**Better signals**

- Calibrate scores (for example, isotonic or Platt scaling on a time-based slice) so
  thresholds survive retraining.
- Try history keyed on a better user identifier than `card1` alone, such as a
  combination of card, address and email fields.
- Study which fraud is being missed at the current budget, and look for features that
  target those cases.

**More labelled data**

- Longer time spans, to test across more periods and seasons.
- Analyst feedback on alerts, to learn from reviewed cases.

---

## Credits and data source

- Data: [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) on
  Kaggle. Please read and follow the competition's own data rules before using the data.
- Libraries: LightGBM, XGBoost, scikit-learn, pandas, NumPy, Matplotlib.

---

## Open questions

Things that are missing or unclear in the repo:

1. **`data.zip` is committed to git (about 124 MB).** GitHub rejects files over 100 MB,
   so a push will likely fail. Redistributing the Kaggle data may also conflict with the
   competition's data rules. Consider removing it from history and linking to Kaggle.
2. **No same-model random-split experiment.** The random-split results are only in the
   older notebook, with different preprocessing, undersampling and weighted metrics. A
   direct comparison (same model, random vs time split) would make the point stronger.
3. **PCA results not saved.** `models/xgb_fraud_pca.json` exists and the code says PCA
   lost signal, but no metrics for the PCA run were found.
4. **Cause of the XGBoost score shift not confirmed.** The tree count difference (443 vs
   1,508 to 2,997) is a likely factor, but no experiment isolates it.
5. **`results/time_cv_full590k/`** is from an earlier CV run over all 590,540 rows,
   which includes what later became the holdout period. It is not used in the final
   evaluation, but it means that time period had been looked at before the holdout was
   set aside.
6. **The 3,000-alert budget** is not explained anywhere in the code beyond "the alert
   budget the fraud team set". It matches the size of the 88,581-row test window in the
   earlier 70/15/15 split. If it is an assumption, it should be stated as one.
7. **`main.py` is a placeholder** and does not run the pipeline.
8. **`results/` is not yet tracked by git**, and `models/` and `logs/` are git-ignored.
   Links to plots and CSVs in this README only work once `results/` is committed.
9. **`zipfile36` is listed as a dependency** but the code uses the standard library
   `zipfile`. It may not be needed.
10. **No tests.** The leakage checks run inside the experiment script, not as a separate
    test suite.
11. **Gap rows and history features.** In v3, rows inside the 3-day gap are not used as
    history for validation rows. In production they would be. The effect on results was
    not measured.
