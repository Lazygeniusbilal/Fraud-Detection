# Notes

## DONE (2026-11-15): Fix threshold leakage in evaluation

Implemented option 1. The labelled train data is now split by time 70/15/15 into
train/valid/test (Kaggle test files untouched). The threshold is chosen on valid, and
metrics are reported on test with `metrics_at_threshold`.

**Problem:** `compute_metrics` (`src/model_training/evaluation.py`) chooses the decision threshold
that gives the best F1 on the validation set. Precision, recall and F1 are then reported on
that same validation set, so they're optimistic. PR-AUC and ROC-AUC don't depend on the
threshold, so they're unaffected.

**Fix options:**
1. **(Preferred) Add a test set.** Split by time into train, validation and test, with test
   as the latest window. Choose the threshold on validation, then report precision, recall
   and F1 on test. This gives honest numbers that are close to production behaviour and fits
   the deploy-and-monitor plan.
2. Report only PR-AUC and ROC-AUC, and drop the threshold-based metrics.

**Places to change:**
- `time_split` in `src/model_training/train_xgboost.py`: return train, valid and test.
- `evaluate` and `train_xgboost.py` main: choose the threshold on validation, then score
  test using that threshold.
- `compare_model.py`: report the test-set metrics for every model.

## Related: HistGradientBoosting early stopping

`HistGradientBoostingClassifier(early_stopping=True)` holds out its own random 10% of the
training data and stops on loss. XGBoost stops early on the time-based validation set using
PR-AUC, so the model comparison isn't exactly like-for-like. Consider
`validation_fraction=None` with a fixed `max_iter`, or `scoring='average_precision'`.
