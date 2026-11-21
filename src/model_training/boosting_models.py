# Compares only the tree-based models (HistGradientBoosting, Random Forest,
# XGBoost, LightGBM) WITHOUT imputing missing values.
#
# Why no imputation: in this dataset missingness itself is a fraud signal (whole
# blocks of V columns go missing together, identity fields are missing when there
# is no identity record). Median imputation hides that. All four models handle NaN
# natively - at every split they learn which side the missing rows should go to.
#
# Same time-based train/valid/test split as compare_model.py (no shuffling).
#
# Threshold selection (NOT max-F1 - a missed fraud costs more than a false alarm):
#   1. sweep thresholds 0.01..0.99 on VALIDATION only
#   2. alert budget: at most MAX_ALERTS_TEST alerts on the test window, scaled to the
#      validation window by size; keep only thresholds within it
#   3. pick the highest-recall threshold among those (ties -> fewest false alarms)
#   4. if none fits, say so and use the fewest-false-alarms threshold at best recall
#   5. freeze it and apply it ONCE to TEST
# Model selection is by validation PR-AUC; ROC-AUC is reported as a secondary check.
#
# Validation/test probabilities are saved to models/boosting_probas.npz, so the
# threshold step can be re-run without retraining:
#   uv run python -m src.model_training.boosting_models --reuse-probas
# Output is also written to logs/boosting_models_<timestamp>.log.
#
# Run from the project root:  uv run python -m src.model_training.boosting_models

import gc
import sys
import time

import numpy as np

import lightgbm as lgb
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import TRAIN_TRANSACTIONS, TRAIN_IDENTITY
from src.logger import start_logging
from src.data_processing.data_loading import data_loading
from src.model_training.evaluation import (
    metrics_at_threshold, print_report, threshold_sweep, scale_alert_budget,
    choose_threshold, plot_recall_vs_false_alarms,
)
from src.model_training.train_xgboost import MODEL_DIR, time_split, prepare_features, train_model

MAX_ALERTS_TEST = 3000          # most alerts the fraud team can review on the test window
PROBA_FILE = MODEL_DIR / "boosting_probas.npz"


def build_sklearn_models() -> dict:
    """
        sklearn tree models. Both accept NaN directly (Random Forest since sklearn 1.4).
    """
    return {
        'HistGradientBoosting': HistGradientBoostingClassifier(
            max_iter=1000, learning_rate=0.1, max_leaf_nodes=63,
            class_weight='balanced', early_stopping=True, random_state=42,
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, min_samples_leaf=10, max_features='sqrt',
            class_weight='balanced_subsample', n_jobs=8, random_state=42,
        ),
    }


def train_lightgbm(X_train, y_train, X_valid, y_valid) -> lgb.LGBMClassifier:
    """
        LightGBM with the same setup as XGBoost: scale_pos_weight for the imbalance and
        early stopping on validation PR-AUC (average_precision) to pick the number of trees.
    """
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = lgb.LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.05,
        num_leaves=255,
        max_depth=-1,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        min_child_samples=50,
        scale_pos_weight=scale_pos_weight,
        n_jobs=8,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='average_precision',
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(100)],
    )
    print(f"Best iteration: {model.best_iteration_}")
    return model


def train_all() -> tuple[dict, np.ndarray, np.ndarray]:
    """
        Trains the four tree models on the time split and returns
        ({model: (valid_proba, test_proba)}, y_valid, y_test).
    """
    df = data_loading(transaction_data=TRAIN_TRANSACTIONS, identity_data=TRAIN_IDENTITY)
    train_df, valid_df, test_df = time_split(df)
    del df
    gc.collect()
    print(f"Train: {train_df.shape}, fraud {train_df['isFraud'].mean():.4f} | "
          f"Valid: {valid_df.shape}, fraud {valid_df['isFraud'].mean():.4f} | "
          f"Test: {test_df.shape}, fraud {test_df['isFraud'].mean():.4f}")

    X_train, y_train, X_valid, y_valid, X_test, y_test, _ = prepare_features(
        train_df, valid_df, test_df, impute=False,
    )
    del train_df, valid_df, test_df
    gc.collect()
    print(f"X_train: {X_train.shape}, X_valid: {X_valid.shape}, X_test: {X_test.shape}")
    print(f"Missing values kept in X_train: {X_train.isna().mean().mean():.1%} of all cells")

    probas, minutes = {}, {}

    def predict(name, model, start):
        minutes[name] = (time.time() - start) / 60
        probas[name] = (model.predict_proba(X_valid)[:, 1], model.predict_proba(X_test)[:, 1])

    for name, model in build_sklearn_models().items():
        print(f"\n=== {name} ===")
        start = time.time()
        model.fit(X_train, y_train)
        predict(name, model, start)
        del model
        gc.collect()

    print("\n=== LightGBM ===")
    start = time.time()
    predict('LightGBM', train_lightgbm(X_train, y_train, X_valid, y_valid), start)
    gc.collect()

    print("\n=== XGBoost (GPU) ===")
    start = time.time()
    predict('XGBoost (GPU)', train_model(X_train, y_train, X_valid, y_valid), start)

    y_valid, y_test = np.asarray(y_valid), np.asarray(y_test)
    MODEL_DIR.mkdir(exist_ok=True)
    np.savez(PROBA_FILE, names=np.array(list(probas)), y_valid=y_valid, y_test=y_test,
             minutes=np.array([minutes[n] for n in probas]),
             **{f"valid__{n}": v for n, (v, _) in probas.items()},
             **{f"test__{n}": t for n, (_, t) in probas.items()})
    print(f"Saved validation/test probabilities to {PROBA_FILE}")
    return probas, y_valid, y_test


def load_probas() -> tuple[dict, np.ndarray, np.ndarray]:
    data = np.load(PROBA_FILE)
    probas = {n: (data[f"valid__{n}"], data[f"test__{n}"]) for n in data['names']}
    return probas, data['y_valid'], data['y_test']


def select_and_report(probas: dict, y_valid: np.ndarray, y_test: np.ndarray) -> None:
    """
        Steps 1-8: sweep on validation, alert-budget threshold, frozen test evaluation,
        per-model tables, the recall-vs-false-alarms plot and the comparison table.
    """
    max_alerts_valid = scale_alert_budget(MAX_ALERTS_TEST, len(y_valid), len(y_test))
    print(f"\nAlert budget: {MAX_ALERTS_TEST:,} on test ({len(y_test):,} rows) "
          f"-> {max_alerts_valid:,} on validation ({len(y_valid):,} rows)")

    cols = ['threshold', 'recall', 'precision', 'frauds_caught', 'frauds_missed',
            'false_alarms', 'alerts']
    sweeps, chosen, per_model, summary = {}, {}, [], []
    pd.set_option('display.width', 200)

    for name, (valid_proba, test_proba) in probas.items():
        # 1. sweep on validation only
        sweep = threshold_sweep(y_valid, valid_proba)
        sweeps[name] = sweep
        # 2-4. budget filter -> highest recall (fallback if nothing fits)
        best, feasible = choose_threshold(sweep, max_alerts_valid)
        chosen[name] = best
        t = float(best['threshold'])
        # 5. freeze and apply once to test
        test_m = metrics_at_threshold(y_test, test_proba, t)
        test_alerts = test_m['frauds_caught'] + test_m['false_alarms']
        valid_pr_auc = average_precision_score(y_valid, valid_proba)
        valid_roc_auc = roc_auc_score(y_valid, valid_proba)

        print(f"\n=== {name} ===")
        if feasible:
            print(f"Threshold {t:.2f}: highest validation recall with alerts <= {max_alerts_valid:,}")
        else:
            print(f"NO threshold in 0.01-0.99 keeps validation alerts <= {max_alerts_valid:,}. "
                  f"Showing {t:.2f}: fewest false alarms at the best recall (over budget).")

        # 6. per-model table: validation (selection) vs test (frozen threshold)
        table = pd.DataFrame([
            {'split': 'validation', **{c: best[c] for c in cols},
             'alert_budget': max_alerts_valid, 'pr_auc': valid_pr_auc, 'roc_auc': valid_roc_auc},
            {'split': 'test', 'threshold': t, 'recall': test_m['fraud_recall'],
             'precision': test_m['fraud_precision'], 'frauds_caught': test_m['frauds_caught'],
             'frauds_missed': test_m['frauds_missed'], 'false_alarms': test_m['false_alarms'],
             'alerts': test_alerts, 'alert_budget': MAX_ALERTS_TEST,
             'pr_auc': test_m['pr_auc'], 'roc_auc': test_m['roc_auc']},
        ])
        print(table.round(4).to_string(index=False))
        if test_alerts > MAX_ALERTS_TEST:
            print(f"Note: test alerts ({test_alerts:,}) exceed the {MAX_ALERTS_TEST:,} budget "
                  f"at the frozen threshold - not re-tuned on test.")
        print_report(y_test, test_proba, test_m, name=f"{name} Test")
        per_model.append(table.assign(model=name, budget_met_on_valid=feasible))

        summary.append({
            'model': name, 'valid_pr_auc': valid_pr_auc, 'valid_roc_auc': valid_roc_auc,
            'threshold': t, 'budget_met_on_valid': feasible,
            'valid_recall': best['recall'], 'valid_alerts': int(best['alerts']),
            'test_pr_auc': test_m['pr_auc'], 'test_roc_auc': test_m['roc_auc'],
            'test_recall': test_m['fraud_recall'], 'test_precision': test_m['fraud_precision'],
            'test_frauds_caught': test_m['frauds_caught'], 'test_frauds_missed': test_m['frauds_missed'],
            'test_false_alarms': test_m['false_alarms'], 'test_alerts': test_alerts,
        })

    # 7. recall vs false alarms on validation, chosen point marked
    plot_path = MODEL_DIR / "recall_vs_false_alarms_valid.png"
    plot_recall_vs_false_alarms(sweeps, chosen, max_alerts_valid, plot_path)
    print(f"\nSaved plot to {plot_path}")

    # 8. model selection by VALIDATION PR-AUC (ROC-AUC secondary); test shown for reporting
    summary = pd.DataFrame(summary).sort_values('valid_pr_auc', ascending=False)
    print(f"\nModels ranked by validation PR-AUC. Threshold = highest-recall under the "
          f"alert budget on validation, applied once to test.")
    print(summary.round(4).to_string(index=False))
    print(f"Selected model (best validation PR-AUC): {summary.iloc[0]['model']}")

    MODEL_DIR.mkdir(exist_ok=True)
    summary.round(4).to_csv(MODEL_DIR / "boosting_comparison_alert_budget.csv", index=False)
    pd.concat(per_model).round(4).to_csv(MODEL_DIR / "threshold_selection_by_model.csv", index=False)
    pd.concat([s.assign(model=n) for n, s in sweeps.items()]).round(4).to_csv(
        MODEL_DIR / "threshold_sweep_valid.csv", index=False)
    print(f"Saved boosting_comparison_alert_budget.csv, threshold_selection_by_model.csv "
          f"and threshold_sweep_valid.csv to {MODEL_DIR}/")


if __name__ == "__main__":
    start_logging("boosting_models")
    if '--reuse-probas' in sys.argv and PROBA_FILE.exists():
        print(f"Reusing saved probabilities from {PROBA_FILE}")
        probas, y_valid, y_test = load_probas()
    else:
        probas, y_valid, y_test = train_all()
    select_and_report(probas, y_valid, y_test)
