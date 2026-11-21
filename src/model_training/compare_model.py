# Trains several models on the same time-based train/valid/test split and the same
# features, then compares them on fraud-focused metrics (PR-AUC first).
# For each model the threshold is chosen on VALIDATION only, and the reported
# precision / recall / F1 / confusion matrix come from the untouched TEST window.
# Only the labelled train files are used; the Kaggle test files are not touched.
# Every model handles the class imbalance with class weights.
# Output is also written to logs/compare_model_<timestamp>.log.
#
# Run from the project root:  uv run python -m src.model_training.compare_model

import gc
import time

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import TRAIN_TRANSACTIONS, TRAIN_IDENTITY
from src.logger import start_logging
from src.data_processing.data_loading import data_loading
from src.model_training.evaluation import compute_metrics, metrics_at_threshold, print_report
from src.model_training.train_xgboost import MODEL_DIR, time_split, prepare_features, train_model


def build_models() -> dict:
    """
        The candidate models. XGBoost is trained separately through train_model()
        because it uses the validation set for early stopping.
    """
    return {
        'Logistic Regression': LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=42,
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, min_samples_leaf=10, max_features='sqrt',
            class_weight='balanced_subsample', n_jobs=8, random_state=42,
        ),
        'HistGradientBoosting': HistGradientBoostingClassifier(
            max_iter=1000, learning_rate=0.1, max_leaf_nodes=63,
            class_weight='balanced', early_stopping=True, random_state=42,
        ),
    }


if __name__ == "__main__":
    start_logging("compare_model")
    df = data_loading(transaction_data=TRAIN_TRANSACTIONS, identity_data=TRAIN_IDENTITY)
    train_df, valid_df, test_df = time_split(df)
    del df
    gc.collect()
    print(f"Train: {train_df.shape}, fraud {train_df['isFraud'].mean():.4f} | "
          f"Valid: {valid_df.shape}, fraud {valid_df['isFraud'].mean():.4f} | "
          f"Test: {test_df.shape}, fraud {test_df['isFraud'].mean():.4f}")

    X_train, y_train, X_valid, y_valid, X_test, y_test, _ = prepare_features(train_df, valid_df, test_df)
    del train_df, valid_df, test_df
    gc.collect()
    print(f"X_train: {X_train.shape}, X_valid: {X_valid.shape}, X_test: {X_test.shape}")

    def score(name, model, train_minutes):
        """Threshold from validation, metrics from test."""
        valid_m = compute_metrics(y_valid, model.predict_proba(X_valid)[:, 1])
        test_proba = model.predict_proba(X_test)[:, 1]
        test_m = metrics_at_threshold(y_test, test_proba, valid_m['threshold'])
        print_report(y_test, test_proba, test_m, name=f"{name} Test")
        return {'model': name, 'valid_pr_auc': valid_m['pr_auc'], **test_m, 'train_minutes': train_minutes}

    results = []
    for name, model in build_models().items():
        print(f"\n=== {name} ===")
        start = time.time()
        model.fit(X_train, y_train)
        results.append(score(name, model, (time.time() - start) / 60))
        del model
        gc.collect()

    print("\n=== XGBoost (GPU) ===")
    start = time.time()
    xgb_model = train_model(X_train, y_train, X_valid, y_valid)
    results.append(score('XGBoost (GPU)', xgb_model, (time.time() - start) / 60))

    print("\nAll metrics below are on the TEST window (threshold chosen on validation).")
    table = pd.DataFrame(results).sort_values('pr_auc', ascending=False).round(4)
    pd.set_option('display.width', 200)
    print("\n" + table.to_string(index=False))

    MODEL_DIR.mkdir(exist_ok=True)
    table.to_csv(MODEL_DIR / "model_comparison.csv", index=False)
    print(f"Saved comparison to {MODEL_DIR}/model_comparison.csv")
