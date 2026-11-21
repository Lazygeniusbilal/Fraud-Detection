# Trains an XGBoost fraud classifier on the GPU.
# The labelled data (train files only, the Kaggle test files are never used here) is
# split by time (TransactionDT) into three windows:
#   train (oldest 70%)  -> fit the model and every preprocessing step
#   valid (next 15%)    -> early stopping + choosing the decision threshold
#   test  (latest 15%)  -> scored ONCE at the validation threshold, never seen before
# This mirrors a real fraud model that always predicts future transactions.
#
# Run from the project root:  uv run python -m src.model_training.train_xgboost

import gc
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb

from src.config import TRAIN_TRANSACTIONS, TRAIN_IDENTITY
from src.logger import start_logging
from src.data_processing.data_loading import data_loading
from src.data_processing.data_cleaning import clean_data
from src.data_processing.feature_engineering import fit_features, transform_features
from src.model_training.evaluation import compute_metrics, metrics_at_threshold, print_report

MODEL_DIR = Path("models")
VALID_SIZE = 0.15
TEST_SIZE = 0.15


def time_split(df: pd.DataFrame, valid_size: float = VALID_SIZE, test_size: float = TEST_SIZE):
    """
        Sorts by TransactionDT and splits into train / valid / test windows,
        oldest to newest.
    """
    df = df.sort_values('TransactionDT').reset_index(drop=True)
    n = len(df)
    valid_start = int(n * (1 - valid_size - test_size))
    test_start = int(n * (1 - test_size))
    return (df.iloc[:valid_start].reset_index(drop=True),
            df.iloc[valid_start:test_start].reset_index(drop=True),
            df.iloc[test_start:].reset_index(drop=True))


def prepare_features(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame, impute: bool = True):
    """
        Cleans all parts (columns decided on train), fits the feature pipeline on train
        and applies it to valid and test. Returns X/y for each plus the fitted objects.
        impute=False keeps NaN for models that handle missing values natively.
    """
    train_df, valid_df = clean_data(train_df=train_df, test_df=valid_df)
    test_df = test_df[valid_df.columns]  # same columns dropped as valid
    X_train, y_train, fitted = fit_features(train_df, impute=impute)
    X_valid, y_valid = transform_features(valid_df, fitted)
    X_test, y_test = transform_features(test_df, fitted)
    return X_train, y_train, X_valid, y_valid, X_test, y_test, fitted


def train_model(X_train, y_train, X_valid, y_valid) -> xgb.XGBClassifier:
    """
        XGBoost on the GPU. scale_pos_weight (legit/fraud ratio) handles the ~28:1
        class imbalance; early stopping on validation PR-AUC picks the number of trees.
    """
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        device='cuda',
        tree_method='hist',
        n_estimators=3000,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        scale_pos_weight=scale_pos_weight,
        eval_metric='aucpr',
        early_stopping_rounds=200,
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=100)
    print(f"Best iteration: {model.best_iteration}")
    return model


def evaluate(model, X_valid, y_valid, X_test, y_test) -> tuple[dict, dict]:
    """
        Picks the best-F1 threshold on validation only, then scores the test window
        once at that fixed threshold. Returns (valid_metrics, test_metrics).
    """
    valid_proba = model.predict_proba(X_valid)[:, 1]
    valid_metrics = compute_metrics(y_valid, valid_proba)
    print_report(y_valid, valid_proba, valid_metrics, name="Validation")

    test_proba = model.predict_proba(X_test)[:, 1]
    test_metrics = metrics_at_threshold(y_test, test_proba, valid_metrics['threshold'])
    print_report(y_test, test_proba, test_metrics, name="Test")
    return valid_metrics, test_metrics


if __name__ == "__main__":
    start_logging("train_xgboost")
    df = data_loading(transaction_data=TRAIN_TRANSACTIONS, identity_data=TRAIN_IDENTITY)
    train_df, valid_df, test_df = time_split(df)
    del df
    gc.collect()
    print(f"Train: {train_df.shape}, fraud {train_df['isFraud'].mean():.4f} | "
          f"Valid: {valid_df.shape}, fraud {valid_df['isFraud'].mean():.4f} | "
          f"Test: {test_df.shape}, fraud {test_df['isFraud'].mean():.4f}")

    X_train, y_train, X_valid, y_valid, X_test, y_test, preprocessors = prepare_features(train_df, valid_df, test_df)
    del train_df, valid_df, test_df
    gc.collect()
    print(f"X_train: {X_train.shape}, X_valid: {X_valid.shape}, X_test: {X_test.shape}")

    model = train_model(X_train, y_train, X_valid, y_valid)
    valid_metrics, _ = evaluate(model, X_valid, y_valid, X_test, y_test)
    threshold = valid_metrics['threshold']

    MODEL_DIR.mkdir(exist_ok=True)
    model.save_model(MODEL_DIR / "xgb_fraud.json")
    joblib.dump({**preprocessors, 'threshold': threshold}, MODEL_DIR / "preprocessors.joblib")
    print(f"Saved model and preprocessors to {MODEL_DIR}/")
