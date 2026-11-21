# In this file we will do data cleaning ----
# The columns to drop are decided on the TRAIN data only and then the same
# columns are dropped from TEST, so both end up with matching columns.

import pandas as pd

# columns that must never be dropped
PROTECTED_COLUMNS = ['TransactionID', 'isFraud']


def find_high_missing_columns(df: pd.DataFrame, threshold: float = 0.8) -> list[str]:
    """
        Returns the columns where more than `threshold` (default 80%) of values are missing.
    """
    missing_ratio = df.isnull().mean()
    return [c for c in missing_ratio[missing_ratio > threshold].index if c not in PROTECTED_COLUMNS]


def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
        Drops rows that are identical in every column except TransactionID
        (TransactionID is unique per row, so including it would never find duplicates).
    """
    subset = [c for c in df.columns if c != 'TransactionID']
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep='first').reset_index(drop=True)
    print(f"Duplicate rows dropped: {before - len(df)}")
    return df


def clean_data(train_df: pd.DataFrame, test_df: pd.DataFrame, missing_threshold: float = 0.8):
    """
        Steps (columns decided on train, then the same ones dropped from test):
          1. columns with more than `missing_threshold` missing
          2. columns that are completely missing
          3. duplicate rows (train only)
    """
    # 1. columns with more than 80% missing
    high_missing = find_high_missing_columns(train_df, missing_threshold)
    print(f"Columns with more than {missing_threshold:.0%} missing: {len(high_missing)}")
    train_df = train_df.drop(columns=high_missing)

    # 2. columns that are completely missing
    all_missing = [c for c in train_df.columns if train_df[c].isnull().all() and c not in PROTECTED_COLUMNS]
    print(f"Completely missing columns: {len(all_missing)}")
    train_df = train_df.drop(columns=all_missing)

    # 3. rows are only dropped from train: test needs every row for the submission
    train_df = drop_duplicate_rows(train_df)

    test_df = test_df.drop(columns=high_missing + all_missing, errors='ignore')
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df
