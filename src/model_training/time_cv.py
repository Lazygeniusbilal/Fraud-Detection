# Forward-chaining (expanding window) time-based cross-validation.
#
# Data used: ONLY the training window of the labelled train files - the oldest 70% by
# TransactionDT from time_split() (413,378 rows, 413,377 after dropping 1 duplicate).
# The latest 30% (the internal valid + test windows, 177,162 rows) is held out and never
# enters any block, and the Kaggle test files are never loaded.
#
# 1. Sort by TransactionDT and cut into 5 consecutive blocks A..E with (almost) equal
#    row counts. Cuts fall on TransactionDT values so rows with the same second are
#    never split across blocks.
# 2. Fold k trains on the blocks before it and validates on the next block:
#       Fold 1: A -> B,  Fold 2: A+B -> C,  Fold 3: A+B+C -> D,  Fold 4: A+B+C+D -> E
#    A GAP_DAYS gap is cut from the END of the training period, so validation blocks
#    stay complete and never overlap.
# 3. Each fold's training period is split by time (oldest -> newest) into:
#       fit       (first 70%)  - the model is fit here (all four models, same rows)
#       es        (next 15%)   - early stopping for XGBoost / LightGBM / HistGB
#       threshold (latest 15%) - the alert-budget threshold is chosen here
#    then the gap, then the validation block. The validation block is scored ONCE at
#    the frozen threshold; nothing is fit, stopped or tuned on it.
# 4. Alert budget: MAX_ALERTS per BUDGET_REF_ROWS rows, scaled to the row count of the
#    slice it is applied to (threshold slice when choosing, validation block when checking).
# 5. Preprocessing (dropped columns, encoders) is fit on the fold's training period only.
#    No imputation: all four tree models handle NaN natively.
#
# Model settings are the existing ones from boosting_models.py / train_xgboost.py; the
# only change is that HistGradientBoosting stops on the time-based ES slice instead of
# its own random 10% split.
#
# Per-fold probabilities are cached in results/time_cv/probas/, so a crashed run resumes
# and the tables/plot can be rebuilt without retraining:
#   uv run python -m src.model_training.time_cv --reuse-probas
#
# Run from the project root:  uv run python -m src.model_training.time_cv

import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import TRAIN_TRANSACTIONS, TRAIN_IDENTITY
from src.logger import start_logging
from src.data_processing.data_loading import data_loading
from src.data_processing.data_cleaning import clean_data, drop_duplicate_rows
from src.data_processing.feature_engineering import fit_features, transform_features
from src.model_training.evaluation import (
    threshold_sweep, choose_threshold, scale_alert_budget, metrics_at_threshold,
)
from src.model_training.boosting_models import build_sklearn_models, train_lightgbm
from src.model_training.train_xgboost import time_split, train_model as train_xgboost

OUT_DIR = Path("results/time_cv")
PROBA_DIR = OUT_DIR / "probas"

N_BLOCKS = 5
BLOCK_NAMES = list("ABCDE")
GAP_DAYS = 3                 # training rows this close to the validation block are dropped
ES_FRACTION = 0.15           # share of the training period used for early stopping
THR_FRACTION = 0.15          # latest share of the training period used to choose the threshold
MAX_ALERTS = 3000            # alert budget the fraud team set ...
BUDGET_REF_ROWS = 88_581     # ... for a window of this many rows
MODELS = ['LightGBM', 'XGBoost', 'HistGradientBoosting', 'Random Forest']
SECONDS_PER_DAY = 86_400


def alert_budget(n_rows: int) -> int:
    """floor(MAX_ALERTS * n_rows / BUDGET_REF_ROWS)"""
    return scale_alert_budget(MAX_ALERTS, n_rows, BUDGET_REF_ROWS)


# ---------------------------------------------------------------------------
# Data, blocks and folds
# ---------------------------------------------------------------------------

def load_training_window() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
        Loads the labelled train files, keeps the oldest 70% (time_split's training
        window) and drops duplicate rows in it. Returns (training window, held-out 30%
        with TransactionID and TransactionDT only, for the leakage check).
    """
    df = data_loading(transaction_data=TRAIN_TRANSACTIONS, identity_data=TRAIN_IDENTITY)
    train_window, valid_window, test_window = time_split(df)
    del df
    holdout = pd.concat([valid_window, test_window])[['TransactionID', 'TransactionDT']]
    del valid_window, test_window
    gc.collect()
    print(f"Training window (oldest 70%): {len(train_window):,} rows; "
          f"held out (latest 30%): {len(holdout):,} rows")
    train_window = drop_duplicate_rows(train_window)
    return train_window, holdout


def make_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """
        Sorts by TransactionDT and adds a `block` column (0..N_BLOCKS-1). Each cut is
        placed at the TransactionDT of the row at that quantile, so ties stay together.
    """
    df = df.sort_values('TransactionDT', kind='stable').reset_index(drop=True)
    dt = df['TransactionDT'].to_numpy()
    cut_values = [dt[int(len(df) * i / N_BLOCKS)] for i in range(1, N_BLOCKS)]
    df['block'] = np.searchsorted(cut_values, dt, side='right').astype(np.int8)
    return df


def block_table(df: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    holdout_ids = set(holdout['TransactionID'])
    holdout_start = holdout['TransactionDT'].min()
    rows = []
    for b, g in df.groupby('block'):
        rows.append({
            'block': BLOCK_NAMES[b], 'rows': len(g), 'frauds': int(g['isFraud'].sum()),
            'fraud_rate': g['isFraud'].mean(),
            'start_day': g['TransactionDT'].min() / SECONDS_PER_DAY,
            'end_day': g['TransactionDT'].max() / SECONDS_PER_DAY,
            'holdout_rows_in_block': int(g['TransactionID'].isin(holdout_ids).sum()),
            'ends_before_holdout': bool(g['TransactionDT'].max() <= holdout_start),
            'source': 'train_transaction.csv (oldest 70%)',
        })
    return pd.DataFrame(rows)


def fold_indices(df: pd.DataFrame, k: int) -> dict:
    """
        Fold k (1..4): training period = blocks 0..k-1 minus the gap, validation = block k.
    """
    valid_idx = np.flatnonzero(df['block'].to_numpy() == k)
    valid_start = df['TransactionDT'].iloc[valid_idx[0]]
    in_train_blocks = df['block'].to_numpy() < k
    before_gap = df['TransactionDT'].to_numpy() < valid_start - GAP_DAYS * SECONDS_PER_DAY
    train_idx = np.flatnonzero(in_train_blocks & before_gap)
    gap_rows = int(in_train_blocks.sum() - len(train_idx))
    return {'train': train_idx, 'valid': valid_idx, 'gap_rows': gap_rows}


def split_training_period(train_period: pd.DataFrame):
    """Time-ordered fit / early-stopping / threshold slices of the training period."""
    n = len(train_period)
    n_thr = int(round(n * THR_FRACTION))
    n_es = int(round(n * ES_FRACTION))
    fit_df = train_period.iloc[:n - n_es - n_thr]
    es_df = train_period.iloc[n - n_es - n_thr:n - n_thr]
    thr_df = train_period.iloc[n - n_thr:]
    return [d.reset_index(drop=True) for d in (fit_df, es_df, thr_df)]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def fit_model(name, X_fit, y_fit, X_es, y_es):
    """Fits one model on the fit slice (early stopping on the ES slice where supported)."""
    if name == 'LightGBM':
        model = train_lightgbm(X_fit, y_fit, X_es, y_es)
        return model, model.best_iteration_
    if name == 'XGBoost':
        model = train_xgboost(X_fit, y_fit, X_es, y_es)
        return model, model.best_iteration
    if name == 'HistGradientBoosting':
        model: HistGradientBoostingClassifier = build_sklearn_models()[name]
        model.fit(X_fit, y_fit, X_val=X_es, y_val=y_es)   # time-based ES, not a random split
        return model, model.n_iter_
    if name == 'Random Forest':
        model: RandomForestClassifier = build_sklearn_models()[name]
        model.fit(X_fit, y_fit)                           # no early stopping
        return model, np.nan
    raise ValueError(name)


def proba_path(k: int, name: str) -> Path:
    return PROBA_DIR / f"fold{k}__{name.replace(' ', '_')}.npz"


def run_folds() -> pd.DataFrame:
    """
        Trains every model on every fold (skipping model-folds already cached) and
        saves threshold-slice and validation-block probabilities per model-fold.
    """
    PROBA_DIR.mkdir(parents=True, exist_ok=True)
    window, holdout = load_training_window()
    df = make_blocks(window)
    del window
    gc.collect()

    blocks = block_table(df, holdout)
    print("\nBlocks (training window only, sorted by TransactionDT):")
    print(blocks.round(4).to_string(index=False))
    assert blocks['holdout_rows_in_block'].sum() == 0 and blocks['ends_before_holdout'].all()
    blocks.round(4).to_csv(OUT_DIR / "blocks.csv", index=False)
    del holdout

    layout = []
    for k in range(1, N_BLOCKS):
        idx = fold_indices(df, k)
        train_period = df.iloc[idx['train']].drop(columns='block').reset_index(drop=True)
        valid_df = df.iloc[idx['valid']].drop(columns='block').reset_index(drop=True)

        # columns decided on the training period only
        train_period, valid_df = clean_data(train_df=train_period, test_df=valid_df)
        fit_df, es_df, thr_df = split_training_period(train_period)

        # strict time order: fit <= es <= thr < gap < valid
        assert fit_df['TransactionDT'].max() <= es_df['TransactionDT'].min()
        assert es_df['TransactionDT'].max() <= thr_df['TransactionDT'].min()
        assert thr_df['TransactionDT'].max() < valid_df['TransactionDT'].min() - GAP_DAYS * SECONDS_PER_DAY

        day = lambda d, f: d['TransactionDT'].agg(f) / SECONDS_PER_DAY
        layout.append({
            'fold': k,
            'train_blocks': '+'.join(BLOCK_NAMES[:k]), 'valid_block': BLOCK_NAMES[k],
            'train_period_rows': len(train_period), 'gap_rows_dropped': idx['gap_rows'],
            'fit_rows': len(fit_df), 'fit_fraud_rate': fit_df['isFraud'].mean(),
            'es_rows': len(es_df), 'es_fraud_rate': es_df['isFraud'].mean(),
            'thr_rows': len(thr_df), 'thr_fraud_rate': thr_df['isFraud'].mean(),
            'thr_alert_budget': alert_budget(len(thr_df)),
            'valid_rows': len(valid_df), 'valid_fraud_rate': valid_df['isFraud'].mean(),
            'valid_alert_budget': alert_budget(len(valid_df)),
            'fit_days': f"{day(fit_df, 'min'):.1f}-{day(fit_df, 'max'):.1f}",
            'es_days': f"{day(es_df, 'min'):.1f}-{day(es_df, 'max'):.1f}",
            'thr_days': f"{day(thr_df, 'min'):.1f}-{day(thr_df, 'max'):.1f}",
            'valid_days': f"{day(valid_df, 'min'):.1f}-{day(valid_df, 'max'):.1f}",
        })
        print(f"\n######## Fold {k}: {layout[-1]['train_blocks']} -> {layout[-1]['valid_block']} | "
              f"fit {len(fit_df):,}, ES {len(es_df):,}, threshold {len(thr_df):,}, "
              f"gap dropped {idx['gap_rows']:,}, valid {len(valid_df):,}")

        todo = [m for m in MODELS if not proba_path(k, m).exists()]
        if not todo:
            print("All models cached for this fold, skipping training.")
            continue

        X_fit, y_fit, fitted = fit_features(fit_df, impute=False)
        X_es, y_es = transform_features(es_df, fitted)
        X_thr, y_thr = transform_features(thr_df, fitted)
        X_valid, y_valid = transform_features(valid_df, fitted)
        del train_period, fit_df, es_df, thr_df, valid_df
        gc.collect()

        for name in todo:
            print(f"\n=== Fold {k} | {name} ===")
            start = time.time()
            model, best_iter = fit_model(name, X_fit, y_fit, X_es, y_es)
            minutes = (time.time() - start) / 60
            p_thr = model.predict_proba(X_thr)[:, 1]
            p_valid = model.predict_proba(X_valid)[:, 1]
            del model
            gc.collect()
            np.savez(proba_path(k, name), proba_thr=p_thr, y_thr=np.asarray(y_thr),
                     proba=p_valid, y=np.asarray(y_valid),
                     best_iteration=best_iter, minutes=minutes)
            print(f"Fold {k} {name}: valid PR-AUC {average_precision_score(y_valid, p_valid):.4f}, "
                  f"{minutes:.1f} min")
        del X_fit, y_fit, X_es, y_es, X_thr, y_thr, X_valid, y_valid
        gc.collect()

    layout = pd.DataFrame(layout)
    layout.round(4).to_csv(OUT_DIR / "folds.csv", index=False)
    return layout


# ---------------------------------------------------------------------------
# Scoring, tables and plot
# ---------------------------------------------------------------------------

def score_folds() -> pd.DataFrame:
    """
        Threshold chosen on the fold's threshold slice (inside its training period),
        then applied once to the validation block.
    """
    rows = []
    for name in MODELS:
        for k in range(1, N_BLOCKS):
            d = np.load(proba_path(k, name))
            # 1. choose on the threshold slice, against that slice's scaled budget
            thr_budget = alert_budget(len(d['y_thr']))
            best, thr_feasible = choose_threshold(threshold_sweep(d['y_thr'], d['proba_thr']), thr_budget)
            t = float(best['threshold'])
            # 2. apply once to the validation block, check against ITS budget (no re-tuning)
            y, proba = d['y'], d['proba']
            m = metrics_at_threshold(y, proba, t)
            budget = alert_budget(len(y))
            alerts = int((proba >= t).sum())
            assert alerts == m['frauds_caught'] + m['false_alarms']
            rows.append({
                'model': name, 'fold': k,
                'train_blocks': '+'.join(BLOCK_NAMES[:k]), 'valid_block': BLOCK_NAMES[k],
                'thr_rows': len(d['y_thr']), 'thr_alert_budget': thr_budget,
                'thr_alerts': int(best['alerts']), 'thr_budget_met': thr_feasible,
                'thr_recall': float(best['recall']),
                'threshold': t,
                'valid_rows': len(y), 'valid_frauds': int(y.sum()),
                'pr_auc': m['pr_auc'], 'roc_auc': m['roc_auc'],
                'alert_budget_exact': MAX_ALERTS * len(y) / BUDGET_REF_ROWS,
                'alert_budget': budget, 'alerts_flagged': alerts,
                'budget_met': alerts <= budget,
                'recall': m['fraud_recall'], 'precision': m['fraud_precision'],
                'frauds_caught': m['frauds_caught'], 'frauds_missed': m['frauds_missed'],
                'false_alarms': m['false_alarms'],
                'best_iteration': float(d['best_iteration']),
                'train_minutes': float(d['minutes']),
            })
    return pd.DataFrame(rows)


def summarise(per_fold: pd.DataFrame) -> pd.DataFrame:
    metrics = ['pr_auc', 'roc_auc', 'recall', 'precision', 'alerts_flagged']
    g = per_fold.groupby('model')
    agg = g[metrics].agg(['mean', 'std'])   # std: sample (ddof=1)
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg['alert_budget_per_fold'] = g['alert_budget'].first()
    agg['max_alerts_flagged'] = g['alerts_flagged'].max()
    agg['folds_budget_met'] = g['budget_met'].sum()
    agg['folds_total'] = g.size()
    return agg.reset_index().sort_values('pr_auc_mean', ascending=False)


def plot_pr_auc(per_fold: pd.DataFrame, path: Path) -> None:
    """
        Box plot of per-fold PR-AUC for each model, with the four fold values drawn on
        top (with 4 folds the box is mostly a summary of these points).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ink, muted, grid, box, dot = '#0b0b0b', '#52514e', '#e4e3df', '#c9d9f0', '#2a78d6'
    order = per_fold.groupby('model')['pr_auc'].mean().sort_values(ascending=False).index.tolist()
    data = [per_fold.loc[per_fold['model'] == m].sort_values('fold') for m in order]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.boxplot([d['pr_auc'] for d in data], tick_labels=order, widths=0.5, patch_artist=True,
               showfliers=False,
               boxprops=dict(facecolor=box, edgecolor=muted, lw=1),
               medianprops=dict(color=ink, lw=2), whiskerprops=dict(color=muted, lw=1),
               capprops=dict(color=muted, lw=1))
    for i, d in enumerate(data, start=1):
        x = i + np.linspace(-0.12, 0.12, len(d))
        ax.scatter(x, d['pr_auc'], s=40, color=dot, edgecolor='white', linewidth=1.5, zorder=3)
        for xi, (_, r) in zip(x, d.iterrows()):
            ax.annotate(f"F{int(r['fold'])}", (xi, r['pr_auc']), xytext=(7, -3),
                        textcoords='offset points', fontsize=8, color=muted)
        ax.annotate(f"mean {d['pr_auc'].mean():.3f}", (i, d['pr_auc'].max()), xytext=(0, 10),
                    textcoords='offset points', ha='center', fontsize=9, color=ink)

    ax.margins(y=0.12)
    ax.set_ylabel('PR-AUC on validation block', color=muted)
    ax.set_title('PR-AUC across 4 forward-chaining folds (oldest 70% of train files)',
                 loc='left', fontsize=12, color=ink, pad=12)
    ax.grid(True, axis='y', color=grid, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=muted)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(grid)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report(layout: pd.DataFrame | None) -> None:
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    if layout is not None:
        print("\nFold layout:")
        print(layout.round(4).to_string(index=False))

    per_fold = score_folds()
    cols = ['fold', 'train_blocks', 'valid_block',
            'thr_rows', 'thr_alert_budget', 'thr_alerts', 'thr_budget_met', 'threshold',
            'valid_rows', 'pr_auc', 'roc_auc', 'alert_budget_exact', 'alert_budget',
            'alerts_flagged', 'budget_met', 'recall', 'precision',
            'frauds_caught', 'frauds_missed', 'false_alarms', 'best_iteration', 'train_minutes']
    for name in MODELS:
        t = per_fold.loc[per_fold['model'] == name, cols]
        print(f"\n=== {name}: per fold (threshold chosen on threshold slice, applied once to validation) ===")
        print(t.round(4).to_string(index=False))
        t.round(4).to_csv(OUT_DIR / f"per_fold_{name.replace(' ', '_')}.csv", index=False)
    per_fold.round(4).to_csv(OUT_DIR / "per_fold_all_models.csv", index=False)

    summary = summarise(per_fold)
    print("\n=== Summary across folds (mean and sample std, ranked by mean PR-AUC) ===")
    print(summary.round(4).to_string(index=False))
    summary.round(4).to_csv(OUT_DIR / "summary.csv", index=False)

    plot_pr_auc(per_fold, OUT_DIR / "pr_auc_boxplot.png")
    print(f"\nSaved CSVs and pr_auc_boxplot.png to {OUT_DIR}/")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start_logging("time_cv")
    cached = all(proba_path(k, m).exists() for k in range(1, N_BLOCKS) for m in MODELS)
    if '--reuse-probas' in sys.argv and cached:
        print("All fold probabilities cached; rebuilding tables only.")
        layout = pd.read_csv(OUT_DIR / "folds.csv") if (OUT_DIR / "folds.csv").exists() else None
    else:
        layout = run_folds()   # resumes from any cached model-folds
    report(layout)
