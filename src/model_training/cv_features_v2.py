# Time-based CV with the v2 card-history features. LightGBM and XGBoost only.
#
# Same data, blocks, folds, slices, model settings, early stopping and threshold rule
# as time_cv.py; the only change is the extra features from history_features.py:
#   - oldest 70% of train_transaction.csv (+ identity), latest 30% never used,
#     Kaggle test files never loaded
#   - 5 time blocks, 4 expanding folds, 3-day gap before each validation block
#   - training period -> fit (70%) / early stopping (15%) / threshold (latest 15%)
#   - threshold: highest recall within 3000 alerts per 88,581 rows, scaled to the
#     threshold slice, then applied ONCE to the validation block
#
# Leakage handling for the new features (per fold):
#   - history = rows with a strictly earlier TransactionDT
#   - only the fold's training period counts as history; validation rows see training
#     rows only (not each other); gap rows are never history
#   - existing encoders are fit on the fit slice, cleaning decided on the training period
#
# No cached scores are read: every probability is regenerated in this run.
#
# Run from the project root:  uv run python -m src.model_training.cv_features_v2

import gc
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.config import TRAIN_IDENTITY
from src.logger import start_logging
from src.data_processing.data_cleaning import find_high_missing_columns, drop_duplicate_rows
from src.data_processing.feature_engineering import fit_features, transform_features
from src.data_processing.history_features import add_history_features, NEW_FEATURES
from src.data_processing.leakage_checks import (
    check_reference, check_label_independence, check_future_truncation,
)
from src.model_training.evaluation import threshold_sweep, choose_threshold, metrics_at_threshold
from src.model_training.time_cv import (
    load_training_window, make_blocks, fold_indices, split_training_period, alert_budget,
    fit_model, BLOCK_NAMES, N_BLOCKS, GAP_DAYS, SECONDS_PER_DAY, MAX_ALERTS, BUDGET_REF_ROWS,
)

# --history train            (v2): validation rows see training-period rows only
# --history train_and_valid  (v3): validation rows also see EARLIER rows of their own
#                                  validation block (features only, never labels)
HISTORY = sys.argv[sys.argv.index('--history') + 1] if '--history' in sys.argv else 'train'
assert HISTORY in ('train', 'train_and_valid'), HISTORY
OUT_DIR = Path("results/cv_features_v2" if HISTORY == 'train' else "results/cv_features_v3")
PROBA_DIR = OUT_DIR / "probas"
MODELS = ['LightGBM', 'XGBoost']
PREVIOUS = {'v1': Path("results/time_cv/summary.csv"),
            'v2': Path("results/cv_features_v2/summary.csv")}


def load_data() -> pd.DataFrame:
    """Oldest-70% window (same as time_cv) + raw DeviceInfo + missing count, sorted by time."""
    window, holdout = load_training_window()
    holdout_start = holdout['TransactionDT'].min()
    del holdout
    # missing values per row, over the loaded columns (before any new feature)
    raw_cols = [c for c in window.columns if c not in ('isFraud', 'TransactionID', 'TransactionDT')]
    window['n_missing'] = window[raw_cols].isna().sum(axis=1).astype(np.float32)
    # data_loading replaces DeviceInfo with grouped brands; the novelty flag needs the raw string
    dev = pd.read_csv(TRAIN_IDENTITY, usecols=['TransactionID', 'DeviceInfo'])
    window = window.merge(dev.rename(columns={'DeviceInfo': 'DeviceInfo_raw'}),
                          on='TransactionID', how='left')
    df = make_blocks(window)
    assert df['TransactionDT'].max() <= holdout_start
    return df


def score_scale(p, y) -> dict:
    legit = p[np.asarray(y) == 0]
    return {'median': float(np.median(legit)), 'p99': float(np.percentile(legit, 99))}


def run() -> None:
    if PROBA_DIR.exists():
        shutil.rmtree(PROBA_DIR)          # never reuse scores from an earlier run
    PROBA_DIR.mkdir(parents=True)

    df = load_data()
    gc.collect()
    rows, scales, importances, shift, checks = [], [], [], [], []

    for k in range(1, N_BLOCKS):
        idx = fold_indices(df, k)
        train_period = df.iloc[idx['train']].drop(columns='block').reset_index(drop=True)
        valid_df = df.iloc[idx['valid']].drop(columns='block').reset_index(drop=True)

        # history features: training period sees itself (strictly earlier rows);
        # validation sees the training period only
        n_tr = len(train_period)
        both = pd.concat([train_period, valid_df], ignore_index=True)
        contributes = np.r_[np.ones(n_tr, bool), np.full(len(valid_df), HISTORY == 'train_and_valid')]
        feats = add_history_features(both, contributes)

        # leakage checks - the run stops here if any fails
        n_label = check_label_independence(both, contributes, feats)
        vdt = valid_df['TransactionDT'].to_numpy()
        cuts = [np.quantile(vdt, q, method='lower') for q in (0.1, 0.5, 0.9)]
        n_trunc = check_future_truncation(both, feats, cuts) if HISTORY == 'train_and_valid' else 0
        ref = check_reference(both, feats) if HISTORY == 'train_and_valid' else None
        n_ref = int(ref['mismatches'].sum()) if ref is not None else 0
        checks.append({'fold': k, 'rows_checked': len(both),
                       'label_independence_mismatches': n_label,
                       'future_truncation_mismatches': n_trunc,
                       'reference_impl_mismatches': n_ref})
        print(f"Leakage checks fold {k}: label {n_label}, truncation {n_trunc}, reference {n_ref}")
        if ref is not None and n_ref:
            print(ref[ref['mismatches'] > 0].to_string(index=False))
        assert n_label == 0 and n_trunc == 0 and n_ref == 0, "leakage check failed"
        both = feats.drop(columns='DeviceInfo_raw')
        del feats
        train_period, valid_df = both.iloc[:n_tr].reset_index(drop=True), both.iloc[n_tr:].reset_index(drop=True)
        del both

        # same cleaning as clean_data (decided on the training period), except the new
        # features are kept even if mostly missing (e.g. card1_new_device is NaN without DeviceInfo)
        drop = [c for c in find_high_missing_columns(train_period) if c not in NEW_FEATURES]
        drop += [c for c in train_period.columns if c not in drop and c not in NEW_FEATURES
                 and c not in ('TransactionID', 'isFraud') and train_period[c].isna().all()]
        train_period = drop_duplicate_rows(train_period.drop(columns=drop))
        valid_df = valid_df.drop(columns=drop)
        fit_df, es_df, thr_df = split_training_period(train_period)
        assert fit_df['TransactionDT'].max() <= es_df['TransactionDT'].min()
        assert es_df['TransactionDT'].max() <= thr_df['TransactionDT'].min()
        assert thr_df['TransactionDT'].max() < valid_df['TransactionDT'].min() - GAP_DAYS * SECONDS_PER_DAY
        missing_new = [f for f in NEW_FEATURES if f not in fit_df.columns]
        print(f"\n######## Fold {k}: {'+'.join(BLOCK_NAMES[:k])} -> {BLOCK_NAMES[k]} | fit {len(fit_df):,}, "
              f"ES {len(es_df):,}, threshold {len(thr_df):,}, valid {len(valid_df):,} | "
              f"new features dropped by cleaning: {missing_new or 'none'}")

        # how different the velocity features look on the threshold slice vs validation
        for name, d in [('threshold_slice', thr_df), ('validation', valid_df)]:
            shift.append({'fold': k, 'slice': name,
                          **{f'share_nonzero_{c}': float((d[c] > 0).mean())
                             for c in ['card1_cnt_1h', 'card1_cnt_24h', 'card1_cnt_7d']},
                          'median_secs_since_prev': float(d['card1_secs_since_prev'].median()),
                          'share_new_card': float(d['card1_secs_since_prev'].isna().mean())})

        X_fit, y_fit, fitted = fit_features(fit_df, impute=False)
        X_es, y_es = transform_features(es_df, fitted)
        X_thr, y_thr = transform_features(thr_df, fitted)
        X_valid, y_valid = transform_features(valid_df, fitted)
        y_thr, y_valid = np.asarray(y_thr), np.asarray(y_valid)
        del train_period, fit_df, es_df, thr_df, valid_df
        gc.collect()

        for name in MODELS:
            print(f"\n=== Fold {k} | {name} === ({X_fit.shape[1]} features)")
            start = time.time()
            model, best_iter = fit_model(name, X_fit, y_fit, X_es, y_es)   # ES on the ES slice
            minutes = (time.time() - start) / 60
            p_thr = model.predict_proba(X_thr)[:, 1]
            p_valid = model.predict_proba(X_valid)[:, 1]
            np.savez(PROBA_DIR / f"fold{k}__{name}.npz", proba_thr=p_thr, y_thr=y_thr,
                     proba=p_valid, y=y_valid, best_iteration=best_iter, minutes=minutes)

            # threshold: chosen on the threshold slice against its scaled budget ...
            thr_budget = alert_budget(len(y_thr))
            best, thr_ok = choose_threshold(threshold_sweep(y_thr, p_thr), thr_budget)
            t = float(best['threshold'])
            # ... applied once to the validation block
            m = metrics_at_threshold(y_valid, p_valid, t)
            budget = alert_budget(len(y_valid))
            alerts = m['frauds_caught'] + m['false_alarms']
            rows.append({
                'model': name, 'fold': k, 'train_blocks': '+'.join(BLOCK_NAMES[:k]),
                'valid_block': BLOCK_NAMES[k], 'n_features': X_fit.shape[1],
                'thr_rows': len(y_thr), 'thr_alert_budget': thr_budget,
                'thr_alerts': int(best['alerts']), 'thr_budget_met': thr_ok,
                'thr_recall': float(best['recall']), 'threshold': t,
                'valid_rows': len(y_valid), 'valid_frauds': int(y_valid.sum()),
                'pr_auc': m['pr_auc'], 'roc_auc': m['roc_auc'],
                'recall': m['fraud_recall'], 'precision': m['fraud_precision'],
                'alert_budget': budget, 'alerts_flagged': alerts, 'budget_met': alerts <= budget,
                'frauds_caught': m['frauds_caught'], 'frauds_missed': m['frauds_missed'],
                'false_alarms': m['false_alarms'],
                'best_iteration': float(best_iter), 'train_minutes': minutes,
            })
            s_thr, s_val = score_scale(p_thr, y_thr), score_scale(p_valid, y_valid)
            scales.append({'model': name, 'fold': k, 'threshold': t,
                           'legit_median_thr_slice': s_thr['median'], 'legit_median_valid': s_val['median'],
                           'legit_p99_thr_slice': s_thr['p99'], 'legit_p99_valid': s_val['p99'],
                           'median_ratio_valid_to_thr': s_val['median'] / s_thr['median'],
                           'p99_ratio_valid_to_thr': s_val['p99'] / s_thr['p99'],
                           'share_legit_above_t_thr_slice': float((p_thr[y_thr == 0] >= t).mean()),
                           'share_legit_above_t_valid': float((p_valid[y_valid == 0] >= t).mean())})
            print(f"Fold {k} {name}: PR-AUC {m['pr_auc']:.4f}, recall {m['fraud_recall']:.4f}, "
                  f"precision {m['fraud_precision']:.4f}, alerts {alerts:,}/{budget:,}, "
                  f"t={t:.2f}, {minutes:.1f} min")

            if k == N_BLOCKS - 1:     # final fold: gain importance
                if name == 'LightGBM':
                    gain = pd.Series(model.booster_.feature_importance('gain'),
                                     index=model.booster_.feature_name())
                else:
                    gain = pd.Series(model.get_booster().get_score(importance_type='total_gain'))
                gain = gain.reindex(X_fit.columns).fillna(0).sort_values(ascending=False)
                imp = pd.DataFrame({'model': name, 'fold': k, 'rank': np.arange(1, len(gain) + 1),
                                    'feature': gain.index, 'total_gain': gain.values,
                                    'gain_share': gain.values / gain.sum(),
                                    'is_new_feature': gain.index.isin(NEW_FEATURES)})
                importances.append(imp)
            del model
            gc.collect()

        del X_fit, y_fit, X_es, y_es, X_thr, X_valid
        gc.collect()

    pd.DataFrame(checks).to_csv(OUT_DIR / "leakage_checks.csv", index=False)
    print(pd.DataFrame(checks).to_string(index=False))
    report(pd.DataFrame(rows), pd.DataFrame(scales), pd.concat(importances), pd.DataFrame(shift))


def report(per_fold, scales, importance, shift) -> None:
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    for name in MODELS:
        t = per_fold[per_fold['model'] == name]
        print(f"\n=== {name}: per fold ===")
        print(t.round(4).to_string(index=False))
        t.round(4).to_csv(OUT_DIR / f"per_fold_{name.lower()}.csv", index=False)

    metrics = ['pr_auc', 'roc_auc', 'recall', 'precision', 'alerts_flagged']
    g = per_fold.groupby('model')
    summary = g[metrics].agg(['mean', 'std'])                      # sample std (ddof=1)
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary['alert_budget_per_fold'] = g['alert_budget'].first()
    summary['max_alerts_flagged'] = g['alerts_flagged'].max()
    summary['folds_budget_met'] = g['budget_met'].sum()
    summary['folds_total'] = g.size()
    for tag, path in PREVIOUS.items():
        if path.parent == OUT_DIR or not path.exists():
            continue
        prev = pd.read_csv(path).set_index('model')
        for m in ['pr_auc_mean', 'roc_auc_mean', 'recall_mean', 'precision_mean', 'alerts_flagged_mean']:
            summary[f'{tag}_{m}'] = prev[m].reindex(summary.index)
            summary[f'delta_vs_{tag}_{m}'] = summary[m] - summary[f'{tag}_{m}']
    summary = summary.reset_index()
    print("\n=== Summary (mean, sample std) vs previous CV (results/time_cv) ===")
    print(summary.round(4).to_string(index=False))
    summary.round(4).to_csv(OUT_DIR / "summary.csv", index=False)

    print("\n=== Score scale on legitimate transactions (threshold slice vs validation) ===")
    print(scales.round(4).to_string(index=False))
    scales.round(5).to_csv(OUT_DIR / "score_scale.csv", index=False)

    print("\n=== Velocity feature shift: threshold slice vs validation ===")
    print(shift.round(4).to_string(index=False))
    shift.round(4).to_csv(OUT_DIR / "feature_shift.csv", index=False)

    importance.round(6).to_csv(OUT_DIR / "feature_importance.csv", index=False)
    for name in MODELS:
        top = importance[(importance['model'] == name) & (importance['rank'] <= 20)]
        print(f"\n=== {name}: top 20 features by total gain (fold {N_BLOCKS - 1}) ===")
        print(top[['rank', 'feature', 'gain_share', 'is_new_feature']].round(4).to_string(index=False))
    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start_logging("cv_features_v2" if HISTORY == "train" else "cv_features_v3")
    run()
