# Final, one-shot holdout evaluation.
#
# Data: the labelled train files only (train_transaction + the train_identity left join);
# the Kaggle test files are never loaded.
#
# Step 1  (--thresholds)
#   For each model, the threshold slices (the latest 15% of each fold's training period)
#   from the 4-fold time CV on the oldest 70% are POOLED into one slice. The cached
#   fold probabilities in results/time_cv/probas/ are used. On the pooled slice we take
#   the highest-recall threshold (0.01..0.99) with alerts <= floor(3000 * rows / 88,581).
#   That gives one threshold per model, written to results/final_holdout/frozen_thresholds.json.
#   The file is never overwritten.
#
# Step 2+3  (--final)
#   Each model is trained on the oldest 70% (413,377 rows after dedup): fit on the oldest
#   85% of that window, early stopping on the newest 15% (Random Forest has no early stopping
#   and is fit on all 413,377 rows). The latest 30% (177,162 rows) is scored ONCE at the
#   frozen thresholds, with an alert budget of 6,000. Nothing is tuned on the holdout.
#
# Run from the project root:
#   uv run python -m src.model_training.final_holdout --thresholds
#   uv run python -m src.model_training.final_holdout --final

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import TRAIN_TRANSACTIONS, TRAIN_IDENTITY
from src.logger import start_logging
from src.data_processing.data_loading import data_loading
from src.data_processing.data_cleaning import clean_data, drop_duplicate_rows
from src.data_processing.feature_engineering import fit_features, transform_features
from src.model_training.evaluation import threshold_sweep, choose_threshold, metrics_at_threshold
from src.model_training.train_xgboost import time_split
from src.model_training.time_cv import (
    MODELS, N_BLOCKS, MAX_ALERTS, BUDGET_REF_ROWS, ES_FRACTION, alert_budget, proba_path, fit_model,
)

OUT_DIR = Path("results/final_holdout")
FROZEN_FILE = OUT_DIR / "frozen_thresholds.json"
HOLDOUT_PROBA_FILE = OUT_DIR / "holdout_probas.npz"
MIN_POOLED_FRAUDS = 500
EXPECTED_TRAIN_ROWS = 413_377
EXPECTED_HOLDOUT_ROWS = 177_162


def slug(name: str) -> str:
    return name.replace(' ', '_')


# ---------------------------------------------------------------------------
# Step 1: pooled threshold selection
# ---------------------------------------------------------------------------

def step1_thresholds() -> dict:
    if FROZEN_FILE.exists():
        raise SystemExit(f"{FROZEN_FILE} already exists - thresholds are frozen, not recomputing.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frozen, sweeps = {}, []
    for name in MODELS:
        ys, ps, per_fold = [], [], []
        for k in range(1, N_BLOCKS):
            d = np.load(proba_path(k, name))
            ys.append(d['y_thr']), ps.append(d['proba_thr'])
            per_fold.append({'fold': k, 'rows': len(d['y_thr']), 'frauds': int(d['y_thr'].sum())})
        y, p = np.concatenate(ys), np.concatenate(ps)
        n, n_fraud = len(y), int(y.sum())
        budget = alert_budget(n)
        sweep = threshold_sweep(y, p)
        best, feasible = choose_threshold(sweep, budget)
        sweeps.append(sweep.assign(model=name))

        frozen[name] = {
            'threshold': float(best['threshold']),
            'pooled_slice_rows': n,
            'pooled_slice_frauds': n_fraud,
            'pooled_slice_fraud_rate': n_fraud / n,
            'pooled_alert_budget_exact': MAX_ALERTS * n / BUDGET_REF_ROWS,
            'pooled_alert_budget': budget,
            'pooled_alerts': int(best['alerts']),
            'pooled_recall': float(best['recall']),
            'pooled_precision': float(best['precision']),
            'budget_met_on_pooled_slice': feasible,
            'fewer_than_500_frauds': n_fraud < MIN_POOLED_FRAUDS,
            'per_fold_slices': per_fold,
        }
        f = frozen[name]
        print(f"\n=== {name} ===")
        print(f"Pooled threshold slice: {n:,} rows, {n_fraud:,} frauds ({n_fraud / n:.4f}); "
              f"per fold rows {[x['rows'] for x in per_fold]}")
        if f['fewer_than_500_frauds']:
            print(f"WARNING: pooled slice has fewer than {MIN_POOLED_FRAUDS} frauds.")
        status = "within" if feasible else "NO threshold within"
        print(f"Budget {budget:,} alerts ({f['pooled_alert_budget_exact']:.1f} exact): {status} budget. "
              f"Frozen threshold {f['threshold']:.2f} -> {f['pooled_alerts']:,} alerts, "
              f"recall {f['pooled_recall']:.4f}, precision {f['pooled_precision']:.4f}")

    with open(FROZEN_FILE, 'x') as fh:          # 'x': fail rather than overwrite
        json.dump(frozen, fh, indent=2)
    pd.concat(sweeps).round(4).to_csv(OUT_DIR / "pooled_threshold_sweep.csv", index=False)
    print(f"\nFrozen thresholds written to {FROZEN_FILE}")
    return frozen


# ---------------------------------------------------------------------------
# Step 2: final models, Step 3: one holdout run
# ---------------------------------------------------------------------------

def load_windows():
    df = data_loading(transaction_data=TRAIN_TRANSACTIONS, identity_data=TRAIN_IDENTITY)
    train_window, valid_window, test_window = time_split(df)
    del df
    holdout = pd.concat([valid_window, test_window]).reset_index(drop=True)
    del valid_window, test_window
    gc.collect()
    train_window = drop_duplicate_rows(train_window)
    assert len(train_window) == EXPECTED_TRAIN_ROWS, len(train_window)
    assert len(holdout) == EXPECTED_HOLDOUT_ROWS, len(holdout)
    assert train_window['TransactionDT'].max() <= holdout['TransactionDT'].min()
    return train_window, holdout


def step2_3_train_and_score(frozen: dict) -> tuple[dict, np.ndarray, dict]:
    train_window, holdout = load_windows()
    # columns decided on the 70% training window only
    train_window, holdout = clean_data(train_df=train_window, test_df=holdout)

    n = len(train_window)
    n_es = int(round(n * ES_FRACTION))
    fit_df = train_window.iloc[:n - n_es].reset_index(drop=True)
    es_df = train_window.iloc[n - n_es:].reset_index(drop=True)
    assert fit_df['TransactionDT'].max() <= es_df['TransactionDT'].min()

    info = {
        'train_rows': n, 'train_frauds': int(train_window['isFraud'].sum()),
        'train_fraud_rate': float(train_window['isFraud'].mean()),
        'fit_rows': len(fit_df), 'es_rows': len(es_df),
        'holdout_rows': len(holdout), 'holdout_frauds': int(holdout['isFraud'].sum()),
        'holdout_fraud_rate': float(holdout['isFraud'].mean()),
        'train_days': [float(train_window['TransactionDT'].min() / 86400), float(train_window['TransactionDT'].max() / 86400)],
        'holdout_days': [float(holdout['TransactionDT'].min() / 86400), float(holdout['TransactionDT'].max() / 86400)],
    }
    print(f"\nFit {len(fit_df):,} | ES {len(es_df):,} | holdout {len(holdout):,}")

    # Features fit on the fit slice (as in every CV fold); RF gets its own fit on all 70%.
    X_fit, y_fit, fitted = fit_features(fit_df, impute=False)
    X_es, y_es = transform_features(es_df, fitted)
    X_hold, y_hold = transform_features(holdout, fitted)
    y_hold = np.asarray(y_hold)

    probas, train_meta = {}, {}
    for name in MODELS:
        if name == 'Random Forest':
            continue
        print(f"\n=== Final {name} (fit {len(X_fit):,}, ES {len(X_es):,}) ===")
        start = time.time()
        model, best_iter = fit_model(name, X_fit, y_fit, X_es, y_es)
        probas[name] = model.predict_proba(X_hold)[:, 1]
        train_meta[name] = {'best_iteration': float(best_iter), 'fit_rows': len(X_fit),
                            'es_rows': len(X_es), 'minutes': (time.time() - start) / 60}
        del model
        gc.collect()
    del X_fit, y_fit, X_es, y_es, X_hold
    gc.collect()

    # Random Forest: no early stopping, so it uses all 413,377 rows.
    print(f"\n=== Final Random Forest (fit {n:,}, no early stopping) ===")
    X_all, y_all, fitted_all = fit_features(train_window, impute=False)
    X_hold_rf, _ = transform_features(holdout, fitted_all)
    del train_window, fit_df, es_df, holdout
    gc.collect()
    start = time.time()
    model, _ = fit_model('Random Forest', X_all, y_all, None, None)
    probas['Random Forest'] = model.predict_proba(X_hold_rf)[:, 1]
    train_meta['Random Forest'] = {'best_iteration': np.nan, 'fit_rows': len(X_all), 'es_rows': 0,
                                   'minutes': (time.time() - start) / 60}
    del model, X_all, y_all, X_hold_rf
    gc.collect()

    np.savez(HOLDOUT_PROBA_FILE, y=y_hold, **{slug(m): p for m, p in probas.items()})
    with open(OUT_DIR / "final_training_info.json", 'w') as fh:
        json.dump({'data': info, 'models': train_meta}, fh, indent=2, default=float)
    return probas, y_hold, info


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(frozen: dict, probas: dict, y: np.ndarray, info: dict) -> pd.DataFrame:
    budget = alert_budget(len(y))
    assert budget == 6000, budget
    rows, sweeps, points = [], {}, {}
    for name in MODELS:
        f, p = frozen[name], probas[name]
        t = f['threshold']
        m = metrics_at_threshold(y, p, t)
        alerts = m['frauds_caught'] + m['false_alarms']
        rows.append({
            'model': name, 'threshold': t,
            'pooled_slice_rows': f['pooled_slice_rows'], 'pooled_slice_frauds': f['pooled_slice_frauds'],
            'pooled_alert_budget': f['pooled_alert_budget'], 'pooled_alerts': f['pooled_alerts'],
            'pooled_recall': f['pooled_recall'],
            'holdout_rows': len(y), 'holdout_frauds': int(y.sum()),
            'pr_auc': m['pr_auc'], 'roc_auc': m['roc_auc'],
            'recall': m['fraud_recall'], 'precision': m['fraud_precision'],
            'frauds_caught': m['frauds_caught'], 'frauds_missed': m['frauds_missed'],
            'false_alarms': m['false_alarms'], 'total_alerts': alerts,
            'alert_budget': budget, 'within_budget': alerts <= budget,
        })
        sweeps[name] = threshold_sweep(y, p)
        points[name] = rows[-1]

    summary = pd.DataFrame(rows)
    summary.round(4).to_csv(OUT_DIR / "summary.csv", index=False)
    for r in rows:
        pd.DataFrame([r]).round(4).to_csv(OUT_DIR / f"per_model_{slug(r['model'])}.csv", index=False)
    pd.concat([s.assign(model=n) for n, s in sweeps.items()]).round(4).to_csv(
        OUT_DIR / "holdout_threshold_sweep.csv", index=False)

    rates = pd.DataFrame([
        {'window': 'training 70%', 'rows': info['train_rows'], 'frauds': info['train_frauds'],
         'fraud_rate': info['train_fraud_rate']},
        {'window': 'holdout 30%', 'rows': info['holdout_rows'], 'frauds': info['holdout_frauds'],
         'fraud_rate': info['holdout_fraud_rate']},
    ])
    rates.round(5).to_csv(OUT_DIR / "fraud_rates.csv", index=False)

    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    print("\n=== Holdout (scored once at frozen thresholds) ===")
    print(summary.round(4).to_string(index=False))
    print("\n=== Fraud rate ===")
    print(rates.round(5).to_string(index=False))

    plot(sweeps, points, budget, OUT_DIR / "recall_vs_false_alarms_holdout.png")
    print(f"\nSaved outputs to {OUT_DIR}/")
    return summary


def plot(sweeps: dict, points: dict, budget: int, path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    line, ink, muted, grid = '#2a78d6', '#0b0b0b', '#52514e', '#e4e3df'
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, MODELS):
        s = sweeps[name].sort_values('false_alarms')
        x = s['false_alarms'].clip(lower=1)
        ok = s['alerts'] <= budget
        ax.plot(x, s['recall'], color=line, lw=2, alpha=0.3, label=f'alerts > {budget:,}')
        ax.plot(x[ok], s['recall'][ok], color=line, lw=2, label=f'alerts ≤ {budget:,}')
        c = points[name]
        ax.scatter([max(c['false_alarms'], 1)], [c['recall']], s=70, color=ink, edgecolor='white',
                   linewidth=2, zorder=5, label='frozen threshold')
        ax.annotate(f"t = {c['threshold']:.2f} (frozen)\nrecall {c['recall']:.3f}\n"
                    f"{c['false_alarms']:,} false alarms\n{c['total_alerts']:,} alerts",
                    (max(c['false_alarms'], 1), c['recall']), xytext=(10, -58),
                    textcoords='offset points', fontsize=9, color=ink)
        ax.set_xscale('log')
        ax.set_ylim(0, 1)
        ax.set_title(f"{name}  ·  PR-AUC {c['pr_auc']:.3f}", fontsize=11, loc='left', color=ink)
        ax.grid(True, color=grid, lw=0.8)
        ax.tick_params(colors=muted)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(grid)
    for ax in axes[1, :]:
        ax.set_xlabel('False alarms on holdout (log scale)', color=muted)
    for ax in axes[:, 0]:
        ax.set_ylabel('Recall (fraud caught)', color=muted)
    axes.flat[0].legend(loc='upper left', fontsize=9, frameon=False)
    fig.suptitle('Holdout (latest 30%, 177,162 rows): recall vs false alarms, thresholds 0.01–0.99',
                 x=0.01, ha='left', fontsize=13, color=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if '--thresholds' in sys.argv:
        start_logging("final_holdout_step1")
        step1_thresholds()
    elif '--final' in sys.argv:
        start_logging("final_holdout")
        if not FROZEN_FILE.exists():
            raise SystemExit("Run --thresholds first.")
        if HOLDOUT_PROBA_FILE.exists():
            raise SystemExit(f"{HOLDOUT_PROBA_FILE} exists - the holdout was already scored once. "
                             f"Use --report to rebuild tables from it.")
        frozen = json.loads(FROZEN_FILE.read_text())
        probas, y, info = step2_3_train_and_score(frozen)
        report(frozen, probas, y, info)
    elif '--report' in sys.argv:
        start_logging("final_holdout_report")
        frozen = json.loads(FROZEN_FILE.read_text())
        d = np.load(HOLDOUT_PROBA_FILE)
        info = json.loads((OUT_DIR / "final_training_info.json").read_text())['data']
        report(frozen, {m: d[slug(m)] for m in MODELS}, d['y'], info)
    else:
        raise SystemExit("Pass --thresholds, --final or --report.")
