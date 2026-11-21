# Shared evaluation helpers so every model is scored the same way.

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    classification_report, confusion_matrix,
)


def compute_metrics(y_true, proba) -> dict:
    """
        Scores fraud probabilities: PR-AUC, ROC-AUC, plus fraud-class precision,
        recall and F1 at the threshold that maximises F1.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best = int(np.argmax(f1[:-1]))
    threshold = float(thresholds[best])
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()

    return {
        'pr_auc': average_precision_score(y_true, proba),
        'roc_auc': roc_auc_score(y_true, proba),
        'threshold': threshold,
        'fraud_precision': float(precision[best]),
        'fraud_recall': float(recall[best]),
        'fraud_f1': float(f1[best]),
        'frauds_caught': int(tp),
        'frauds_missed': int(fn),
        'false_alarms': int(fp),
    }


def metrics_at_threshold(y_true, proba, threshold: float) -> dict:
    """
        Scores probabilities at a FIXED threshold (chosen elsewhere, e.g. on validation).
        Use this on the test window so the reported numbers are not tuned to it.
    """
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        'pr_auc': average_precision_score(y_true, proba),
        'roc_auc': roc_auc_score(y_true, proba),
        'threshold': float(threshold),
        'fraud_precision': float(precision),
        'fraud_recall': float(recall),
        'fraud_f1': float(f1),
        'frauds_caught': int(tp),
        'frauds_missed': int(fn),
        'false_alarms': int(fp),
        'true_negatives': int(tn),
    }


def print_report(y_true, proba, metrics: dict, name: str = "Validation") -> None:
    """
        Prints the metrics from compute_metrics() / metrics_at_threshold() plus the
        confusion matrix and per-class report at the given threshold.
    """
    print(f"\n{name} PR-AUC : {metrics['pr_auc']:.4f}")
    print(f"{name} ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Threshold: {metrics['threshold']:.4f} "
          f"(precision {metrics['fraud_precision']:.3f}, recall {metrics['fraud_recall']:.3f})")
    pred = (proba >= metrics['threshold']).astype(int)
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_true, pred, labels=[0, 1]))
    print(classification_report(y_true, pred, target_names=['legit', 'fraud'], digits=4))


# ---------------------------------------------------------------------------
# Threshold selection under an alert budget (recall first, not F1).
# Missed fraud costs more than a false alarm, so we take the threshold with the
# highest recall whose alert volume (flagged transactions = TP + FP) fits the
# budget the fraud team can review. Chosen on VALIDATION only, then frozen.
# ---------------------------------------------------------------------------

THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)   # 0.01 ... 0.99


def threshold_sweep(y_true, proba, thresholds=THRESHOLDS) -> pd.DataFrame:
    """
        Confusion counts, precision and recall at every threshold (flag if proba >= t).
    """
    y_true = np.asarray(y_true).astype(bool)
    proba = np.asarray(proba)
    n_fraud = int(y_true.sum())
    rows = []
    for t in thresholds:
        flagged = proba >= t
        tp = int((flagged & y_true).sum())
        fp = int((flagged & ~y_true).sum())
        alerts = tp + fp
        rows.append({
            'threshold': float(t),
            'precision': tp / alerts if alerts else 0.0,
            'recall': tp / n_fraud if n_fraud else 0.0,
            'frauds_caught': tp,
            'frauds_missed': n_fraud - tp,
            'false_alarms': fp,
            'alerts': alerts,
        })
    return pd.DataFrame(rows)


def scale_alert_budget(max_alerts_test: int, n_valid: int, n_test: int) -> int:
    """
        Scales the test-set alert budget to the validation set by relative size
        (rounded down, so the validation limit is never looser than the test one).
    """
    return int(np.floor(max_alerts_test * n_valid / n_test))


def choose_threshold(sweep: pd.DataFrame, max_alerts: int) -> tuple[pd.Series, bool]:
    """
        Keeps thresholds with alerts <= max_alerts and returns the one with the highest
        recall (fewest frauds missed); ties go to the fewest false alarms.
        If no threshold fits the budget, returns the threshold with the fewest false
        alarms at the best recall over the whole sweep, and feasible=False.
    """
    feasible = sweep[sweep['alerts'] <= max_alerts]
    pool = feasible if len(feasible) else sweep
    best = pool.sort_values(['recall', 'false_alarms'], ascending=[False, True]).iloc[0]
    return best, bool(len(feasible))


def plot_recall_vs_false_alarms(sweeps: dict, chosen: dict, max_alerts: int, path) -> None:
    """
        One panel per model: validation recall vs false alarms across the threshold
        sweep. The part of the curve within the alert budget is drawn solid, the part
        over budget faded, and the chosen operating point is marked and labelled.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    line, ink, muted, grid = '#2a78d6', '#0b0b0b', '#52514e', '#e4e3df'
    names = list(sweeps)
    ncols = 2
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.2 * nrows), sharey=True, squeeze=False)

    for ax, name in zip(axes.flat, names):
        s = sweeps[name].sort_values('false_alarms')
        x = s['false_alarms'].clip(lower=1)        # log axis needs x >= 1
        ok = s['alerts'] <= max_alerts
        ax.plot(x, s['recall'], color=line, lw=2, alpha=0.3, label='over alert budget')
        ax.plot(x[ok], s['recall'][ok], color=line, lw=2, label=f'alerts ≤ {max_alerts:,}')

        c = chosen[name]
        ax.scatter([max(c['false_alarms'], 1)], [c['recall']], s=70, color=ink,
                   edgecolor='white', linewidth=2, zorder=5, label='chosen threshold')
        ax.annotate(f"t = {c['threshold']:.2f}\nrecall {c['recall']:.3f}\n"
                    f"{int(c['false_alarms']):,} false alarms",
                    (max(c['false_alarms'], 1), c['recall']), xytext=(10, -42),
                    textcoords='offset points', fontsize=9, color=ink)

        ax.set_xscale('log')
        ax.set_title(name, fontsize=11, loc='left', color=ink)
        ax.set_xlabel('False alarms on validation (log scale)', color=muted)
        ax.set_ylim(0, 1)
        ax.grid(True, color=grid, lw=0.8)
        ax.tick_params(colors=muted)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(grid)
    for ax in axes[:, 0]:
        ax.set_ylabel('Recall (fraud caught)', color=muted)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)

    axes.flat[0].legend(loc='upper left', fontsize=9, frameon=False)
    fig.suptitle('Validation: recall vs false alarms, thresholds 0.01–0.99',
                 x=0.01, ha='left', fontsize=13, color=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
