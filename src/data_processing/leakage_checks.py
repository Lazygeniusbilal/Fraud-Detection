# Leakage checks for history_features.add_history_features.
#
# 1. reference_history_features(): an independent, slower implementation that uses
#    pandas time-based rolling windows (closed='left') and per-(card, second)
#    aggregation shifted by one timestamp. Every history row it uses has a strictly
#    earlier TransactionDT by construction. Comparing it to the fast version on
#    EVERY row checks the strictly-earlier rule for every row and every feature.
#    It assumes every row may serve as history (contributes all True), which is
#    the v3 rule: training period + validation block, gap rows excluded.
# 2. check_label_independence(): recomputes features after replacing the labels
#    (shuffled, then removed) and requires identical output.
# 3. check_future_truncation(): for several cut times T, features of rows with
#    dt <= T must not change when every row after T is removed.

import numpy as np
import pandas as pd

from src.data_processing.history_features import add_history_features, NEW_FEATURES, VELOCITY_WINDOWS, STD_EPS


def _same(a, b, rtol=1e-4, atol=1e-3) -> np.ndarray:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return (np.isnan(a) & np.isnan(b)) | np.isclose(a, b, rtol=rtol, atol=atol)


def reference_history_features(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame({'dt': df['TransactionDT'].to_numpy(np.int64),
                      'amt': df['TransactionAmt'].to_numpy(np.float64),
                      'card': df['card1'].to_numpy(np.float64),
                      'addr1': df['addr1'].to_numpy(np.float64),
                      'email': df['P_emaildomain'].astype(object).to_numpy(),
                      'product': df['ProductCD'].astype(object).to_numpy(),
                      'device': df['DeviceInfo_raw'].astype(object).to_numpy()})
    d['row'] = np.arange(len(d))
    ref = pd.DataFrame(index=d.index)
    has_card = d['card'].notna()
    c = d[has_card].copy()

    # velocity: time-based rolling windows [t - w, t)  (closed='left' excludes time t itself)
    c['ts'] = pd.to_datetime(c['dt'], unit='s')
    c = c.sort_values(['card', 'dt'], kind='stable')
    g = c.set_index('ts').groupby('card', sort=False)['amt']
    for name, w in VELOCITY_WINDOWS.items():
        cnt = g.rolling(f'{w}s', closed='left').count().fillna(0).to_numpy()
        ref.loc[c['row'].to_numpy(), f'card1_cnt_{name}'] = cnt
        if name == '24h':
            s = g.rolling(f'{w}s', closed='left').sum().fillna(0).to_numpy()
            ref.loc[c['row'].to_numpy(), 'card1_spend_24h'] = s

    # expanding stats over strictly earlier seconds: aggregate per (card, second),
    # then cumulate and shift by one second within the card
    sec = c.groupby(['card', 'dt']).agg(n=('amt', 'size'), s=('amt', 'sum'),
                                        mx=('amt', 'max')).reset_index()
    gc_ = sec.groupby('card')
    sec['n_past'] = gc_['n'].cumsum() - sec['n']
    sec['prev_dt'] = gc_['dt'].shift(1)
    sec['max_past'] = gc_['mx'].cummax().groupby(sec['card']).shift(1)
    m = c[['row', 'card', 'dt', 'amt']].merge(sec[['card', 'dt', 'n_past', 'prev_dt', 'max_past']],
                                              on=['card', 'dt'], how='left')
    # past mean / std from the explicit list of earlier amounts (two-pass, no cumsum tricks)
    means, stds = np.full(len(m), np.nan), np.full(len(m), np.nan)
    m = m.sort_values(['card', 'dt']).reset_index(drop=True)
    for _, grp in m.groupby('card', sort=False):
        a, t, idx = grp['amt'].to_numpy(), grp['dt'].to_numpy(), grp.index.to_numpy()
        k = np.searchsorted(t, t, side='left')            # rows strictly before each time
        csum = np.concatenate([[0.0], np.cumsum(a)])
        for j in np.flatnonzero(k > 0):
            past = a[:k[j]]
            means[idx[j]] = csum[k[j]] / k[j]
            if k[j] >= 2:
                stds[idx[j]] = past.std()
    rows = m['row'].to_numpy()
    amt = m['amt'].to_numpy()
    has_hist = m['n_past'].to_numpy() > 0
    ref.loc[rows, 'card1_secs_since_prev'] = np.where(has_hist, m['dt'] - m['prev_dt'], np.nan)
    ref.loc[rows, 'amt_to_card_mean'] = np.where(has_hist, amt / means, np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        z = (amt - means) / stds
    ref.loc[rows, 'amt_card_zscore'] = np.where(stds > STD_EPS, z, np.nan)
    ref.loc[rows, 'amt_to_card_max'] = np.where(has_hist, amt / m['max_past'], np.nan)

    # (card, value) pairs: past count, first-time flag, distinct past values per card
    for col, short in [('addr1', 'addr1'), ('email', 'email'), ('product', 'product'), ('device', 'device')]:
        p = c[c[col].notna()][['row', 'card', col, 'dt']]
        ps = p.groupby(['card', col, 'dt']).size().rename('n').reset_index()
        ps['n_past'] = ps.groupby(['card', col])['n'].cumsum() - ps['n']
        p = p.merge(ps, on=['card', col, 'dt'], how='left')
        ref.loc[p['row'].to_numpy(), f'card1_new_{short}'] = (p['n_past'] == 0).astype(float).to_numpy()
        if short in ('addr1', 'email'):
            ref.loc[p['row'].to_numpy(), f'card1_{short}_count'] = p['n_past'].to_numpy()
            first = p.groupby(['card', col])['dt'].min().rename('first').reset_index()
            cnt = np.zeros(len(c))
            cr, cd = c['card'].to_numpy(), c['dt'].to_numpy()
            fdict = {k: np.sort(v.to_numpy()) for k, v in first.groupby('card')['first']}
            for i in range(len(c)):
                f = fdict.get(cr[i])
                cnt[i] = 0 if f is None else np.searchsorted(f, cd[i], side='left')
            ref.loc[c['row'].to_numpy(), f'card1_n_distinct_{short}'] = cnt
    return ref


HISTORY_ONLY = [f for f in NEW_FEATURES if f.startswith('card1_') or f.startswith('amt_to')
                or f == 'amt_card_zscore']


def check_reference(df: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Compares every history feature on every row. Returns one row per feature."""
    ref = reference_history_features(df)
    out = []
    for f in HISTORY_ONLY:
        ok = _same(feats[f].to_numpy(), ref[f].reindex(range(len(df))).to_numpy())
        out.append({'feature': f, 'rows': len(df), 'mismatches': int((~ok).sum())})
    return pd.DataFrame(out)


def check_label_independence(df: pd.DataFrame, contributes, feats: pd.DataFrame, seed=0) -> int:
    """Mismatching cells after shuffling labels and after removing the label column."""
    bad = 0
    shuffled = df.copy()
    shuffled['isFraud'] = np.random.default_rng(seed).permutation(shuffled['isFraud'].to_numpy())
    for variant in (shuffled, df.drop(columns='isFraud')):
        alt = add_history_features(variant, contributes)
        for f in NEW_FEATURES:
            if f in feats:
                bad += int((~_same(feats[f], alt[f], rtol=0, atol=0)).sum())
    return bad


def check_future_truncation(df: pd.DataFrame, feats: pd.DataFrame, cuts) -> int:
    """Rows with dt <= T must have identical features when all rows after T are removed."""
    bad = 0
    dt = df['TransactionDT'].to_numpy()
    for T in cuts:
        keep = dt <= T
        alt = add_history_features(df[keep].reset_index(drop=True), np.ones(keep.sum(), bool))
        for f in NEW_FEATURES:
            if f in feats:
                bad += int((~_same(feats[f].to_numpy()[keep], alt[f].to_numpy(), rtol=0, atol=0)).sum())
    return bad
