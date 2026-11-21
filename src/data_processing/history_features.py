# Card-history features (velocity, amount-vs-normal, new-for-this-card flags,
# combined keys) plus row-level time and missingness features.
#
# Leakage rules:
#   - A row's history only contains rows with a STRICTLY earlier TransactionDT
#     (rows in the same second are never counted).
#   - Only rows marked `contributes=True` can be history. Inside a CV fold this is the
#     fold's training period, so validation rows only see training rows as history,
#     never other validation rows.
#   - No labels are used, and nothing is fitted.
#
# Everything is vectorised: rows are sorted by (group, TransactionDT), and each
# "how many history rows before time t" question is a searchsorted on a combined
# int64 key group_id * KEY_SCALE + TransactionDT.

import numpy as np
import pandas as pd

KEY_SCALE = np.int64(10 ** 9)        # > max TransactionDT (~1.6e7 s)
STD_EPS = 1e-4                       # past std below this = constant amounts -> z-score NaN
HOUR, DAY = 3_600, 86_400
VELOCITY_WINDOWS = {'1h': HOUR, '24h': DAY, '7d': 7 * DAY}
NOVELTY_COLUMNS = {'addr1': 'addr1', 'P_emaildomain': 'email', 'ProductCD': 'product',
                   'DeviceInfo_raw': 'device'}

NEW_FEATURES = (
    [f'card1_cnt_{w}' for w in VELOCITY_WINDOWS]
    + ['card1_secs_since_prev', 'card1_spend_24h',
       'amt_to_card_mean', 'amt_card_zscore', 'amt_to_card_max', 'amt_log', 'amt_cents',
       'card1_new_addr1', 'card1_new_email', 'card1_new_product', 'card1_new_device',
       'card1_n_distinct_email', 'card1_n_distinct_addr1',
       'hour_of_day', 'day_of_week', 'n_missing',
       'card1_addr1_count', 'card1_email_count']
)


def _group_ids(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Integer id per distinct combination of `cols`; -1 where any of them is missing."""
    keys = df[cols].astype(object)
    missing = keys.isna().any(axis=1).to_numpy()
    ids = pd.MultiIndex.from_frame(keys.fillna('__nan__')).factorize()[0].astype(np.int64)
    ids[missing] = -1
    return ids


def _past_counts(gid, dt, contributes, query_dt, query_gid):
    """
        For each query (gid, t): number of contributing rows with the same gid and
        dt < t. Also returns the sorted contributing keys/positions for reuse.
    """
    ok = contributes & (gid >= 0)
    keys = gid[ok] * KEY_SCALE + dt[ok]
    order = np.argsort(keys, kind='stable')
    keys = keys[order]
    q = query_gid * KEY_SCALE + query_dt
    hi = np.searchsorted(keys, q, side='left')                  # rows with key < (g, t)
    start = np.searchsorted(keys, query_gid * KEY_SCALE, side='left')
    return hi, start, keys, np.flatnonzero(ok)[order]


def add_history_features(df: pd.DataFrame, contributes: np.ndarray) -> pd.DataFrame:
    """
        Adds the card1 history features to every row of df. `contributes` marks the
        rows that may serve as history for others (all rows still get features).
        df must have TransactionDT, TransactionAmt, card1, addr1, P_emaildomain,
        ProductCD and DeviceInfo_raw.
    """
    out = {}
    dt = df['TransactionDT'].to_numpy(np.int64)
    amt = df['TransactionAmt'].to_numpy(np.float64)
    contributes = np.asarray(contributes, dtype=bool)

    # ---- card1 velocity and amount-vs-normal -------------------------------------
    card = _group_ids(df, ['card1'])
    has_card = card >= 0
    hi, start, keys, src = _past_counts(card, dt, contributes, dt, card)
    n_past = np.where(has_card, hi - start, 0)

    # Running sums restart for every card (and are centred on the card's first amount),
    # so a card's numbers never depend on other cards' rows and the sum-of-squares
    # variance does not lose precision.
    grp_sorted = keys // KEY_SCALE
    a = amt[src]
    first_pos = np.r_[0, np.flatnonzero(np.diff(grp_sorted)) + 1]
    base = np.repeat(a[first_pos], np.diff(np.r_[first_pos, len(a)]))
    cs = pd.DataFrame({'a': a, 'c': a - base, 'c2': (a - base) ** 2}).groupby(grp_sorted).cumsum()
    cs_a, cs_c, cs_c2 = (cs[c].to_numpy() for c in ('a', 'c', 'c2'))

    def prefix(arr, i):
        """sum of the card's contributing rows before sorted position i (i >= start)."""
        return np.where(i > start, arr[np.maximum(i - 1, 0)], 0.0)

    for name, w in VELOCITY_WINDOWS.items():
        lo = np.maximum(np.searchsorted(keys, card * KEY_SCALE + (dt - w), side='left'), start)
        out[f'card1_cnt_{name}'] = np.where(has_card, hi - lo, np.nan)
        if name == '24h':
            out['card1_spend_24h'] = np.where(has_card, prefix(cs_a, hi) - prefix(cs_a, lo), np.nan)

    has_hist = has_card & (n_past > 0)
    prev_dt = np.where(has_hist, keys[np.maximum(hi - 1, 0)] % KEY_SCALE, 0)
    out['card1_secs_since_prev'] = np.where(has_hist, dt - prev_dt, np.nan)

    card_base = np.where(has_hist, base[np.minimum(start, len(a) - 1)] if len(a) else 0.0, 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        mc = prefix(cs_c, hi) / n_past                 # mean of (amount - base)
        mean = card_base + mc
        var = np.clip(prefix(cs_c2, hi) / n_past - mc ** 2, 0, None)
        std = np.sqrt(var)
        out['amt_to_card_mean'] = np.where(has_hist, amt / mean, np.nan)
        z = (amt - mean) / std
        out['amt_card_zscore'] = np.where(has_hist & (n_past >= 2) & (std > STD_EPS), z, np.nan)

    # running max of past amounts within each card (contributing rows, sorted by key)
    cummax = pd.Series(amt[src]).groupby(grp_sorted).cummax().to_numpy()
    past_max = np.where(has_hist, cummax[np.maximum(hi - 1, 0)], np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        out['amt_to_card_max'] = amt / past_max

    out['amt_log'] = np.log1p(amt)
    out['amt_cents'] = np.round(amt - np.floor(amt), 3)

    # ---- new-for-this-card flags and combined-key counts --------------------------
    for col, short in NOVELTY_COLUMNS.items():
        pair = _group_ids(df, ['card1', col])
        p_hi, p_start, _, _ = _past_counts(pair, dt, contributes, dt, pair)
        pair_count = np.where(pair >= 0, p_hi - p_start, np.nan)
        out[f'card1_new_{short}'] = np.where(pair >= 0, (pair_count == 0).astype(float), np.nan)
        if col == 'addr1':
            out['card1_addr1_count'] = pair_count      # combined key card1 + addr1
        if col == 'P_emaildomain':
            out['card1_email_count'] = pair_count      # combined key card1 + P_emaildomain

    # distinct past values per card: count the (card1, value) pairs whose FIRST
    # contributing occurrence is strictly before t
    for col, short in [('P_emaildomain', 'email'), ('addr1', 'addr1')]:
        pair = _group_ids(df, ['card1', col])
        ok = contributes & (pair >= 0)
        first = (pd.DataFrame({'pair': pair[ok], 'card': card[ok], 'dt': dt[ok]})
                 .groupby('pair').agg(card=('card', 'first'), dt=('dt', 'min')))
        fkeys = np.sort(first['card'].to_numpy(np.int64) * KEY_SCALE + first['dt'].to_numpy(np.int64))
        n = (np.searchsorted(fkeys, card * KEY_SCALE + dt, side='left')
             - np.searchsorted(fkeys, card * KEY_SCALE, side='left'))
        out[f'card1_n_distinct_{short}'] = np.where(has_card, n, np.nan)

    # ---- time of day / week -----------------------------------------------------
    out['hour_of_day'] = (dt // HOUR) % 24
    out['day_of_week'] = (dt // DAY) % 7

    new = pd.DataFrame(out, index=df.index).astype(np.float32)
    return pd.concat([df, new], axis=1)
