"""
Research harness for BTC strategies.

Rules this file enforces so results can't fool us:
  * HOLDOUT (Oct 2025 -> Sep 2026) is cut off before any strategy sees data. Only final_test() touches it.
  * Decisions use closed bars only; orders fill at the NEXT bar's open. Stops fill at the stop price
    (or the open, if price gapped through it).
  * Costs are charged on every fill. Default is conservative (0.25% fee + 0.05% slippage per side).
  * Parameters are chosen by walk-forward: for each test year, pick the best config using only
    earlier data, then record how that pick did in the year it hadn't seen.
  * Every config ever evaluated is counted, and the Deflated Sharpe Ratio corrects for that.
"""
import math
import os
import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import norm, skew, kurtosis

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "btc_1h.pkl")  # run get_data.py first
HOLDOUT_START = pd.Timestamp("2025-10-01", tz="UTC")
DEV_START = pd.Timestamp("2018-01-01", tz="UTC")
# walk-forward test windows (Oct -> Sep years), each picked using only data before it
FOLDS = [(pd.Timestamp(f"{y}-10-01", tz="UTC"), pd.Timestamp(f"{y+1}-10-01", tz="UTC")) for y in range(2020, 2025)]
COST = 0.0030            # per side, conservative (Alpaca base tier 0.25% + slippage)
COST_LOW = 0.0015        # per side, Binance-ish 0.10% + slippage
TRIALS = {"n": 0}
DELAY = {"bars": 0}     # robustness test: act N bars late        # global counter of configs evaluated (for deflated Sharpe)


# ------------------------------------------------------------------ data
def load(tf="1h", holdout=False):
    df = pd.read_pickle(DATA)
    if not holdout:
        df = df[df.index < HOLDOUT_START]
    if tf != "1h":
        rule = {"4h": "4h", "1d": "1D"}[tf]
        df = df.resample(rule).agg({"open": "first", "high": "max", "low": "min",
                                    "close": "last", "volume": "sum"}).dropna()
    return df


BARS_PER_DAY = {"1h": 24, "4h": 6, "1d": 1}


# ------------------------------------------------------------------ indicators
def sma(s, n): return s.rolling(n).mean()
def zscore(s, n): return (s - s.rolling(n).mean()) / s.rolling(n).std()


def rsi(s, n):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(100)


def realized_vol(df, tf, n_days=30):
    bpd = BARS_PER_DAY[tf]
    r = np.log(df.close).diff()
    return r.rolling(n_days * bpd).std() * math.sqrt(365 * bpd)


# ------------------------------------------------------------------ engine
@njit(cache=True)
def _run(o, h, l, c, entry, exit_, size, stop, max_hold, cost):
    """Long-only state machine. entry/exit evaluated at close i, filled at open i+1.
    Returns per-bar equity (close-to-close) and number of round trips."""
    n = len(c)
    eq = np.ones(n)
    cash, qty = 1.0, 0.0
    in_pos, pending_buy, pending_sell = False, False, False
    entry_px, held, trips, buy_size = 0.0, 0, 0, 1.0
    for i in range(n):
        # fills at this bar's open
        if pending_buy:
            spend = cash * buy_size
            px = o[i]
            qty = spend * (1 - cost) / px
            cash -= spend
            entry_px, held, in_pos = px, 0, True
            pending_buy = False
        elif pending_sell:
            cash += qty * o[i] * (1 - cost)
            qty, in_pos, trips = 0.0, False, trips + 1
            pending_sell = False
        # resting stop order during the bar
        if in_pos and stop > 0:
            sp = entry_px * (1 - stop)
            if l[i] <= sp:
                fill = min(o[i], sp)
                cash += qty * fill * (1 - cost)
                qty, in_pos, trips = 0.0, False, trips + 1
        # decide at close
        if in_pos:
            held += 1
            if exit_[i] or (max_hold > 0 and held >= max_hold):
                pending_sell = True
        elif entry[i] and i + 1 < n and not np.isnan(size[i]) and size[i] > 0:
            pending_buy = True
            buy_size = min(size[i], 1.0)
        eq[i] = cash + qty * c[i]
    return eq, trips


def run(df, entry, exit_, size=None, stop=0.0, max_hold=0, cost=COST):
    TRIALS["n"] += 1
    size = np.ones(len(df)) if size is None else np.asarray(size, float)
    if DELAY["bars"]:
        k = DELAY["bars"]
        entry = np.r_[np.zeros(k, bool), np.asarray(entry, bool)[:-k]]
        exit_ = np.r_[np.zeros(k, bool), np.asarray(exit_, bool)[:-k]]
        size = np.r_[np.zeros(k), size[:-k]]
    eq, trips = _run(df.open.values, df.high.values, df.low.values, df.close.values,
                     np.asarray(entry, np.bool_), np.asarray(exit_, np.bool_), size,
                     float(stop), int(max_hold), float(cost))
    return pd.Series(eq, index=df.index), trips


def run_weights(df, w, cost=COST, band=0.10):
    """Fractional target weight in [0,1] decided at close, traded at next open.
    Rebalances only when target drifts more than `band` (saves fees)."""
    TRIALS["n"] += 1
    o, c = df.open.values, df.close.values
    w = np.nan_to_num(np.asarray(w, float))
    if DELAY["bars"]:
        w = np.r_[np.zeros(DELAY["bars"]), w[:-DELAY["bars"]]]
    n = len(c)
    eq = np.ones(n)
    cash, qty, target, pending = 1.0, 0.0, 0.0, None
    turnover = 0.0
    for i in range(n):
        if pending is not None:
            val = cash + qty * o[i]
            want = pending * val / o[i]
            dq = want - qty
            cash -= dq * o[i] + abs(dq) * o[i] * cost
            turnover += abs(dq) * o[i] / val
            qty = want
            target, pending = pending, None
        val = cash + qty * c[i]
        cur = qty * c[i] / val if val > 0 else 0
        if abs(w[i] - cur) > band or (w[i] == 0 and cur > 0.005) or (w[i] >= 0.999 and cur < 1 - band / 2):
            pending = w[i]
        eq[i] = val
    return pd.Series(eq, index=df.index), turnover / 2


# ------------------------------------------------------------------ metrics
def daily(eq):
    return eq.resample("1D").last().pct_change().dropna()


def stats(rets):
    """rets: daily returns."""
    if len(rets) < 20 or rets.std() == 0:
        return dict(cagr=0.0, sharpe=0.0, maxdd=0.0, vol=0.0)
    eq = (1 + rets).cumprod()
    yrs = len(rets) / 365
    return dict(cagr=eq.iloc[-1] ** (1 / yrs) - 1,
                sharpe=rets.mean() / rets.std() * math.sqrt(365),
                maxdd=(eq / eq.cummax() - 1).min(),
                vol=rets.std() * math.sqrt(365))


def deflated_sharpe(rets, n_trials, sr_trials_std=None):
    """Bailey & Lopez de Prado. Probability the true Sharpe > 0 after correcting for
    how many configs were tried (n_trials). Uses per-period (daily) Sharpe internally."""
    sr = rets.mean() / rets.std()
    T = len(rets)
    g3, g4 = skew(rets), kurtosis(rets, fisher=False)
    # expected max Sharpe of n_trials independent zero-skill strategies
    v = sr_trials_std if sr_trials_std is not None else 1 / math.sqrt(T)
    emc = 0.5772156649
    n = max(n_trials, 2)
    sr0 = v * ((1 - emc) * norm.ppf(1 - 1 / n) + emc * norm.ppf(1 - 1 / (n * math.e)))
    denom = math.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return float(norm.cdf((sr - sr0) * math.sqrt(T - 1) / denom))


# ------------------------------------------------------------------ walk-forward selection
def walk_forward(results):
    """results: dict name -> daily returns over full dev period (computed with past-only indicators).
    For each fold pick the config with best Sharpe on data BEFORE the fold, collect its fold returns."""
    oos, picks = [], []
    for a, b in FOLDS:
        best, best_sr = None, -9
        for k, r in results.items():
            tr = r[r.index < a]
            tr = tr[tr.index >= tr.index[0] + pd.Timedelta(days=200)]  # skip indicator warm-up
            s = stats(tr)["sharpe"]
            if s > best_sr:
                best, best_sr = k, s
        picks.append((a.year, best, round(best_sr, 2)))
        oos.append(results[best][(results[best].index >= a) & (results[best].index < b)])
    return pd.concat(oos), picks


def bh(df, cost=COST):
    e = df.close / df.open.iloc[0] * (1 - cost)
    return e
