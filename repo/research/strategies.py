"""Strategy families. Each generator yields (name, callable(df, tf) -> equity series)."""
import itertools
import numpy as np
import pandas as pd
from lab import run, run_weights, sma, zscore, rsi, realized_vol, BARS_PER_DAY, COST


def vol_size(df, tf, target):
    if not target:
        return None
    rv = realized_vol(df, tf)
    return (target / rv).clip(upper=1.0).values


# A. z-score mean reversion (generalised v1)
def mr_z(cost=COST):
    for tf, lb, ez, xz, rf, tr, st, vt in itertools.product(
            ["1h", "4h", "1d"], [20, 50], [1.5, 2.0, 2.5], [0.0, 0.5], [35, 101], [0, 200], [0.0, 0.05, 0.10], [0, 0.5]):
        def f(df, tf=tf, lb=lb, ez=ez, xz=xz, rf=rf, tr=tr, st=st, vt=vt):
            z, r = zscore(df.close, lb), rsi(df.close, 14)
            ok = (z <= -ez) & (r <= rf)
            if tr:
                ok &= df.close > sma(df.close, tr)  # tr bars, as in v1
            hold = {"1h": 48, "4h": 30, "1d": 10}[tf]
            return run(df, ok.values, (z >= xz).values, vol_size(df, tf, vt), st, hold, cost)[0]
        yield f"mrz|{tf}|lb{lb}|z{ez}|x{xz}|rsi{rf}|tr{tr}|st{st}|vt{vt}", tf, f


# B. Connors-style RSI(2) pullback in an uptrend
def rsi2(cost=COST):
    for tf, buy, trd, ex, st, vt in itertools.product(
            ["4h", "1d"], [5, 10, 20], [50, 100, 200], ["sma5", "rsi70"], [0.0, 0.10], [0, 0.5]):
        def f(df, tf=tf, buy=buy, trd=trd, ex=ex, st=st, vt=vt):
            n = trd * BARS_PER_DAY[tf]
            r2 = rsi(df.close, 2)
            ok = (r2 < buy) & (df.close > sma(df.close, n))
            out = (df.close > sma(df.close, 5)) if ex == "sma5" else (r2 > 70)
            return run(df, ok.values, out.values, vol_size(df, tf, vt), st, 0, cost)[0]
        yield f"rsi2|{tf}|b{buy}|tr{trd}d|{ex}|st{st}|vt{vt}", tf, f


# C. Trend: price above moving average (lengths in days)
def trend_sma(cost=COST):
    for tf, n, vt in itertools.product(["4h", "1d"], [20, 50, 100, 150, 200], [0, 0.4, 0.6]):
        def f(df, tf=tf, n=n, vt=vt):
            up = df.close > sma(df.close, n * BARS_PER_DAY[tf])
            return run(df, up.values, (~up).values, vol_size(df, tf, vt), 0.0, 0, cost)[0]
        yield f"sma|{tf}|{n}d|vt{vt}", tf, f


# D. Donchian breakout (lengths in days)
def donchian(cost=COST):
    for tf, n, m, vt in itertools.product(["4h", "1d"], [20, 55, 100], [10, 20, 50], [0, 0.4, 0.6]):
        if m >= n:
            continue
        def f(df, tf=tf, n=n, m=m, vt=vt):
            b = BARS_PER_DAY[tf]
            hi = df.high.rolling(n * b).max().shift(1)
            lo = df.low.rolling(m * b).min().shift(1)
            return run(df, (df.close > hi).values, (df.close < lo).values, vol_size(df, tf, vt), 0.0, 0, cost)[0]
        yield f"donch|{tf}|{n}/{m}d|vt{vt}", tf, f


# E. Ensemble trend: fraction of SMA lengths price is above, optionally vol-targeted
def ens_trend(cost=COST):
    sets = {"short": [10, 20, 50], "mid": [20, 50, 100, 200], "wide": [10, 20, 50, 100, 150, 200]}
    for tf, s, vt, band in itertools.product(["4h", "1d"], sets, [0, 0.4, 0.6], [0.10, 0.25]):
        def f(df, tf=tf, s=s, vt=vt, band=band):
            b = BARS_PER_DAY[tf]
            w = sum((df.close > sma(df.close, n * b)).astype(float) for n in sets[s]) / len(sets[s])
            if vt:
                w = w * (vt / realized_vol(df, tf)).clip(upper=1.0)
            return run_weights(df, w.values, cost, band)[0]
        yield f"ens|{tf}|{s}|vt{vt}|band{band}", tf, f


# F. Hybrid: trend core + buy-the-dip add-on inside uptrends
def hybrid(cost=COST):
    for tf, n, ez, vt in itertools.product(["4h", "1d"], [50, 100, 200], [1.5, 2.0], [0, 0.5]):
        def f(df, tf=tf, n=n, ez=ez, vt=vt):
            b = BARS_PER_DAY[tf]
            up = (df.close > sma(df.close, n * b)).astype(float)
            z = zscore(df.close, 20 * b if tf == "1d" else 20)
            dip = ((z <= -ez) & (df.close > sma(df.close, 2 * n * b))).astype(float)
            w = (0.6 * up + 0.4 * dip.rolling(5).max()).clip(upper=1.0)
            if vt:
                w = w * (vt / realized_vol(df, tf)).clip(upper=1.0)
            return run_weights(df, w.values, cost, 0.15)[0]
        yield f"hyb|{tf}|{n}d|z{ez}|vt{vt}", tf, f


# Benchmarks
def benchmarks(cost=COST):
    yield "B&H", "1d", lambda df: run(df, np.ones(len(df), bool), np.zeros(len(df), bool), None, 0, 0, cost)[0]
    for vt in [0.4, 0.6]:
        def f(df, vt=vt):
            return run_weights(df, (vt / realized_vol(df, "1d")).clip(upper=1).values, cost, 0.10)[0]
        yield f"B&H volTarget{vt}", "1d", f


FAMILIES = {"mean_reversion_z": mr_z, "rsi2_pullback": rsi2, "trend_sma": trend_sma,
            "donchian": donchian, "ensemble_trend": ens_trend, "hybrid_trend_dip": hybrid}
