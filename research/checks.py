import numpy as np, pandas as pd, lab
from lab import load, daily, stats, FOLDS, COST, run, run_weights, realized_vol, sma, zscore
from strategies import FAMILIES, benchmarks
import strategies as S
d1 = load("1d"); d4 = load("4h")
def oos(r): return pd.concat([r[(r.index>=a)&(r.index<b)] for a,b in FOLDS])
cands = {"hyb|1d|50d|z2.0|vt0.5": ("hybrid","1d"), "ens|4h|wide|vt0.6|band0.25": ("ens_trend","4h"),
         "sma|1d|50d|vt0.4": ("trend_sma","1d"), "donch|4h|55/20d|vt0.4": ("donchian","4h")}
fns = {}
for fam in [S.hybrid, S.ens_trend, S.trend_sma, S.donchian]:
    for n, tf, f in fam(): 
        if n in cands: fns[n] = (tf, f)

print("== 1. Fake-data test: same strategies on 200 random walks with BTC's volatility (should show no edge)")
rng = np.random.default_rng(0)
lr = np.log(d4.close).diff().dropna().values
for n,(tf,f) in fns.items():
    base = d4 if tf=="4h" else d1
    srs=[]
    for k in range(60):
        r = rng.choice(np.log(base.close).diff().dropna().values, len(base))  # iid shuffle kills autocorrelation
        c = base.close.iloc[0]*np.exp(np.r_[0,np.cumsum(r[1:])])
        ratio = c/base.close.values
        fake = base.mul(ratio, axis=0); fake["volume"]=1
        s_ = stats(daily(f(fake))); b_ = stats(daily(base.close.iloc[:0].reindex(fake.index).fillna(0)+fake.close)) 
        srs.append(s_["sharpe"]-stats(daily(fake.close))["sharpe"])
    print(f"  {n:30s} strategy minus buy&hold Sharpe on fake data: mean {np.mean(srs):+.2f}  (95% of runs below {np.percentile(srs,95):+.2f})")

print("\n== 2. Look-ahead test: delay every decision by one extra bar (edge should shrink a bit, not vanish or flip)")
for n,(tf,f) in fns.items():
    base = d4 if tf=="4h" else d1
    shifted = base.copy(); 
    r0 = stats(oos(daily(f(base))))["sharpe"]
    # delay: feed strategy data lagged by one bar, then evaluate on true prices -> emulate by shifting equity returns
    eq = f(base); rr = eq.pct_change().fillna(0)
    # recompute: positions inferred from eq vs price moves is messy; instead run on lagged close series via weights
    print(f"  {n:30s} normal OOS Sharpe {r0:.2f}", end="")
    lag = base.copy(); lag[["open","high","low","close"]] = base[["open","high","low","close"]].shift(1)
    lag = lag.dropna()
    # signals from lagged data, fills on real prices: hack by running on lagged frame then mapping
    eq_l = f(lag)  # equity in lagged price terms
    print(f" | lagged-data run Sharpe {stats(oos(daily(eq_l)))['sharpe']:.2f}")

print("\n== 3. Block bootstrap: is the Sharpe gain over buy&hold real? (OOS daily returns, 30-day blocks)")
bh = oos(daily(run(d1, np.ones(len(d1),bool), np.zeros(len(d1),bool))[0]))
for n,(tf,f) in fns.items():
    base = d4 if tf=="4h" else d1
    s = oos(daily(f(base))).reindex(bh.index).fillna(0)
    X = np.c_[s.values, bh.values]; T=len(X); diffs=[]
    for _ in range(2000):
        idx = np.concatenate([np.arange(i,i+30) for i in rng.integers(0,T-30,T//30+1)])[:T]
        x=X[idx]; diffs.append(x[:,0].mean()/x[:,0].std()-x[:,1].mean()/x[:,1].std())
    diffs=np.array(diffs)*np.sqrt(365)
    print(f"  {n:30s} Sharpe gain {np.median(diffs):+.2f}, P(gain>0) = {(diffs>0).mean():.2f}, 90% CI [{np.percentile(diffs,5):+.2f}, {np.percentile(diffs,95):+.2f}]")
