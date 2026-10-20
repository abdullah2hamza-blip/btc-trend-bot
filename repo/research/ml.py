import numpy as np, pandas as pd, lab
from lab import load, daily, stats, FOLDS, run_weights, realized_vol, sma, rsi, COST
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
d = load("1d")
c = d.close
X = pd.DataFrame({f"ret{n}": c.pct_change(n) for n in [1,3,7,14,30,60,120]})
for n in [20,50,100,200]: X[f"dist{n}"] = c/sma(c,n)-1
X["vol7"] = np.log(c).diff().rolling(7).std(); X["vol30"] = np.log(c).diff().rolling(30).std()
X["volratio"] = X.vol7/X.vol30; X["rsi14"] = rsi(c,14); X["rsi2"] = rsi(c,2)
X["volu"] = np.log(d.volume).diff(7); X["dow"] = d.index.dayofweek
H = 5
y = (c.shift(-H)/c - 1 > 0).astype(int)
valid = X.notna().all(axis=1)
models = {"gbm": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0),
          "logit": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))}
for name, mk in models.items():
  for rule in ["thresh", "prob"]:
    p = pd.Series(np.nan, index=d.index)
    for a, b in FOLDS:
        tr = valid & (d.index < a - pd.Timedelta(days=H))   # embargo so labels don't peek into the test year
        te = valid & (d.index >= a) & (d.index < b)
        m = mk().fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
    w = (p > 0.55).astype(float) if rule == "thresh" else ((p - 0.5) * 4).clip(0, 1)
    w = (w * (0.5 / realized_vol(d, "1d")).clip(upper=1)).fillna(0)
    eq, _ = run_weights(d, w.values, COST, 0.15)
    r = daily(eq); r = pd.concat([r[(r.index>=a)&(r.index<b)] for a,b in FOLDS])
    s = stats(r)
    acc = ((p > 0.5).astype(int) == y)[p.notna()].mean()
    print(f"ML {name:5s} {rule:6s} | hit rate {acc:.3f} | OOS Sharpe {s['sharpe']:.2f} CAGR {s['cagr']*100:5.1f}% maxDD {s['maxdd']*100:5.1f}% | by year",
          [round(stats(r[(r.index>=a)&(r.index<b)])['cagr']*100,1) for a,b in FOLDS])
print("base rate of up-weeks:", round(y[d.index>=FOLDS[0][0]].mean(),3))
