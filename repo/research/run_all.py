import sys, time, pickle, numpy as np, pandas as pd
import lab
from lab import load, daily, stats, walk_forward, deflated_sharpe, FOLDS, COST, COST_LOW
from strategies import FAMILIES, benchmarks

cost = COST_LOW if "--low" in sys.argv else COST
data = {tf: load(tf) for tf in ["1h", "4h", "1d"]}
def fold_table(r):
    return [round(stats(r[(r.index>=a)&(r.index<b)])["cagr"]*100,1) for a,b in FOLDS]

out = {}
bench = {n: daily(f(data[tf])) for n, tf, f in benchmarks(cost)}
for n, r in bench.items():
    oos = pd.concat([r[(r.index>=a)&(r.index<b)] for a,b in FOLDS]); s = stats(oos)
    print(f"{n:28s} OOS sharpe {s['sharpe']:.2f} cagr {s['cagr']*100:6.1f}% maxdd {s['maxdd']*100:6.1f}%  by year {fold_table(r)}")
print()
for fam, gen in FAMILIES.items():
    t = time.time(); res = {}
    for name, tf, f in gen(cost):
        res[name] = daily(f(data[tf]))
    oos, picks = walk_forward(res)
    s = stats(oos)
    srs = np.array([r.mean()/r.std() for r in res.values() if r.std()>0])
    dsr = deflated_sharpe(oos, len(res), srs.std())
    out[fam] = dict(res=res, oos=oos, picks=picks)
    print(f"{fam:20s} {len(res):4d} cfgs | walk-fwd OOS sharpe {s['sharpe']:5.2f} cagr {s['cagr']*100:6.1f}% maxdd {s['maxdd']*100:6.1f}% "
          f"| deflated P(skill) {dsr:.2f} | {time.time()-t:.0f}s")
    print("     picks:", picks)
    print("     OOS by year:", [round(stats(oos[(oos.index>=a)&(oos.index<b)])['cagr']*100,1) for a,b in FOLDS])
print("\ntotal configs evaluated:", lab.TRIALS["n"])
pickle.dump({k:{"oos":v["oos"],"picks":v["picks"]} for k,v in out.items()} | {"_bench":bench, "_trials":lab.TRIALS["n"]},
            open(f"results_{'low' if cost==COST_LOW else 'base'}.pkl","wb"))
