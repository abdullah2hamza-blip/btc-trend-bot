# Pre-registration — written 2026-10-20 BEFORE looking at holdout data

Holdout: 2025-10-01 -> 2026-09-30 (BTCUSDT, Binance). Not loaded by any experiment so far.

**Champion (v2): multi-length trend ensemble with volatility targeting**
- Daily bars. Target weight = (fraction of SMA lengths {40,50,60,80,100,150} days that close is above)
  x min(1, 40% / 30-day realized annualized vol). Long-only, 0-100% of capital.
- Decide at daily close, trade at next open, rebalance only if target drifts > 15 points.
- Costs 0.30%/side.

**Pass criteria (decided now):**
1. Sharpe on holdout > 0, and
2. Max drawdown on holdout shallower than buy & hold's, and
3. Sharpe >= vol-matched buy & hold's Sharpe minus 0.3 (one year is noisy; we only require "not clearly worse").

Also reported, not used to decide: v1 mean reversion, buy & hold, hybrid, Donchian.
Dev evidence: 1,147 configs tried; walk-forward OOS 2020-10..2025-09 Sharpe 1.40 vs B&H 1.09;
bootstrap P(beats B&H Sharpe) ~0.8-0.9 for the trend family; no edge on shuffled data; robust to 1-2 bar delay.
