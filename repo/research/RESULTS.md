# BTC strategy research — results log

**Data:** Binance BTCUSDT hourly, Jan 2018 – Sep 2026 (76,560 bars), from data.binance.vision.
**Costs:** 0.25% fee + 0.05% slippage per side (Alpaca base tier). Also re-run at 0.10% + 0.05%.
**Method:** Decide on closed bars and fill at the next open. Pick parameters walk-forward (each test year Oct–Sep, 2020-21 through 2024-25, uses only earlier data to choose). The final year (Oct 2025 – Sep 2026) was locked and tested **once**, after the choice was written down (PREREGISTRATION.md).

## 1. Mean reversion does not work on BTC (v1 retired)
| Family | Configs | Walk-forward Sharpe | CAGR | Deflated P(skill) |
|---|---|---|---|---|
| z-score mean reversion (1h/4h/1d) | 864 | 0.05 | 0.0% | 0.00 |
| RSI(2) pullback | 144 | −0.06 | −3.6% | 0.00 |

Mean reversion still had no edge with low fees (Sharpe 0.14). Its small backtest profits vanish if it acts 6 bars late, which means it is fragile.

## 2. Trend-following works, mainly by cutting crashes
| Strategy | Walk-forward Sharpe | CAGR | Max DD |
|---|---|---|---|
| Buy & hold | 1.09 | 60.3% | −76.6% |
| Buy & hold, same risk as v2 | 1.09 | 32.8% | −49.4% |
| Trend: price > SMA | 1.28 | 53.5% | −61.9% |
| Donchian breakout | 1.19 | 44.9% | −40.3% |
| Ensemble trend | 1.33 | 50.9% | −41.6% |
| Trend + dip-buy hybrid | 1.48 | 37.0% | −35.3% |
| **v2 (chosen): multi-SMA 40–150d ensemble, 40% vol target** | **1.40** | **46.0%** | **−41.0%** |

**Checks:**
- **Fake data:** on shuffled returns (trends destroyed), every trend strategy loses to buy & hold by about 0.35 Sharpe. So the edge comes from real trending in BTC, not from a code bug.
- **Acting 1–2 bars late:** Sharpe moves from 1.5 to about 1.45, so there is no look-ahead leak. At 6 bars late, Sharpe is still about 1.2.
- **Settings:** SMA lengths from 40 to 150 days all give Sharpe 1.3–1.6, a plateau rather than a single spike. That is why v2 averages across the whole range.
- **Block bootstrap vs buy & hold:** P(Sharpe gain > 0) is about 0.77–0.94. That points to an edge but is not statistically conclusive.
- **Dip-buy add-on:** removing it changes almost nothing, so the mean reversion component adds nothing even inside a trend system.
- **Machine learning** (gradient boosting and logistic regression on 17 features, walk-forward, with embargo): 51% hit rate vs a 54% base rate of up-weeks, Sharpe 0.3–0.6. Rejected.

## 3. Locked holdout, Oct 2025 – Sep 2026 (tested once)
| | Return | Sharpe | Max DD |
|---|---|---|---|
| **v2** | **+7.9%** | **0.46** | **−20.8%** |
| Buy & hold | −26.7% | −0.47 | −53.0% |
| v1 mean reversion | −11.1% | −1.58 | −12.9% |
| Hybrid | +5.5% | 0.41 | −15.2% |
| Donchian | −12.5% | −0.47 | −29.7% |

All 3 pre-registered pass criteria were met. The holdout is now **spent**: it cannot be used to validate future changes.

## 4. Honest limits
- One holdout year is a small sample. v2 is expected to **lag** buy & hold in strong bull markets (2020-21: +176% vs +307%). Its value is that it loses less in crashes.
- 1,147 configs were tried in total. The trend result survives the deflated Sharpe correction, but the uncertainty is real.
- These are backtests. Live out-of-sample tracking only began 2026-10-20, so there is almost no real forward evidence yet. Paper trade before trusting it with anything.

## 5. Out-of-sample tracking since the freeze
Strategy frozen 2026-10-20. `research/forward_log.md` has two clearly separated parts: bars from
2026-09-30 to 2026-10-19 **backfilled** by applying the frozen rules to past data (never used in
development or selection, but not observed live), and **live** rows appended daily from 2026-10-20
onward. Nothing below the freeze line is used to re-tune v2.
