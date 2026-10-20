# Experiment ledger

Every config tried counts toward the deflated-Sharpe correction. Dates are when the work was done,
not when this file was last written.

| Date | What | Configs | Running total | Outcome |
|---|---|---|---|---|
| 2026-05-28 | v1 mean reversion grid (Bitstamp) | ~200 | ~200 | overfit, rejected |
| 2026-10-20 | 6 families walk-forward (MR-z, RSI2, SMA, Donchian, ensemble, hybrid) | 1,140 | ~1,340 | trend families pass; MR fails |
| 2026-10-20 | ML (GBM, logit) | 4 | ~1,344 | rejected |
| 2026-10-20 | v2 chosen and pre-registered; holdout tested once | 1 | ~1,345 | PASS; holdout now spent |

## Rules for new ideas
1. Develop only on data up to 2025-09-30, walk-forward. Log the number of configs here.
2. A challenger must beat v2's walk-forward Sharpe (1.40) by at least 0.15 at the same costs, survive the fake-data and delay checks, and sit on a plateau of nearby settings.
3. A challenger that passes goes into **shadow forward-tracking** next to v2 in forward_log.md. It replaces v2 only after at least 6 months of forward data in which it does no worse than v2. Never re-tune v2 on forward data.
