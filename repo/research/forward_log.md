# v2 out-of-sample tracking

Strategy frozen 2026-10-20. Rules are fixed in `mr_bot.py` / `trend_bot.py` and are not re-tuned on anything below.

**Two distinct periods — read the labels.**

- **2026-09-30 to 2026-10-19 — backfilled.** Computed *after* the freeze by applying the frozen rules to past bars. Development data ended 2025-09-30 and the holdout ended 2026-09-30, so these bars were never used to build or select the strategy — but they were not observed live, and they are not a forward test.
- **2026-10-20 onward — live.** Appended once per day by the scheduled run, as the bars arrive.

Start: $10,000 on 2026-09-30 close. Costs 0.30%/side. Data: Binance BTCUSDT daily (data.binance.vision).

| Date | BTC close | v2 target | v2 equity | Buy&hold equity |
|---|---|---|---|---|
| 2026-09-30 | 83,624 | 97% | 10,000 | 10,000 |
| 2026-10-01 | 84,880 | 98% | 10,146 | 10,150 |
| 2026-10-02 | 84,518 | 98% | 10,104 | 10,107 |
| 2026-10-03 | 84,754 | 100% | 10,131 | 10,135 |
| 2026-10-04 | 86,530 | 100% | 10,338 | 10,348 |
| 2026-10-05 | 85,767 | 100% | 10,249 | 10,256 |
| 2026-10-06 | 85,550 | 100% | 10,224 | 10,230 |
| 2026-10-07 | 83,322 | 100% | 9,965 | 9,964 |
| 2026-10-08 | 81,754 | 100% | 9,783 | 9,776 |
| 2026-10-09 | 82,636 | 100% | 9,885 | 9,882 |
| 2026-10-10 | 82,988 | 100% | 9,926 | 9,924 |
| 2026-10-11 | 83,773 | 100% | 10,017 | 10,018 |
| 2026-10-12 | 81,850 | 100% | 9,794 | 9,788 |
| 2026-10-13 | 83,169 | 100% | 9,947 | 9,946 |
| 2026-10-14 | 81,682 | 83% | 9,774 | 9,768 |
| 2026-10-15 | 82,246 | 100% | 9,840 | 9,835 |
| 2026-10-16 | 81,616 | 83% | 9,766 | 9,760 |
| 2026-10-17 | 81,491 | 83% | 9,752 | 9,745 |
| 2026-10-18 | 82,116 | 100% | 9,825 | 9,820 |
| 2026-10-19 | 81,938 | 83% | 9,804 | 9,798 |

*(rows above are backfilled; live daily rows begin 2026-10-20)*
