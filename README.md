# BTC Trend Bot (v2)

A long-only Bitcoin trend-following strategy with volatility targeting, plus the research that
produced it. Checks once a day, holds 0–100% of the account in BTC.

Why this repo is structured the way it is: It is easy to find a backtest that looks good on
crypto and hard to know whether it means anything. So the method here is deliberately conservative:
every configuration tried is logged (1,147 of them) and counted against a deflated-Sharpe
correction; the final strategy and its pass/fail criteria were written down in
[`research/PREREGISTRATION.md`](research/PREREGISTRATION.md) before the holdout year was ever
loaded; and the whole thing is re-run on shuffled data and with delayed decisions to check that the
edge is not a code bug or a look-ahead leak. Results that went the wrong way are kept, not deleted —
mean reversion was my original idea and it does not work, and a machine-learning attempt was
rejected at a 51% hit rate against a 54% base rate.

## Files
- `trend_bot.py` — **the v2 bot** (strategy, backtest, paper trading). Run this one.
- `mr_bot.py` — the old v1 mean reversion bot. v2 imports its broker and data code from here, so keep both files in the same folder.
- `requirements.txt` — `pip install -r requirements.txt`

## How v2 decides (once a day, after the UTC daily close)
1. **Trend score:** the fraction of the 40/50/60/80/100/150-day moving averages that price is above (0 to 1).
2. **Volatility scaler:** min(1, 40% ÷ last 30 days' annualized volatility). In wild markets it holds less.
3. **Target = trend score × volatility scaler.** It trades only if the current holding is more than 15 percentage points off target, which saves fees.

## Run it
```bash
python trend_bot.py backtest --days 1500        # history from Kraken (or --data file.csv)
python trend_bot.py sim                         # paper trading on live prices, no account needed
python trend_bot.py alpaca                      # Alpaca paper account (set ALPACA_API_KEY / ALPACA_API_SECRET)
python trend_bot.py binance                     # Binance Spot Testnet (set BINANCE_TESTNET_KEY / BINANCE_TESTNET_SECRET)
```
Broker modes are hard-wired to paper/testnet. Paper modes log every day to `v2_log.csv`.

## Results (details in research/RESULTS.md)
- **Walk-forward test, 2020–2025:** Sharpe 1.40 vs 1.09 for buy & hold; worst drop −41% vs −77%.
- **Locked final year, Oct 2025 – Sep 2026, tested once:** +7.9% vs −26.7% for buy & hold.
- **Expect it to trail buy & hold in strong bull runs.** Its edge is losing less in crashes.

## Research folder
The code behind the results. To reproduce them:
```bash
cd research
python get_data.py      # downloads ~8 years of hourly BTC from Binance's public archive
python run_all.py       # all 6 strategy families, walk-forward  (--low for lower fees)
python checks.py        # fake-data, look-ahead and bootstrap checks
python ml.py            # the machine-learning attempt
```
Results are in `RESULTS.md`, the written-down choice in `PREREGISTRATION.md`, every experiment in `EXPERIMENT_LEDGER.md`, and out-of-sample tracking since the freeze in `forward_log.md` (backfilled and live rows are labelled separately).
