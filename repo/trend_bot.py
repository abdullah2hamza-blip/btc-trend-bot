#!/usr/bin/env python3
"""
v2: BTC multi-length trend ensemble with volatility targeting.

Once a day, after the daily candle closes:
  trend score  = fraction of the SMAs {40,50,60,80,100,150 days} that price is above   (0..1)
  vol scaler   = min(1, 40% / 30-day realized annualized volatility)
  target       = trend score x vol scaler   -> fraction of the account held in BTC (0..100%)
Rebalance only if the current holding is more than 15 points away from target (or target is 0).

Why this and not mean reversion: see research/RESULTS.md. Mean reversion lost money in every
test (1,000+ variants, every timeframe). Trend-following beat buy & hold on risk-adjusted
return in walk-forward tests 2020-2025, and on the locked 2025-26 holdout (+7.9% vs -26.7%).

Modes:  backtest | sim | alpaca | binance     (broker modes are paper/testnet only)
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from mr_bot import SimBroker, CcxtBroker, Params, load_data, ccxt

LENGTHS = (40, 50, 60, 80, 100, 150)
VOL_TARGET = 0.40
VOL_DAYS = 30
BAND = 0.15


def target_weight(daily: pd.DataFrame) -> pd.Series:
    c = daily["close"]
    trend = sum((c > c.rolling(n).mean()).astype(float) for n in LENGTHS) / len(LENGTHS)
    vol = np.log(c).diff().rolling(VOL_DAYS).std() * math.sqrt(365)
    return (trend * (VOL_TARGET / vol).clip(upper=1.0)).fillna(0.0)


def needs_trade(target: float, current: float) -> bool:
    return abs(target - current) > BAND or (target == 0 and current > 0.005) or (target >= 0.999 and current < 1 - BAND / 2)


def backtest(daily: pd.DataFrame, cost: float, cash0: float):
    w = target_weight(daily).values
    o, c = daily.open.values, daily.close.values
    cash, qty, pend, eq, trades = cash0, 0.0, None, [], 0
    for i in range(len(c)):
        if pend is not None:
            val = cash + qty * o[i]
            dq = pend * val / o[i] - qty
            cash -= dq * o[i] + abs(dq) * o[i] * cost
            qty += dq
            pend, trades = None, trades + 1
        val = cash + qty * c[i]
        if needs_trade(w[i], qty * c[i] / val):
            pend = w[i]
        eq.append(val)
    return pd.Series(eq, index=daily.index), trades


def paper(args):
    p = Params(fee=args.fee, slippage=0.0005)
    data_ex = getattr(ccxt, args.data)({"enableRateLimit": True})
    broker = SimBroker(args.state, args.cash, p) if args.mode == "sim" else CcxtBroker(args.mode)
    sym = args.symbol + ("T" if args.mode == "binance" and args.symbol.endswith("/USD") else "")
    log = csv.writer(open(args.log, "a", newline=""))
    print(f"[{args.mode}] v2 trend bot on {sym}. Checks once per day after the UTC daily close.")
    while True:
        try:
            raw = data_ex.fetch_ohlcv(args.symbol, "1d", limit=200)
            df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
            df = df.set_index(pd.to_datetime(df.ts, unit="ms", utc=True)).iloc[:-1]  # drop today's open candle
            tgt = float(target_weight(df).iloc[-1])
            px = float(data_ex.fetch_ticker(args.symbol)["last"])
            qty, cash = broker.position(sym), broker.cash()
            equity = cash + qty * px
            cur = qty * px / equity if equity else 0
            action = "hold"
            if needs_trade(tgt, cur):
                delta_usd = tgt * equity - qty * px
                if delta_usd > 10:
                    broker.buy(sym, delta_usd, px); action = f"BUY ${delta_usd:,.0f}"
                elif delta_usd < -10:
                    broker.sell(sym, min(qty, -delta_usd / px), px); action = f"SELL ${-delta_usd:,.0f}"
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
            print(f"{now} close={df.close.iloc[-1]:,.0f} target={tgt:.0%} held={cur:.0%} equity={equity:,.2f} {action}")
            log.writerow([now, df.close.iloc[-1], round(tgt, 3), round(cur, 3), round(equity, 2), action])
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("error:", type(e).__name__, e)
        if args.once:
            break
        time.sleep(86400 - time.time() % 86400 + 120)  # 2 min after the next UTC midnight


def main():
    ap = argparse.ArgumentParser(description="v2 BTC trend ensemble bot")
    ap.add_argument("mode", choices=["backtest", "sim", "alpaca", "binance"])
    ap.add_argument("--data", default="kraken", help="ccxt exchange id, or a .csv of daily OHLCV")
    ap.add_argument("--symbol", default="BTC/USD")
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--cash", type=float, default=10_000)
    ap.add_argument("--fee", type=float, default=0.0025)
    ap.add_argument("--state", default="v2_state.json")
    ap.add_argument("--log", default="v2_log.csv")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    if a.mode == "backtest":
        d = load_data(a.data, a.symbol, "1d", a.days)
        eq, n = backtest(d, a.fee + 0.0005, a.cash)
        r = eq.pct_change().dropna()
        bh = d.close / d.close.iloc[0] * a.cash
        print(f"period {d.index[0]:%Y-%m-%d} -> {d.index[-1]:%Y-%m-%d}, {n} rebalances")
        for name, e in [("v2", eq), ("buy&hold", bh)]:
            rr = e.pct_change().dropna()
            print(f"{name:9s} return {(e.iloc[-1]/e.iloc[0]-1)*100:7.1f}%  sharpe {rr.mean()/rr.std()*math.sqrt(365):5.2f}"
                  f"  maxDD {((e/e.cummax())-1).min()*100:6.1f}%")
        print(f"today's target: {target_weight(d).iloc[-1]:.0%} of account in BTC")
    else:
        paper(a)


if __name__ == "__main__":
    main()
