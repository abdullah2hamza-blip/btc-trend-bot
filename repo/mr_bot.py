#!/usr/bin/env python3
"""
Mean reversion BTC trading bot.

Strategy (long-only, spot):
  * z-score = (close - SMA(lookback)) / StdDev(lookback)
  * BUY  when z <= -entry_z AND RSI <= rsi_buy AND (optional) price is above the long trend SMA
  * SELL when z >= exit_z, or stop loss hit, or position held longer than max_hold bars
  Signals are computed on closed candles; fills happen at the NEXT bar's open (no look-ahead).

Modes:
  backtest  - historical test on exchange data or a CSV (timestamp,open,high,low,close,volume)
  sim       - live prices, simulated fills, no account needed (local paper trading)
  alpaca    - Alpaca paper account (paper-api.alpaca.markets) - needs paper API keys
  binance   - Binance Spot Testnet (testnet.binance.vision)   - needs testnet API keys

This file never trades real money: broker modes always run in sandbox/paper mode.
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    import ccxt
except ImportError:  # backtest from CSV works without ccxt
    ccxt = None


# --------------------------------------------------------------------------- config
@dataclass
class Params:
    lookback: int = 20          # bars for SMA / std (Bollinger window)
    entry_z: float = 2.0        # buy when z-score <= -entry_z
    exit_z: float = 0.0         # sell when z-score >= exit_z (0 = back at the mean)
    rsi_period: int = 14
    rsi_buy: float = 35.0       # also require RSI at or below this to buy (100 disables)
    trend_sma: int = 200        # only buy above this SMA (0 disables the trend filter)
    stop_loss: float = 0.04     # 4% stop below entry
    max_hold: int = 48          # force exit after N bars
    risk_fraction: float = 0.95 # fraction of cash used per entry
    fee: float = 0.0025         # per side (0.25% ~ Alpaca's base crypto taker tier)
    slippage: float = 0.0005    # per side


# --------------------------------------------------------------------------- indicators
def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100)


def add_indicators(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    df = df.copy()
    df["sma"] = df["close"].rolling(p.lookback).mean()
    df["std"] = df["close"].rolling(p.lookback).std()
    df["z"] = (df["close"] - df["sma"]) / df["std"]
    df["rsi"] = rsi(df["close"], p.rsi_period)
    df["trend"] = df["close"].rolling(p.trend_sma).mean() if p.trend_sma > 0 else -np.inf
    return df


def entry_signal(row, p: Params) -> bool:
    if pd.isna(row["z"]) or pd.isna(row["trend"]):
        return False
    return row["z"] <= -p.entry_z and row["rsi"] <= p.rsi_buy and row["close"] > row["trend"]


def exit_reason(row, entry_price: float, bars_held: int, p: Params):
    if row["low"] <= entry_price * (1 - p.stop_loss):
        return "stop"
    if row["z"] >= p.exit_z:
        return "mean"
    if bars_held >= p.max_hold:
        return "timeout"
    return None


# --------------------------------------------------------------------------- data
def load_data(source: str, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    if source.endswith(".csv"):
        df = pd.read_csv(source)
        ts = df["timestamp"]
        if pd.api.types.is_numeric_dtype(ts):  # epoch seconds or milliseconds
            df["timestamp"] = pd.to_datetime(ts, unit="ms" if ts.iloc[0] > 1e11 else "s", utc=True)
        else:
            df["timestamp"] = pd.to_datetime(ts, utc=True)
        return df.set_index("timestamp")[["open", "high", "low", "close", "volume"]].astype(float)

    if ccxt is None:
        sys.exit("pip install ccxt  (needed to download data)")
    ex = getattr(ccxt, source)({"enableRateLimit": True})
    since = ex.milliseconds() - days * 86_400_000
    rows = []
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + 1
        if len(batch) < 2 or since >= ex.milliseconds():
            break
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"]).drop_duplicates("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp")


# --------------------------------------------------------------------------- backtest
def backtest(df: pd.DataFrame, p: Params, start_cash: float = 10_000.0):
    df = add_indicators(df, p)
    cash, qty = start_cash, 0.0
    entry_price, bars_held = 0.0, 0
    pending = None              # order decided on bar i, filled at open of bar i+1
    trades, equity = [], []
    cost = p.fee + p.slippage

    for ts, row in df.iterrows():
        # 1) fill yesterday's decision at today's open
        if pending == "buy":
            px = row["open"] * (1 + p.slippage)
            spend = cash * p.risk_fraction
            qty = spend * (1 - p.fee) / px
            cash -= spend
            entry_price, bars_held = px, 0
            trades.append({"entry_time": ts, "entry": px})
        elif pending and pending.startswith("sell"):
            reason = pending.split(":")[1]
            px = row["open"] * (1 - p.slippage)  # market exit at next open, same as the paper loop
            cash += qty * px * (1 - p.fee)
            trades[-1].update(exit_time=ts, exit=px, reason=reason,
                              pnl_pct=(px * (1 - p.fee)) / (trades[-1]["entry"] / (1 - p.fee)) - 1)
            qty = 0.0
        pending = None

        # 2) decide on this bar's close
        if qty > 0:
            bars_held += 1
            r = exit_reason(row, entry_price, bars_held, p)
            if r:
                pending = f"sell:{r}"
        elif entry_signal(row, p):
            pending = "buy"

        equity.append(cash + qty * row["close"])

    eq = pd.Series(equity, index=df.index)
    return eq, pd.DataFrame(trades), df


def report(eq: pd.Series, trades: pd.DataFrame, df: pd.DataFrame, start_cash: float) -> dict:
    rets = eq.pct_change().dropna()
    bar_secs = (df.index[1] - df.index[0]).total_seconds() if len(df) > 1 else 3600
    per_year = 365 * 86400 / bar_secs
    closed = trades.dropna(subset=["pnl_pct"]) if len(trades) and "pnl_pct" in trades else pd.DataFrame()
    out = {
        "period": f"{df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}",
        "bars": len(df),
        "final_equity": round(eq.iloc[-1], 2),
        "strategy_return_%": round((eq.iloc[-1] / start_cash - 1) * 100, 2),
        "buy_and_hold_%": round((df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100, 2),
        "max_drawdown_%": round(((eq / eq.cummax()) - 1).min() * 100, 2),
        "sharpe": round(rets.mean() / rets.std() * math.sqrt(per_year), 2) if rets.std() > 0 else 0.0,
        "trades": len(closed),
        "win_rate_%": round((closed["pnl_pct"] > 0).mean() * 100, 1) if len(closed) else 0.0,
        "avg_trade_%": round(closed["pnl_pct"].mean() * 100, 2) if len(closed) else 0.0,
        "exits": closed["reason"].value_counts().to_dict() if len(closed) else {},
        "time_in_market_%": round(
            sum((t.exit_time - t.entry_time).total_seconds() for t in closed.itertuples())
            / max((df.index[-1] - df.index[0]).total_seconds(), 1) * 100, 1) if len(closed) else 0.0,
    }
    return out


def walk_forward(df, p, start_cash, folds=4):
    """Split history into chunks and test each separately - one good period isn't an edge."""
    n = len(df) // folds
    rows = []
    for k in range(folds):
        chunk = df.iloc[max(0, k * n - p.trend_sma):(k + 1) * n]  # warm-up bars for indicators
        eq, tr, d = backtest(chunk, p, start_cash)
        r = report(eq.iloc[p.trend_sma if k else 0:], tr, d.iloc[p.trend_sma if k else 0:], start_cash)
        rows.append({"fold": k + 1, **{x: r[x] for x in
                     ("period", "strategy_return_%", "buy_and_hold_%", "max_drawdown_%", "trades", "win_rate_%")}})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- paper trading
class SimBroker:
    """Local paper account: real prices, simulated fills, persisted to a JSON file."""

    def __init__(self, path, start_cash, p: Params):
        self.path, self.p = path, p
        self.state = {"cash": start_cash, "qty": 0.0}
        if os.path.exists(path):
            with open(path) as f:
                self.state = json.load(f)

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.state, f, indent=2)

    def position(self, symbol):
        return self.state["qty"]

    def cash(self):
        return self.state["cash"]

    def buy(self, symbol, usd, price):
        px = price * (1 + self.p.slippage)
        self.state["qty"] += usd * (1 - self.p.fee) / px
        self.state["cash"] -= usd
        self._save()
        return px

    def sell(self, symbol, qty, price):
        px = price * (1 - self.p.slippage)
        self.state["cash"] += qty * px * (1 - self.p.fee)
        self.state["qty"] -= qty
        self._save()
        return px


class CcxtBroker:
    """Alpaca paper or Binance Spot Testnet via ccxt sandbox mode. Never live."""

    def __init__(self, name):
        if ccxt is None:
            sys.exit("pip install ccxt")
        key_env = {"alpaca": ("ALPACA_API_KEY", "ALPACA_API_SECRET"),
                   "binance": ("BINANCE_TESTNET_KEY", "BINANCE_TESTNET_SECRET")}[name]
        key, secret = os.getenv(key_env[0]), os.getenv(key_env[1])
        if not key or not secret:
            sys.exit(f"Set {key_env[0]} and {key_env[1]} (paper/testnet keys) in your environment.")
        self.ex = getattr(ccxt, name)({"apiKey": key, "secret": secret, "enableRateLimit": True})
        self.ex.set_sandbox_mode(True)  # Alpaca -> paper-api.alpaca.markets, Binance -> testnet
        self.ex.load_markets()

    def position(self, symbol):
        base = symbol.split("/")[0]
        return float(self.ex.fetch_balance().get(base, {}).get("free") or 0)

    def cash(self):
        bal = self.ex.fetch_balance()
        return float((bal.get("USD") or bal.get("USDT") or {}).get("free") or 0)

    def buy(self, symbol, usd, price):
        amt = float(self.ex.amount_to_precision(symbol, usd / price))
        o = self.ex.create_order(symbol, "market", "buy", amt)
        return o.get("average") or price

    def sell(self, symbol, qty, price):
        amt = float(self.ex.amount_to_precision(symbol, qty))
        o = self.ex.create_order(symbol, "market", "sell", amt)
        return o.get("average") or price


TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}


def paper_loop(args, p: Params):
    data_ex = getattr(ccxt, args.data)({"enableRateLimit": True})
    broker = SimBroker(args.state, args.cash, p) if args.mode == "sim" else CcxtBroker(args.mode)
    trade_symbol = args.symbol
    if args.mode == "binance" and trade_symbol.endswith("/USD"):
        trade_symbol += "T"  # Binance testnet quotes in USDT
    meta_path = args.state + ".meta"
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {"entry": 0.0, "bars": 0}
    log = open(args.log, "a", newline="")
    writer = csv.writer(log)
    need = max(p.lookback, p.trend_sma, p.rsi_period) + 5
    print(f"[{args.mode}] trading {trade_symbol} on {args.timeframe} candles; prices from {args.data}. Ctrl+C to stop.")

    last_bar = None
    while True:
        try:
            raw = data_ex.fetch_ohlcv(args.symbol, args.timeframe, limit=need + 1)
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = add_indicators(df.set_index("timestamp").iloc[:-1], p)  # drop the still-forming candle
            row, bar_time = df.iloc[-1], df.index[-1]
            if bar_time != last_bar:
                last_bar = bar_time
                price = float(data_ex.fetch_ticker(args.symbol)["last"])
                qty = broker.position(trade_symbol)
                action = "hold"
                if qty * price > 10:  # holding a position (ignore dust)
                    meta["bars"] += 1
                    reason = exit_reason(row, meta["entry"], meta["bars"], p)
                    if reason:
                        fill = broker.sell(trade_symbol, qty, price)
                        action = f"SELL ({reason}) @ {fill:,.2f}"
                        meta = {"entry": 0.0, "bars": 0}
                elif entry_signal(row, p):
                    fill = broker.buy(trade_symbol, broker.cash() * p.risk_fraction, price)
                    action = f"BUY @ {fill:,.2f}"
                    meta = {"entry": fill, "bars": 0}
                json.dump(meta, open(meta_path, "w"))
                equity = broker.cash() + broker.position(trade_symbol) * price
                now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
                print(f"{now}  close={row.close:,.2f}  z={row.z:+.2f}  rsi={row.rsi:5.1f}  "
                      f"equity={equity:,.2f}  {action}")
                writer.writerow([now, row.close, round(row.z, 3), round(row.rsi, 1), round(equity, 2), action])
                log.flush()
        except KeyboardInterrupt:
            break
        except Exception as e:  # network blips shouldn't kill the bot
            print("error:", type(e).__name__, e)
        if args.once:
            break
        # sleep until just after the next candle closes
        step = TF_SECONDS.get(args.timeframe, 3600)
        time.sleep(step - time.time() % step + 5)


# --------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description="Mean reversion BTC bot")
    ap.add_argument("mode", choices=["backtest", "sim", "alpaca", "binance"])
    ap.add_argument("--data", default="kraken", help="ccxt exchange id for prices, or a .csv path (backtest)")
    ap.add_argument("--symbol", default="BTC/USD")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--cash", type=float, default=10_000)
    ap.add_argument("--state", default="paper_state.json")
    ap.add_argument("--log", default="paper_log.csv")
    ap.add_argument("--once", action="store_true", help="paper modes: run one cycle and exit")
    ap.add_argument("--folds", type=int, default=4, help="walk-forward folds in backtest")
    for f, v in asdict(Params()).items():
        ap.add_argument(f"--{f.replace('_', '-')}", type=type(v), default=v)
    args = ap.parse_args()
    p = Params(**{f: getattr(args, f) for f in asdict(Params())})

    if args.mode == "backtest":
        df = load_data(args.data, args.symbol, args.timeframe, args.days)
        eq, trades, d = backtest(df, p, args.cash)
        print("\n=== Backtest ===")
        for k, v in report(eq, trades, d, args.cash).items():
            print(f"{k:>18}: {v}")
        print("\n=== Walk-forward (each period tested separately) ===")
        print(walk_forward(df, p, args.cash, args.folds).to_string(index=False))
        trades.to_csv("backtest_trades.csv", index=False)
        eq.rename("equity").to_csv("backtest_equity.csv")
        print("\nSaved backtest_trades.csv and backtest_equity.csv")
    else:
        paper_loop(args, p)


if __name__ == "__main__":
    main()
