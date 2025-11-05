"""Binance spot trading signal bot.

This script downloads recent OHLCV data from Binance's public REST API,
derives technical indicators, and produces trading signals together with
stop-loss and take-profit levels. The strategy implemented is:

- Use the 50-period and 200-period exponential moving averages (EMAs) on
  closing prices to determine trend direction.
- Require a bullish (or bearish) crossover confirmation between the EMAs.
- Filter signals with the 14-period Relative Strength Index (RSI) to avoid
  exhausted momentum (RSI > 55 for long entries, RSI < 45 for short entries).
- Size stop-loss and take-profit levels using the 14-period Average True Range
  (ATR) to adapt to recent volatility. Risk-to-reward is set at 1:1.5 (stop
  1.0 × ATR, target 1.5 × ATR).

Only the strongest signal (based on risk/reward and momentum confidence) is
reported back to the user.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests


BINANCE_API_URL = "https://api.binance.com"


@dataclasses.dataclass
class Signal:
    symbol: str
    direction: str  # "long" or "short"
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    indicators: Dict[str, float]
    generated_at: dt.datetime

    def as_dict(self) -> Dict[str, object]:
        data = dataclasses.asdict(self)
        data["generated_at"] = self.generated_at.isoformat()
        return data


def fetch_klines(symbol: str, interval: str = "1h", limit: int = 500) -> pd.DataFrame:
    """Fetch OHLCV kline data for a symbol/interval from Binance."""

    url = f"{BINANCE_API_URL}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    raw = response.json()

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]

    df = pd.DataFrame(raw, columns=columns)
    numeric_cols = ["open", "high", "low", "close", "volume", "quote_asset_volume", "taker_buy_base", "taker_buy_quote"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = ema(df["close"], 50)
    df["ema_slow"] = ema(df["close"], 200)
    df["rsi"] = rsi(df["close"], 14)
    df["atr"] = atr(df, 14)
    return df


def detect_signal(df: pd.DataFrame, symbol: str) -> Optional[Signal]:
    # Use the most recently completed candle
    if len(df) < 210:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]

    ema_fast_now = last["ema_fast"]
    ema_fast_prev = prev["ema_fast"]
    ema_slow_now = last["ema_slow"]
    ema_slow_prev = prev["ema_slow"]
    rsi_now = last["rsi"]
    atr_now = last["atr"]

    if any(math.isnan(x) for x in [ema_fast_now, ema_fast_prev, ema_slow_now, ema_slow_prev, rsi_now, atr_now]):
        return None

    close_price = last["close"]

    # Check for bullish signal
    bullish_cross = ema_fast_prev <= ema_slow_prev and ema_fast_now > ema_slow_now
    bearish_cross = ema_fast_prev >= ema_slow_prev and ema_fast_now < ema_slow_now

    signal: Optional[Signal] = None

    if bullish_cross and rsi_now > 55:
        stop_loss = close_price - atr_now
        take_profit = close_price + 1.5 * atr_now
        if stop_loss <= 0:
            return None
        signal = Signal(
            symbol=symbol,
            direction="long",
            entry_price=close_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=(take_profit - close_price) / (close_price - stop_loss),
            indicators={
                "ema_fast": float(ema_fast_now),
                "ema_slow": float(ema_slow_now),
                "rsi": float(rsi_now),
                "atr": float(atr_now),
            },
            generated_at=dt.datetime.now(dt.timezone.utc),
        )

    elif bearish_cross and rsi_now < 45:
        stop_loss = close_price + atr_now
        take_profit = close_price - 1.5 * atr_now
        signal = Signal(
            symbol=symbol,
            direction="short",
            entry_price=close_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=(close_price - take_profit) / (stop_loss - close_price),
            indicators={
                "ema_fast": float(ema_fast_now),
                "ema_slow": float(ema_slow_now),
                "rsi": float(rsi_now),
                "atr": float(atr_now),
            },
            generated_at=dt.datetime.now(dt.timezone.utc),
        )

    return signal


def evaluate_symbols(symbols: Iterable[str], interval: str = "1h") -> List[Signal]:
    signals: List[Signal] = []
    for symbol in symbols:
        try:
            df = fetch_klines(symbol, interval=interval, limit=500)
            df = prepare_indicators(df)
            signal = detect_signal(df, symbol)
            if signal:
                signals.append(signal)
        except requests.RequestException as exc:
            print(f"[warn] Network error for {symbol}: {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[warn] Failed to process {symbol}: {exc}")
    return signals


def pick_best_signal(signals: List[Signal]) -> Optional[Signal]:
    if not signals:
        return None

    def score(sig: Signal) -> float:
        momentum_score = sig.indicators.get("rsi", 0)
        if sig.direction == "short":
            momentum_score = 100 - momentum_score
        return sig.risk_reward * 0.6 + (momentum_score / 100.0) * 0.4

    return max(signals, key=score)


def format_signal(signal: Signal) -> str:
    entry = signal.entry_price
    stop = signal.stop_loss
    target = signal.take_profit
    rr = signal.risk_reward
    indicators = ", ".join(f"{k}={v:.2f}" for k, v in signal.indicators.items())
    return (
        f"Best opportunity: {signal.symbol} ({signal.direction.upper()})\n"
        f"  Entry: {entry:.4f}\n"
        f"  Stop-loss: {stop:.4f}\n"
        f"  Take-profit: {target:.4f}\n"
        f"  Risk/Reward: {rr:.2f}\n"
        f"  Indicators: {indicators}\n"
        f"  Generated at: {signal.generated_at.isoformat()}"
    )


def main():
    # Universe of liquid spot symbols; adjust as needed.
    symbols = [
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "SOLUSDT",
        "XRPUSDT",
        "ADAUSDT",
        "DOGEUSDT",
    ]

    interval = "1h"
    signals = evaluate_symbols(symbols, interval=interval)
    best = pick_best_signal(signals)

    if best:
        print(format_signal(best))
    else:
        print("No qualified signals at this time. Try a different interval or wait for new data.")


if __name__ == "__main__":
    main()

