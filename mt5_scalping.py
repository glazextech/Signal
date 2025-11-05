"""MetaTrader5-powered scalping signal generator for XAUUSD.

This script connects to a running MetaTrader 5 terminal, fetches recent
minute-level price data for XAUUSD, computes short-term technical indicators,
and emits actionable scalping signals complete with suggested risk parameters.

Usage example:

    python mt5_scalping.py --symbol XAUUSD --minutes 720 --risk-reward 1.8

Prerequisites:
    pip install MetaTrader5 pandas

You must have the MetaTrader 5 terminal installed and logged in on the same
machine, or provide account credentials via CLI arguments or environment
variables.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

import pandas as pd


try:
    import MetaTrader5 as mt5  # type: ignore
except ImportError as exc:  # pragma: no cover - informative exit
    raise SystemExit(
        "MetaTrader5 package is required. Install with 'pip install MetaTrader5'."
    ) from exc


logger = logging.getLogger(__name__)


# Default indicator settings tuned for short-term (scalping) setups.
EMA_FAST_PERIOD = 9
EMA_SLOW_PERIOD = 21
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
ATR_PERIOD = 14


@dataclass
class TradeSignal:
    """Structured trading signal information."""

    direction: str
    timestamp: pd.Timestamp
    entry: float
    stop_loss: float
    take_profit: float
    comment: str


def initialize_mt5(
    *,
    login: Optional[int] = None,
    password: Optional[str] = None,
    server: Optional[str] = None,
) -> None:
    """Initialize the MetaTrader 5 terminal connection."""

    if mt5.initialize(login=login, password=password, server=server):
        logger.info("MetaTrader 5 initialized successfully.")
        return

    error_code, error_detail = mt5.last_error()
    raise RuntimeError(
        f"Failed to initialize MetaTrader5 (code={error_code}): {error_detail}"
    )


def shutdown_mt5() -> None:
    """Safely close the MetaTrader 5 terminal connection."""

    if not mt5.shutdown():
        error_code, error_detail = mt5.last_error()
        logger.warning(
            "Failed to shut down MetaTrader5 cleanly (code=%s): %s",
            error_code,
            error_detail,
        )


def ensure_symbol(symbol: str) -> None:
    """Make sure the requested symbol is available in the Market Watch."""

    info = mt5.symbol_info(symbol)
    if info is None:
        raise ValueError(f"Symbol {symbol} not found in MetaTrader 5")

    if info.visible:
        return

    if not mt5.symbol_select(symbol, True):
        error_code, error_detail = mt5.last_error()
        raise RuntimeError(
            f"Unable to select symbol {symbol} (code={error_code}): {error_detail}"
        )


def fetch_rates(
    symbol: str,
    *,
    minutes: int,
    timeframe: int = mt5.TIMEFRAME_M1,
) -> pd.DataFrame:
    """Fetch OHLCV data from MetaTrader 5 as a Pandas DataFrame."""

    ensure_symbol(symbol)

    utc_now = datetime.now(timezone.utc)
    utc_from = utc_now - timedelta(minutes=minutes)

    logger.info(
        "Requesting %s minutes of %s data (timeframe=%s)", minutes, symbol, timeframe
    )

    rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_now)
    if rates is None or len(rates) == 0:
        raise RuntimeError("No price data returned from MetaTrader 5")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    return df


def exponential_moving_average(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def relative_strength_index(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def average_true_range(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int
) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = ranges.max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Append scalping indicators to the OHLCV DataFrame."""

    indicators = df.copy()
    close = indicators["close"]
    indicators["ema_fast"] = exponential_moving_average(close, EMA_FAST_PERIOD)
    indicators["ema_slow"] = exponential_moving_average(close, EMA_SLOW_PERIOD)
    indicators["rsi"] = relative_strength_index(close, RSI_PERIOD)
    indicators["atr"] = average_true_range(
        indicators["high"], indicators["low"], close, ATR_PERIOD
    )
    indicators.dropna(inplace=True)
    return indicators


def generate_signals(
    indicators: pd.DataFrame,
    *,
    risk_reward: float,
) -> Iterable[TradeSignal]:
    """Yield scalping trade signals based on indicator state."""

    fast = indicators["ema_fast"]
    slow = indicators["ema_slow"]
    rsi = indicators["rsi"]
    atr = indicators["atr"]
    close = indicators["close"]

    long_cross = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    short_cross = (fast < slow) & (fast.shift(1) >= slow.shift(1))

    long_signals = indicators[long_cross & (rsi < RSI_OVERBOUGHT)].copy()
    short_signals = indicators[short_cross & (rsi > RSI_OVERSOLD)].copy()

    for timestamp, row in long_signals.iterrows():
        entry = float(row["close"])
        stop = entry - float(row["atr"]) * 1.2
        target = entry + (entry - stop) * risk_reward
        comment = "Bullish momentum aligns with RSI confirmation"
        yield TradeSignal(
            direction="LONG",
            timestamp=pd.Timestamp(timestamp),
            entry=entry,
            stop_loss=stop,
            take_profit=target,
            comment=comment,
        )

    for timestamp, row in short_signals.iterrows():
        entry = float(row["close"])
        stop = entry + float(row["atr"]) * 1.2
        target = entry - (stop - entry) * risk_reward
        comment = "Bearish momentum aligns with RSI confirmation"
        yield TradeSignal(
            direction="SHORT",
            timestamp=pd.Timestamp(timestamp),
            entry=entry,
            stop_loss=stop,
            take_profit=target,
            comment=comment,
        )


def format_signal(signal: TradeSignal) -> str:
    return (
        f"[{signal.timestamp.tz_convert('UTC'):%Y-%m-%d %H:%M}] {signal.direction} "
        f"entry={signal.entry:.2f} SL={signal.stop_loss:.2f} TP={signal.take_profit:.2f} "
        f"| {signal.comment}"
    )


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="XAUUSD", help="Symbol to analyze")
    parser.add_argument(
        "--minutes",
        type=int,
        default=720,
        help="Lookback window (in minutes) for the analysis",
    )
    parser.add_argument(
        "--risk-reward",
        type=float,
        default=1.5,
        help="Target risk-reward ratio for projected take-profit",
    )
    parser.add_argument(
        "--login",
        type=int,
        default=None,
        help="Optional MT5 account login; defaults to active terminal session",
    )
    parser.add_argument("--password", default=None, help="Optional MT5 password")
    parser.add_argument("--server", default=None, help="Optional MT5 trade server")
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        initialize_mt5(login=args.login, password=args.password, server=args.server)
    except Exception as exc:
        logger.error("Initialization error: %s", exc)
        return 1

    try:
        raw_data = fetch_rates(args.symbol, minutes=args.minutes)
        indicators = compute_indicators(raw_data)
        signals = list(generate_signals(indicators, risk_reward=args.risk_reward))
    except Exception as exc:
        logger.error("Failed to produce signals: %s", exc)
        return 1
    finally:
        shutdown_mt5()

    if not signals:
        logger.info("No fresh scalping signals detected in the lookback window.")
        return 0

    logger.info("Generated %d signal(s).", len(signals))
    for signal in signals[-5:]:  # limit output for readability
        print(format_signal(signal))

    last_signal = signals[-1]
    print("\nMost recent plan:")
    print(
        f"  Direction : {last_signal.direction}\n"
        f"  Entry     : {last_signal.entry:.2f}\n"
        f"  Stop Loss : {last_signal.stop_loss:.2f}\n"
        f"  Take Profit: {last_signal.take_profit:.2f}\n"
        f"  Timestamp : {last_signal.timestamp.tz_convert('UTC'):%Y-%m-%d %H:%M}%UTC\n"
        f"  Notes     : {last_signal.comment}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

