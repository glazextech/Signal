"""Automated XAUUSD scalping signal generator using MetaTrader5.

This module connects to a running MetaTrader 5 terminal, downloads recent
price data for XAUUSD, computes a set of fast technical indicators that are
well-suited to intraday scalping, and emits actionable trade signals.  The
script is intentionally modular so you can either:

* run it as a standalone helper that simply prints the latest signal, or
* import the module and integrate `generate_signal` / `prepare_order_request`
  into a larger trade-management workflow.

Usage (shell):
    python xauusd_scalping.py --account 123456 --password secret --server "Broker-Server"

Prerequisites:
    pip install MetaTrader5 pandas numpy

IMPORTANT:
    - Make sure MetaTrader 5 is installed and logged in to the broker account.
    - Allow algorithmic trading in the MT5 terminal.
    - Run Python in the same architecture (32/64 bit) as the MT5 terminal.
    - Trading leveraged products carries significant risk; test thoroughly on
      a demo account before deploying to live capital.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import MetaTrader5 as mt5  # type: ignore
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class StrategyConfig:
    account: int
    password: str
    server: str
    symbol: str = "XAUUSD"
    timeframe: int = mt5.TIMEFRAME_M1
    lookback: int = 600  # fetch 10 hours of 1-minute candles
    lot: float = 0.10
    max_spread_points: float = 30.0
    risk_per_trade: float = 0.005  # 0.5% of equity
    atr_period: int = 14
    ema_fast_period: int = 9
    ema_slow_period: int = 21
    rsi_period: int = 14
    rsi_upper: float = 65.0
    rsi_lower: float = 35.0
    reward_risk_ratio: float = 1.5


# ---------------------------------------------------------------------------
# MT5 helpers
# ---------------------------------------------------------------------------


def initialize_mt5(config: StrategyConfig) -> None:
    """Initialise MT5 terminal and login to the account."""

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed, error code: {mt5.last_error()}")

    authorized = mt5.login(config.account, password=config.password, server=config.server)
    if not authorized:
        last_error = mt5.last_error()
        mt5.shutdown()
        raise RuntimeError(
            f"MT5 login failed (account={config.account}), error: {last_error}"
        )


def shutdown_mt5() -> None:
    """Gracefully close MT5 API connection."""

    mt5.shutdown()


def fetch_rates(config: StrategyConfig) -> pd.DataFrame:
    """Fetch recent price candles for the configured symbol/timeframe."""

    rates = mt5.copy_rates_from_pos(
        config.symbol,
        config.timeframe,
        0,
        config.lookback,
    )

    if rates is None or len(rates) == 0:
        raise RuntimeError(f"Failed to fetch rates for {config.symbol}: {mt5.last_error()}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    return df


# ---------------------------------------------------------------------------
# Indicator engine
# ---------------------------------------------------------------------------


def compute_indicators(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Append EMA, RSI, ATR indicators to the price DataFrame."""

    prices = df.copy()

    prices["ema_fast"] = prices["close"].ewm(span=config.ema_fast_period, adjust=False).mean()
    prices["ema_slow"] = prices["close"].ewm(span=config.ema_slow_period, adjust=False).mean()

    delta = prices["close"].diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1 / config.rsi_period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / config.rsi_period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    prices["rsi"] = 100 - (100 / (1 + rs))

    tr = np.maximum(
        prices["high"] - prices["low"],
        np.maximum(
            prices["high"] - prices["close"].shift(1),
            prices["close"].shift(1) - prices["low"],
        ),
    )
    prices["atr"] = tr.rolling(window=config.atr_period, min_periods=1).mean()

    return prices


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------


@dataclass
class TradeSignal:
    direction: str
    timestamp: datetime
    entry: float
    stop_loss: float
    take_profit: float
    comment: str


def generate_signal(prices: pd.DataFrame, config: StrategyConfig) -> Optional[TradeSignal]:
    """Generate a trade signal based on EMA crossover + RSI filter + volatility."""

    latest = prices.iloc[-1]
    previous = prices.iloc[-2]

    spread_points = (latest["ask"] - latest["bid"]) / mt5.symbol_info(config.symbol).point
    if spread_points > config.max_spread_points:
        logging.info("Spread %.1f exceeds threshold %.1f, skipping signal.", spread_points, config.max_spread_points)
        return None

    # EMA crossover logic
    bullish_cross = previous["ema_fast"] <= previous["ema_slow"] and latest["ema_fast"] > latest["ema_slow"]
    bearish_cross = previous["ema_fast"] >= previous["ema_slow"] and latest["ema_fast"] < latest["ema_slow"]

    atr_points = latest["atr"]
    point = mt5.symbol_info(config.symbol).point

    if bullish_cross and latest["rsi"] < config.rsi_upper:
        entry = latest["ask"]
        stop_loss = entry - 1.5 * atr_points
        take_profit = entry + config.reward_risk_ratio * (entry - stop_loss)
        return TradeSignal(
            direction="buy",
            timestamp=latest.name.to_pydatetime(),
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment="EMA bull cross + RSI filter",
        )

    if bearish_cross and latest["rsi"] > config.rsi_lower:
        entry = latest["bid"]
        stop_loss = entry + 1.5 * atr_points
        take_profit = entry - config.reward_risk_ratio * (stop_loss - entry)
        return TradeSignal(
            direction="sell",
            timestamp=latest.name.to_pydatetime(),
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment="EMA bear cross + RSI filter",
        )

    return None


def prepare_order_request(signal: TradeSignal, config: StrategyConfig) -> dict:
    """Build a ready-to-submit MT5 order request payload."""

    symbol_info = mt5.symbol_info(config.symbol)
    if symbol_info is None:
        raise RuntimeError(f"Symbol info for {config.symbol} not available")

    sl_points = abs(signal.entry - signal.stop_loss) / symbol_info.point
    volume = max(
        round((config.risk_per_trade * mt5.account_info().equity) / (sl_points * symbol_info.trade_tick_value), 2),
        0.01,
    )

    return {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": config.symbol,
        "volume": volume if volume <= symbol_info.volume_max else symbol_info.volume_max,
        "type": mt5.ORDER_TYPE_BUY if signal.direction == "buy" else mt5.ORDER_TYPE_SELL,
        "price": signal.entry,
        "sl": signal.stop_loss,
        "tp": signal.take_profit,
        "deviation": 10,
        "magic": 50817,
        "comment": signal.comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": symbol_info.filling_mode,
    }


# ---------------------------------------------------------------------------
# CLI utilities
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MetaTrader5 XAUUSD scalping signal generator")
    parser.add_argument("--account", type=int, required=True)
    parser.add_argument("--password", type=str, required=True)
    parser.add_argument("--server", type=str, required=True)
    parser.add_argument("--lots", type=float, default=0.10, help="Default lot size for order template")
    parser.add_argument(
        "--max-spread",
        type=float,
        default=30.0,
        help="Maximum spread in points to accept a trade (default: 30)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    config = StrategyConfig(
        account=args.account,
        password=args.password,
        server=args.server,
        lot=args.lots,
        max_spread_points=args.max_spread,
    )

    try:
        initialize_mt5(config)
        logging.info("Connected to MT5 and authenticated (account: %s)", config.account)

        raw = fetch_rates(config)
        symbol_info = mt5.symbol_info_tick(config.symbol)
        if symbol_info is None:
            raise RuntimeError(f"Symbol tick info for {config.symbol} unavailable")

        raw["bid"] = symbol_info.bid
        raw["ask"] = symbol_info.ask

        enriched = compute_indicators(raw, config)
        signal = generate_signal(enriched, config)

        if signal:
            request = prepare_order_request(signal, config)
            logging.info(
                "Signal: %s | entry=%.2f sl=%.2f tp=%.2f | comment=%s",
                signal.direction.upper(),
                signal.entry,
                signal.stop_loss,
                signal.take_profit,
                signal.comment,
            )
            logging.info("Suggested order payload: %s", request)
        else:
            logging.info("No valid signal at %s", datetime.now(timezone.utc))

    finally:
        shutdown_mt5()


if __name__ == "__main__":
    main()

