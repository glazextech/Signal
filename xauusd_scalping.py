"""MetaTrader 5 XAUUSD scalping signal generator.

This script connects to a running MetaTrader 5 terminal, loads 1-minute gold
price data, calculates EMA crossover, RSI confirmation, and MACD confirmation,
and prints actionable BUY/SELL/HOLD signals every minute. The logic is designed
for manual trading; no orders are sent to MT5. All parameters are collected in a
configuration dataclass so you can easily tweak indicator periods or thresholds.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import MetaTrader5 as mt5  # type: ignore
import numpy as np
import pandas as pd

try:  # Prefer TA-Lib if available; fall back to pandas implementations otherwise.
    import talib  # type: ignore
except ImportError:  # pragma: no cover - TA-Lib may not be installed locally.
    talib = None


@dataclass
class StrategyConfig:
    """Runtime parameters for the scalping bot."""

    symbol: str = "XAUUSD"
    timeframe: int = mt5.TIMEFRAME_M1
    history_bars: int = 400  # ~6.5 hours of data to stabilise indicators
    update_interval: int = 60  # seconds between signal evaluations
    ema_fast_period: int = 9
    ema_slow_period: int = 21
    rsi_period: int = 14
    rsi_bull_threshold: float = 50.0
    rsi_bear_threshold: float = 50.0
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9
    reconnect_delay: int = 5  # seconds to wait before attempting reconnection


@dataclass
class SignalSnapshot:
    """Container for the signal decision and diagnostic context."""

    action: str
    ema_relation: str
    rsi_value: float
    macd_state: str
    notes: str = ""


class IndicatorCalculator:
    """Utility helpers for computing EMA, RSI, MACD with optional TA-Lib support."""

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        if talib is not None:
            values = talib.EMA(series.values.astype(float), timeperiod=period)
            return pd.Series(values, index=series.index, dtype=float)
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int) -> pd.Series:
        if talib is not None:
            values = talib.RSI(series.values.astype(float), timeperiod=period)
            return pd.Series(values, index=series.index, dtype=float)

        delta = series.diff()
        gain = delta.clip(lower=0.0).ewm(alpha=1 / period, adjust=False).mean()
        loss = (-delta.clip(upper=0.0)).ewm(alpha=1 / period, adjust=False).mean()
        rs = gain / loss.replace(0.0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def macd(series: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
        if talib is not None:
            macd, macd_signal, macd_hist = talib.MACD(
                series.values.astype(float),
                fastperiod=fast,
                slowperiod=slow,
                signalperiod=signal,
            )
        else:
            ema_fast = series.ewm(span=fast, adjust=False).mean()
            ema_slow = series.ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal, adjust=False).mean()
            macd_hist = macd - macd_signal

        return pd.DataFrame(
            {
                "macd": macd,
                "macd_signal": macd_signal,
                "macd_hist": macd_hist,
            },
            index=series.index,
        )

    @classmethod
    def enrich(cls, prices: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
        enriched = prices.copy()
        enriched["ema_fast"] = cls.ema(enriched["close"], config.ema_fast_period)
        enriched["ema_slow"] = cls.ema(enriched["close"], config.ema_slow_period)
        enriched["rsi"] = cls.rsi(enriched["close"], config.rsi_period)

        macd_df = cls.macd(
            enriched["close"],
            config.macd_fast_period,
            config.macd_slow_period,
            config.macd_signal_period,
        )
        enriched = enriched.join(macd_df)
        return enriched.dropna()


class SignalEngine:
    """Generate entry/exit/hold decisions based on indicator state."""

    @staticmethod
    def evaluate(prices: pd.DataFrame, position: Optional[str], config: StrategyConfig) -> SignalSnapshot:
        latest = prices.iloc[-1]
        previous = prices.iloc[-2]

        bullish_cross = previous["ema_fast"] <= previous["ema_slow"] and latest["ema_fast"] > latest["ema_slow"]
        bearish_cross = previous["ema_fast"] >= previous["ema_slow"] and latest["ema_fast"] < latest["ema_slow"]

        macd_diff = latest["macd"] - latest["macd_signal"]
        prev_macd_diff = previous["macd"] - previous["macd_signal"]
        macd_cross_up = prev_macd_diff <= 0 <= macd_diff and macd_diff > 0
        macd_cross_down = prev_macd_diff >= 0 >= macd_diff and macd_diff < 0

        ema_relation = ">" if latest["ema_fast"] > latest["ema_slow"] else "<" if latest["ema_fast"] < latest["ema_slow"] else "="
        macd_state = "bullish" if macd_diff > 0 else "bearish" if macd_diff < 0 else "neutral"

        rsi_value = float(latest["rsi"])
        notes: list[str] = []

        # Exit logic takes precedence when a position is open.
        if position == "long":
            if bearish_cross or rsi_value >= config.rsi_overbought:
                if bearish_cross:
                    notes.append("EMA crossover reversed")
                if rsi_value >= config.rsi_overbought:
                    notes.append("RSI overbought")
                return SignalSnapshot("EXIT LONG", ema_relation, rsi_value, macd_state, "; ".join(notes))
        elif position == "short":
            if bullish_cross or rsi_value <= config.rsi_oversold:
                if bullish_cross:
                    notes.append("EMA crossover reversed")
                if rsi_value <= config.rsi_oversold:
                    notes.append("RSI oversold")
                return SignalSnapshot("EXIT SHORT", ema_relation, rsi_value, macd_state, "; ".join(notes))

        # Entry logic when flat.
        if position is None:
            if bullish_cross and rsi_value > config.rsi_bull_threshold and macd_cross_up:
                notes.extend(["9 EMA above 21 EMA", "RSI bullish", "MACD bull cross"])
                return SignalSnapshot("BUY", ema_relation, rsi_value, macd_state, "; ".join(notes))
            if bearish_cross and rsi_value < config.rsi_bear_threshold and macd_cross_down:
                notes.extend(["9 EMA below 21 EMA", "RSI bearish", "MACD bear cross"])
                return SignalSnapshot("SELL", ema_relation, rsi_value, macd_state, "; ".join(notes))

        # Otherwise hold position (open positions keep status quo).
        return SignalSnapshot("HOLD", ema_relation, rsi_value, macd_state)

    @staticmethod
    def next_position(current: Optional[str], action: str) -> Optional[str]:
        if action == "BUY":
            return "long"
        if action == "SELL":
            return "short"
        if action.startswith("EXIT"):
            return None
        return current


class ScalpingBot:
    """Co-ordinates connection, data retrieval, and signal output."""

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config
        self.position: Optional[str] = None

    def run(self) -> None:
        try:
            self._connect()
            logging.info("Connected to MetaTrader 5 and subscribed to %s", self.config.symbol)

            while True:
                cycle_start = time.time()
                try:
                    prices = self._load_prices()
                    enriched = IndicatorCalculator.enrich(prices, self.config)
                    if len(enriched) < 2:
                        raise RuntimeError("Not enough data points after indicator warm-up")

                    signal = SignalEngine.evaluate(enriched, self.position, self.config)
                    self._print_signal(signal)
                    self.position = SignalEngine.next_position(self.position, signal.action)
                except Exception as error:  # broad on purpose so the loop keeps running
                    logging.exception("Signal evaluation failed: %s", error)
                    self._recover_connection()

                self._sleep_until_next_cycle(cycle_start)
        except KeyboardInterrupt:
            logging.info("Shutdown requested by user")
        finally:
            mt5.shutdown()
            logging.info("MetaTrader 5 connection closed")

    def _connect(self) -> None:
        if not mt5.initialize():
            raise RuntimeError(f"Failed to initialize MetaTrader 5: {mt5.last_error()}")
        if not mt5.symbol_select(self.config.symbol, True):
            raise RuntimeError(f"Unable to subscribe to symbol {self.config.symbol}")

    def _recover_connection(self) -> None:
        logging.info("Attempting to reconnect in %s seconds...", self.config.reconnect_delay)
        mt5.shutdown()
        time.sleep(self.config.reconnect_delay)
        self._connect()

    def _load_prices(self) -> pd.DataFrame:
        rates = mt5.copy_rates_from_pos(
            self.config.symbol,
            self.config.timeframe,
            0,
            self.config.history_bars,
        )
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No rates returned for {self.config.symbol}: {mt5.last_error()}")

        frame = pd.DataFrame(rates)
        frame["time"] = pd.to_datetime(frame["time"], unit="s")
        frame.set_index("time", inplace=True)
        return frame

    def _print_signal(self, signal: SignalSnapshot) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        ema_text = f"EMA: {self.config.ema_fast_period}{signal.ema_relation}{self.config.ema_slow_period}"
        rsi_text = f"RSI: {signal.rsi_value:.2f}"
        macd_text = f"MACD: {signal.macd_state}"
        parts = [f"[{timestamp}] Signal: {signal.action}", ema_text, rsi_text, macd_text]
        if signal.notes:
            parts.append(f"Notes: {signal.notes}")
        print(" | ".join(parts), flush=True)

    def _sleep_until_next_cycle(self, cycle_start: float) -> None:
        elapsed = time.time() - cycle_start
        sleep_for = max(0.0, self.config.update_interval - elapsed)
        time.sleep(sleep_for)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate manual scalping signals for XAUUSD on a 1-minute chart",
    )
    parser.add_argument("--history", type=int, default=400, help="Number of 1-minute bars to fetch each cycle")
    parser.add_argument("--interval", type=int, default=60, help="Seconds between signal refreshes (default: 60)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = StrategyConfig(history_bars=args.history, update_interval=args.interval)
    bot = ScalpingBot(config)
    bot.run()


if __name__ == "__main__":
    main()

