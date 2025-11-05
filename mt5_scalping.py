"""MetaTrader 5 XAUUSD scalping helper.

Bu modül, MetaTrader 5 terminalinden veri çekerek XAUUSD (Altın) için
yüksek frekanslı (scalping) işlem sinyali üretir. Strateji, çoklu zaman
ölçeğinde trend filtreleme, momentuma dayalı giriş ve volatilite tabanlı
çıkış seviyeleri içerir.

Kurulum:
    pip install MetaTrader5 pandas numpy

MetaTrader 5 terminalinin kurulu ve açık olduğundan emin olun. Gerekirse
giriş bilgilerini (account, password, server) initialize_mt5 fonksiyonuna
geçebilirsiniz.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import MetaTrader5 as mt5
import numpy as np
import pandas as pd


_LOGGER = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Stratejiye ait ayarlar."""

    symbol: str = "XAUUSD"
    trend_timeframe: int = mt5.TIMEFRAME_M5
    entry_timeframe: int = mt5.TIMEFRAME_M1
    ema_fast_period: int = 21
    ema_slow_period: int = 55
    rsi_period: int = 14
    atr_period: int = 14
    atr_multiplier: float = 0.35
    min_tick_volume: int = 50
    trend_lookback: int = 600
    entry_lookback: int = 400


@dataclass
class Signal:
    """Stratejinin ürettiği işlem sinyali."""

    symbol: str
    direction: str  # "BUY", "SELL" veya "FLAT"
    timestamp: datetime
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


def initialize_mt5(account: Optional[int] = None,
                   password: Optional[str] = None,
                   server: Optional[str] = None) -> None:
    """MetaTrader 5 bağlantısını başlat."""

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    if account is not None:
        authorized = mt5.login(account_number=account, password=password, server=server)
        if not authorized:
            raise RuntimeError(f"Login failed: {mt5.last_error()}")


def shutdown_mt5() -> None:
    """MetaTrader 5 bağlantısını kapat."""

    mt5.shutdown()


def _rates_to_dataframe(rates: np.ndarray) -> pd.DataFrame:
    if rates is None or len(rates) == 0:
        raise RuntimeError("MT5, talep edilen veri kümesini döndüremedi.")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    return df[["open", "high", "low", "close", "tick_volume"]]


def fetch_rates(symbol: str, timeframe: int, count: int) -> pd.DataFrame:
    """MetaTrader 5'ten veri al ve DataFrame'e dönüştür."""

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    return _rates_to_dataframe(rates)


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr_components = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    atr = true_range.ewm(alpha=1 / period, adjust=False).mean()
    return atr


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cumulative_tp_vol = (typical_price * df["tick_volume"]).cumsum()
    cumulative_vol = df["tick_volume"].cumsum().replace(0, np.nan)
    return cumulative_tp_vol / cumulative_vol


class ScalpingStrategy:
    """XAUUSD için çoklu zaman ölçekli scalping stratejisi."""

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    def _prepare_trend_dataframe(self) -> pd.DataFrame:
        df = fetch_rates(self.config.symbol, self.config.trend_timeframe, self.config.trend_lookback)
        df["ema_fast"] = compute_ema(df["close"], self.config.ema_fast_period)
        df["ema_slow"] = compute_ema(df["close"], self.config.ema_slow_period)
        df["rsi"] = compute_rsi(df["close"], self.config.rsi_period)
        df["atr"] = compute_atr(df, self.config.atr_period)
        df["vwap"] = compute_vwap(df)
        return df.dropna()

    def _prepare_entry_dataframe(self) -> pd.DataFrame:
        df = fetch_rates(self.config.symbol, self.config.entry_timeframe, self.config.entry_lookback)
        df["ema_fast"] = compute_ema(df["close"], int(self.config.ema_fast_period / 2))
        df["ema_slow"] = compute_ema(df["close"], self.config.ema_fast_period)
        df["rsi"] = compute_rsi(df["close"], max(7, int(self.config.rsi_period / 2)))
        df["atr"] = compute_atr(df, max(10, int(self.config.atr_period / 2)))
        df["vwap"] = compute_vwap(df)
        return df.dropna()

    def generate_signal(self) -> Signal:
        trend_df = self._prepare_trend_dataframe()
        entry_df = self._prepare_entry_dataframe()

        if trend_df.empty or entry_df.empty:
            raise RuntimeError("Gerekli veri hazırlanamadı.")

        trend_row = trend_df.iloc[-1]
        entry_row = entry_df.iloc[-1]

        trend_bias = "BULLISH" if trend_row["ema_fast"] > trend_row["ema_slow"] else "BEARISH"
        atr_value = entry_row["atr"]

        if np.isnan(atr_value) or atr_value == 0:
            raise RuntimeError("ATR hesaplanamadı veya 0 çıktı.")

        price = entry_row["close"]
        stop_loss = None
        take_profit = None
        direction = "FLAT"

        if entry_row["tick_volume"] < self.config.min_tick_volume:
            _LOGGER.info("Yetersiz hacim: tick_volume=%s", entry_row["tick_volume"])
            return Signal(self.config.symbol, direction, entry_row.name.to_pydatetime(), price)

        # Alış sinyali
        if (
            trend_bias == "BULLISH"
            and entry_row["ema_fast"] > entry_row["ema_slow"]
            and entry_row["rsi"] > 55
            and price > entry_row["vwap"]
        ):
            direction = "BUY"
            stop_loss = price - self.config.atr_multiplier * atr_value
            take_profit = price + self.config.atr_multiplier * 1.5 * atr_value

        # Satış sinyali
        elif (
            trend_bias == "BEARISH"
            and entry_row["ema_fast"] < entry_row["ema_slow"]
            and entry_row["rsi"] < 45
            and price < entry_row["vwap"]
        ):
            direction = "SELL"
            stop_loss = price + self.config.atr_multiplier * atr_value
            take_profit = price - self.config.atr_multiplier * 1.5 * atr_value

        timestamp = entry_row.name.to_pydatetime().astimezone(timezone.utc)

        return Signal(
            symbol=self.config.symbol,
            direction=direction,
            timestamp=timestamp,
            price=float(price),
            stop_loss=float(stop_loss) if stop_loss is not None else None,
            take_profit=float(take_profit) if take_profit is not None else None,
        )


def example_usage() -> None:
    """Basit kullanım örneği."""

    logging.basicConfig(level=logging.INFO)

    try:
        initialize_mt5()
        strategy = ScalpingStrategy(StrategyConfig())
        signal = strategy.generate_signal()

        print("Sinyal:", signal)
        if signal.direction == "BUY":
            print("Trend yukarı. Scalping için long pozisyon düşünülebilir.")
        elif signal.direction == "SELL":
            print("Trend aşağı. Scalping için short pozisyon düşünülebilir.")
        else:
            print("Sinyal nötr. Pozisyona girmek için koşullar beklenmeli.")

    finally:
        shutdown_mt5()


if __name__ == "__main__":
    example_usage()
