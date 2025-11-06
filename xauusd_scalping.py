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
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import MetaTrader5 as mt5
import numpy as np
import pandas as pd


_LOGGER = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Stratejiye ait ayarlar."""

    symbol: str = "GOLD#"
    trend_timeframe: int = mt5.TIMEFRAME_M5
    entry_timeframe: int = mt5.TIMEFRAME_M1
    ema_fast_period: int = 21
    ema_slow_period: int = 55
    entry_ema_fast_period: int = 13
    entry_ema_slow_period: int = 34
    rsi_period: int = 14
    entry_rsi_period: int = 9
    atr_period: int = 14
    entry_atr_period: int = 10
    atr_multiplier_stop: float = 0.8
    atr_multiplier_take_profit: float = 1.9
    min_tick_volume: int = 60
    volume_window: int = 120
    min_volume_quantile: float = 0.55
    trend_lookback: int = 800
    entry_lookback: int = 500
    min_trend_strength: float = 0.35
    trend_rsi_bullish: float = 55.0
    trend_rsi_bearish: float = 45.0
    rsi_buy_threshold: float = 57.0
    rsi_sell_threshold: float = 43.0
    min_momentum_ratio: float = 0.25
    max_vwap_distance_atr: float = 1.25
    min_atr_ratio: float = 0.0006
    max_atr_ratio: float = 0.0045
    breakout_lookback: int = 15
    cooldown_seconds: int = 120
    max_open_positions: int = 3
    lot_buy: float = 0.25
    lot_sell: float = 0.08
    deviation_points: int = 20


@dataclass
class Signal:
    """Stratejinin ürettiği işlem sinyali."""

    symbol: str
    direction: str  # "BUY", "SELL" veya "FLAT"
    timestamp: datetime
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 0.0
    comment: str = ""


def initialize_mt5(account: Optional[int] = None,
                   password: Optional[str] = None,
                   server: Optional[str] = None) -> None:
    """MetaTrader 5 bağlantısını başlat."""

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    if account is not None:
        authorized = mt5.login(login=account, password=password, server=server)
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
        self._last_signal_time: Optional[datetime] = None
        self._last_direction: Optional[str] = None

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
        df["ema_fast"] = compute_ema(df["close"], self.config.entry_ema_fast_period)
        df["ema_slow"] = compute_ema(df["close"], self.config.entry_ema_slow_period)
        df["rsi"] = compute_rsi(df["close"], self.config.entry_rsi_period)
        df["atr"] = compute_atr(df, self.config.entry_atr_period)
        df["vwap"] = compute_vwap(df)
        return df.dropna()

    def _volume_threshold(self, entry_df: pd.DataFrame) -> float:
        if entry_df.empty:
            return float(self.config.min_tick_volume)
        window = min(len(entry_df), self.config.volume_window)
        recent = entry_df["tick_volume"].tail(window)
        if len(recent) < 10:
            return float(self.config.min_tick_volume)
        quantile_value = float(np.quantile(recent, self.config.min_volume_quantile))
        return max(self.config.min_tick_volume, quantile_value)

    def _cooldown_active(self, current_time: datetime) -> bool:
        if self._last_signal_time is None:
            return False
        delta = current_time - self._last_signal_time
        return delta < timedelta(seconds=self.config.cooldown_seconds)

    def generate_signal(self) -> Signal:
        trend_df = self._prepare_trend_dataframe()
        entry_df = self._prepare_entry_dataframe()

        if trend_df.empty or entry_df.empty:
            raise RuntimeError("Gerekli veri hazırlanamadı.")

        trend_row = trend_df.iloc[-1]
        entry_row = entry_df.iloc[-1]
        timestamp = entry_row.name.to_pydatetime().astimezone(timezone.utc)

        if self._cooldown_active(timestamp):
            return Signal(self.config.symbol, "FLAT", timestamp, float(entry_row["close"]), comment="Cooldown aktif")

        atr_value = float(entry_row["atr"])
        if np.isnan(atr_value) or atr_value <= 0:
            raise RuntimeError("ATR hesaplanamadı veya 0 çıktı.")

        atr_ratio = atr_value / float(entry_row["close"])
        if atr_ratio < self.config.min_atr_ratio or atr_ratio > self.config.max_atr_ratio:
            return Signal(self.config.symbol, "FLAT", timestamp, float(entry_row["close"]), comment="ATR oranı uygun değil")

        volume_threshold = self._volume_threshold(entry_df)
        if entry_row["tick_volume"] < volume_threshold:
            return Signal(self.config.symbol, "FLAT", timestamp, float(entry_row["close"]), comment="Hacim yetersiz")

        trend_bias = "BULLISH" if trend_row["ema_fast"] > trend_row["ema_slow"] else "BEARISH"
        trend_strength = (trend_row["ema_fast"] - trend_row["ema_slow"]) / max(trend_row["atr"], 1e-6)

        if abs(trend_strength) < self.config.min_trend_strength:
            return Signal(self.config.symbol, "FLAT", timestamp, float(entry_row["close"]), comment="Trend gücü zayıf")

        if trend_bias == "BULLISH" and trend_row["rsi"] < self.config.trend_rsi_bullish:
            return Signal(self.config.symbol, "FLAT", timestamp, float(entry_row["close"]), comment="Trend RSI teyidi yok")
        if trend_bias == "BEARISH" and trend_row["rsi"] > self.config.trend_rsi_bearish:
            return Signal(self.config.symbol, "FLAT", timestamp, float(entry_row["close"]), comment="Trend RSI teyidi yok")

        price = float(entry_row["close"])
        vwap = float(entry_row["vwap"])
        vwap_bias = (price - vwap) / atr_value
        if abs(vwap_bias) > self.config.max_vwap_distance_atr:
            return Signal(self.config.symbol, "FLAT", timestamp, price, comment="VWAP uzaklığı aşırı")

        intrabar_momentum = (entry_row["close"] - entry_row["open"]) / atr_value

        recent_slice = entry_df.iloc[-(self.config.breakout_lookback + 1):-1]
        recent_high = float(recent_slice["high"].max()) if not recent_slice.empty else price
        recent_low = float(recent_slice["low"].min()) if not recent_slice.empty else price

        confidence = min(1.0, max(0.0, abs(trend_strength)))
        comment = ""

        direction = "FLAT"
        stop_loss: Optional[float] = None
        take_profit: Optional[float] = None

        if (
            trend_bias == "BULLISH"
            and entry_row["ema_fast"] > entry_row["ema_slow"]
            and entry_row["rsi"] >= self.config.rsi_buy_threshold
            and intrabar_momentum >= self.config.min_momentum_ratio
            and price >= recent_high - 0.25 * atr_value
            and vwap_bias > -0.1
        ):
            direction = "BUY"
            stop_loss = price - self.config.atr_multiplier_stop * atr_value
            take_profit = price + self.config.atr_multiplier_take_profit * atr_value
            comment = "Trend + momentum uyumu"

        elif (
            trend_bias == "BEARISH"
            and entry_row["ema_fast"] < entry_row["ema_slow"]
            and entry_row["rsi"] <= self.config.rsi_sell_threshold
            and -intrabar_momentum >= self.config.min_momentum_ratio
            and price <= recent_low + 0.25 * atr_value
            and vwap_bias < 0.1
        ):
            direction = "SELL"
            stop_loss = price + self.config.atr_multiplier_stop * atr_value
            take_profit = price - self.config.atr_multiplier_take_profit * atr_value
            comment = "Trend + momentum uyumu"

        signal = Signal(
            symbol=self.config.symbol,
            direction=direction,
            timestamp=timestamp,
            price=price,
            stop_loss=float(stop_loss) if stop_loss is not None else None,
            take_profit=float(take_profit) if take_profit is not None else None,
            confidence=confidence,
            comment=comment,
        )

        if signal.direction != "FLAT":
            self._last_signal_time = timestamp
            self._last_direction = signal.direction

        return signal


def _respect_position_limit(symbol: str, max_open_positions: int) -> bool:
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return True
    return len(positions) < max_open_positions


def example_usage(iteration: int, strategy: ScalpingStrategy) -> None:
    _LOGGER.setLevel(logging.INFO)

    try:
        signal = strategy.generate_signal()
    except Exception as exc:  # broad catch for demo amaçlı
        print(f"[{iteration}] Hata: {exc}")
        return

    if signal.direction == "FLAT":
        print(f"[{iteration}] Sinyal yok → {signal.comment}")
        return

    print(
        f"[{iteration}] {signal.direction} sinyali! Fiyat: {signal.price:.2f} | SL: {signal.stop_loss:.2f} | TP: {signal.take_profit:.2f}"
    )

    if not _respect_position_limit(signal.symbol, strategy.config.max_open_positions):
        print("Açık pozisyon limiti dolu → Yeni işlem açılmıyor.")
        return

    tick = mt5.symbol_info_tick(signal.symbol)
    if not tick:
        print("Tick alınamadı.")
        return

    lot = strategy.config.lot_buy if signal.direction == "BUY" else strategy.config.lot_sell
    price = tick.ask if signal.direction == "BUY" else tick.bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": signal.symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY if signal.direction == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": signal.stop_loss,
        "tp": signal.take_profit,
        "deviation": strategy.config.deviation_points,
        "magic": 12345,
        "comment": f"auto {signal.direction.lower()} ({signal.confidence:.2f})",
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    print(f"{signal.direction} result:", result)

    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        ticket = result.order
        print(f"{signal.direction} açıldı! Ticket: {ticket}")
    else:
        print("Emir gönderilemedi veya reddedildi.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    sayac = 0
    strategy = ScalpingStrategy(StrategyConfig())

    try:
        initialize_mt5()
        print("------------------")
        while True:
            sayac += 1
            example_usage(sayac, strategy)
            time.sleep(1)
    except KeyboardInterrupt:
        print("Döngü kullanıcı tarafından durduruldu.")
    except Exception as exc:
        print("Hata:", exc)
    finally:
        shutdown_mt5()

