from dataclasses import dataclass
from datetime import datetime
import time
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

np.seterr(divide="ignore", invalid="ignore")


@dataclass
class StrategyConfig:
    symbol: str
    timeframe_m5: int
    timeframe_m1: int
    lookback_m5: int
    lookback_m1: int
    ema_fast: int
    ema_slow: int
    rsi_period: int
    atr_period: int
    min_tick_volume: int
    atr_sl_multiplier: float
    atr_tp_multiplier: float
    rsi_momentum_threshold: float


@dataclass
class Signal:
    timestamp: pd.Timestamp
    direction: str
    price: float
    stop_loss: float
    take_profit: float
    atr: float
    rsi: float
    vwap: float


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    values = 100 - (100 / (1 + rs))
    return values.fillna(50)


def atr(high, low, close, period):
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def vwap(high, low, close, volume):
    typical = (high + low + close) / 3
    cumulative_vp = (typical * volume).cumsum()
    cumulative_volume = volume.cumsum()
    return cumulative_vp / cumulative_volume.replace(0, np.nan)


class ScalpingStrategy:
    def __init__(self, config):
        self.config = config
        self.last_signal_time = None
        self.connected = self._connect()

    def _connect(self):
        if mt5.initialize():
            print(f"{datetime.now().strftime('%H:%M:%S')} | BİLGİ | MT5 bağlantısı başarılı")
            return True
        print(f"{datetime.now().strftime('%H:%M:%S')} | HATA | MT5 bağlantısı başarısız: {mt5.last_error()}")
        return False

    def _get_rates(self, timeframe, count):
        rates = mt5.copy_rates_from_pos(self.config.symbol, timeframe, 0, count)
        if rates is None:
            return pd.DataFrame()
        df = pd.DataFrame(rates)
        if df.empty:
            return df
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    def generate_signal(self):
        try:
            if not self.connected:
                self.connected = self._connect()
                if not self.connected:
                    return Signal(pd.Timestamp.utcnow(), "NONE", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
            df_m5 = self._get_rates(self.config.timeframe_m5, self.config.lookback_m5)
            df_m1 = self._get_rates(self.config.timeframe_m1, self.config.lookback_m1)
            if df_m5.empty or df_m1.empty:
                print(f"{datetime.now().strftime('%H:%M:%S')} | UYARI | Veri alınamadı")
                return Signal(pd.Timestamp.utcnow(), "NONE", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
            df_m5["ema_fast"] = ema(df_m5["close"], self.config.ema_fast)
            df_m5["ema_slow"] = ema(df_m5["close"], self.config.ema_slow)
            trend_direction = "BUY" if df_m5["ema_fast"].iloc[-1] > df_m5["ema_slow"].iloc[-1] else "SELL"
            df_m1["rsi"] = rsi(df_m1["close"], self.config.rsi_period)
            df_m1["atr"] = atr(df_m1["high"], df_m1["low"], df_m1["close"], self.config.atr_period)
            df_m1["vwap"] = vwap(df_m1["high"], df_m1["low"], df_m1["close"], df_m1["tick_volume"])
            recent = df_m1.dropna()
            if len(recent) < 3:
                return Signal(pd.Timestamp.utcnow(), "NONE", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
            current = recent.iloc[-1]
            previous = recent.iloc[-2]
            if current["tick_volume"] < self.config.min_tick_volume:
                return Signal(pd.Timestamp.utcnow(), "NONE", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
            if self.last_signal_time is not None and current["time"] == self.last_signal_time:
                return Signal(pd.Timestamp.utcnow(), "NONE", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
            rsi_value = float(current["rsi"])
            rsi_delta = rsi_value - float(previous["rsi"])
            atr_value = float(current["atr"])
            price = float(current["close"])
            vwap_value = float(current["vwap"])
            direction = "NONE"
            if trend_direction == "BUY" and rsi_value > 50 and rsi_delta > self.config.rsi_momentum_threshold and price >= vwap_value:
                direction = "BUY"
            elif trend_direction == "SELL" and rsi_value < 50 and rsi_delta < -self.config.rsi_momentum_threshold and price <= vwap_value:
                direction = "SELL"
            if direction == "NONE" or np.isnan(atr_value) or atr_value <= 0:
                return Signal(pd.Timestamp.utcnow(), "NONE", np.nan, np.nan, np.nan, atr_value, rsi_value, vwap_value)
            if direction == "BUY":
                stop_loss = price - atr_value * self.config.atr_sl_multiplier
                take_profit = price + atr_value * self.config.atr_tp_multiplier
            else:
                stop_loss = price + atr_value * self.config.atr_sl_multiplier
                take_profit = price - atr_value * self.config.atr_tp_multiplier
            self.last_signal_time = current["time"]
            timestamp = current["time"]
            return Signal(timestamp, direction, price, stop_loss, take_profit, atr_value, rsi_value, vwap_value)
        except Exception as exc:
            print(f"{datetime.now().strftime('%H:%M:%S')} | HATA | Sinyal üretimi başarısız: {exc}")
            return Signal(pd.Timestamp.utcnow(), "NONE", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)


def example_usage():
    config = StrategyConfig(
        symbol="XAUUSD",
        timeframe_m5=mt5.TIMEFRAME_M5,
        timeframe_m1=mt5.TIMEFRAME_M1,
        lookback_m5=200,
        lookback_m1=500,
        ema_fast=12,
        ema_slow=26,
        rsi_period=9,
        atr_period=10,
        min_tick_volume=20,
        atr_sl_multiplier=0.25,
        atr_tp_multiplier=2.0,
        rsi_momentum_threshold=0.15,
    )
    strategy = ScalpingStrategy(config)
    while True:
        try:
            signal = strategy.generate_signal()
            if signal.direction != "NONE":
                print(f"{signal.timestamp.strftime('%H:%M:%S')} | {signal.direction:<4} | Fiyat: {signal.price:.2f} | SL: {signal.stop_loss:.2f} | TP: {signal.take_profit:.2f}")
        except Exception as exc:
            print(f"{datetime.now().strftime('%H:%M:%S')} | HATA | Döngü hatası: {exc}")
        time.sleep(1)


if __name__ == "__main__":
    example_usage()

