"""XAUUSD scalping stratejisi için MetaTrader 5 ile veri çekme ve sinyal üretme scripti.

Bu script, MetaTrader 5 terminalinizi kullanarak XAUUSD (Altın) paritesi için
anlık veri toplar, çeşitli teknik indikatörler hesaplar ve kısa vadeli (scalping)
alım-satım sinyalleri üretir. İsteğe bağlı olarak emir açma/kapama örnekleri de
sunulmuştur ancak gerçek hesap bilgilerinizle test etmeden çalıştırmayın.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import MetaTrader5 as mt5
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Konfigürasyon
# ---------------------------------------------------------------------------


@dataclass
class AccountCredentials:
    """MetaTrader 5 hesabı için giriş bilgileri."""

    login: int
    password: str
    server: str


@dataclass
class StrategyConfig:
    """Scalping stratejisi için parametreler."""

    symbol: str = "XAUUSD"
    timeframe: int = mt5.TIMEFRAME_M1  # 1 dakikalık mumlar
    history_bars: int = 1500  # indikatör hesapları için gerekli minimum veri
    refresh_seconds: float = 30.0  # loop tekrar süresi

    ema_fast: int = 8
    ema_slow: int = 21
    ema_trend: int = 89
    rsi_period: int = 14
    rsi_overbought: int = 65
    rsi_oversold: int = 35
    atr_period: int = 14
    atr_multiplier: float = 1.6

    max_spread_points: float = 30.0  # scalping için maksimum kabul edilebilir spread
    risk_per_trade: float = 0.01  # hesap bakiyesi bazında risk yüzdesi
    min_rr_ratio: float = 1.5  # risk/ödül oranı

    volume_step: float = 0.01  # lot adımı
    min_volume: float = 0.01
    max_volume: float = 5.0


# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------


def initialize_logging(level: int = logging.INFO) -> None:
    """Basit logging yapılandırması."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)5s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def initialize_terminal(credentials: Optional[AccountCredentials] = None) -> None:
    """MetaTrader 5 terminalini başlatır ve giriş yapar."""

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize hatası: {mt5.last_error()}")

    if credentials:
        authorized = mt5.login(credentials.login, password=credentials.password, server=credentials.server)
        if not authorized:
            error_code, error_detail = mt5.last_error()
            raise RuntimeError(f"Giriş başarısız: {error_code} | {error_detail}")


def shutdown_terminal() -> None:
    """MetaTrader 5 terminal bağlantısını kapatır."""

    mt5.shutdown()


def fetch_rates(config: StrategyConfig) -> pd.DataFrame:
    """Belirtilen sembol ve zaman dilimi için mum verilerini DataFrame olarak döndürür."""

    rates = mt5.copy_rates_from_pos(config.symbol, config.timeframe, 0, config.history_bars)
    if rates is None or len(rates) == 0:
        error_code, error_detail = mt5.last_error()
        raise RuntimeError(f"Veri alınamadı: {error_code} | {error_detail}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    return df


def compute_indicators(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """EMA, RSI, ATR ve VWAP gibi indikatörleri hesaplar."""

    df = df.copy()

    df[f"ema_{config.ema_fast}"] = df["close"].ewm(span=config.ema_fast, adjust=False).mean()
    df[f"ema_{config.ema_slow}"] = df["close"].ewm(span=config.ema_slow, adjust=False).mean()
    df[f"ema_{config.ema_trend}"] = df["close"].ewm(span=config.ema_trend, adjust=False).mean()

    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).rolling(window=config.rsi_period).mean()
    avg_loss = pd.Series(loss).rolling(window=config.rsi_period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    df["rsi"] = 100 - (100 / (1 + rs))

    tr_components = [
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ]
    true_range = pd.concat(tr_components, axis=1).max(axis=1)
    df["atr"] = true_range.rolling(window=config.atr_period).mean()

    cumulative_volume = df["tick_volume"].cumsum().replace(0, np.nan)
    df["vwap"] = (df["close"] * df["tick_volume"]).cumsum() / cumulative_volume

    df.dropna(inplace=True)
    return df


@dataclass
class TradeSignal:
    direction: str  # "buy" veya "sell"
    entry: float
    stop_loss: float
    take_profit: float
    volume: float
    comment: str


def pip_value(symbol: str) -> float:
    """Sembol pip değerini döner; XAUUSD için tahmini değer."""

    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"{symbol} sembol bilgisi alınamadı")
    return info.point


def calculate_volume(balance: float, stop_distance: float, config: StrategyConfig) -> float:
    """Risk yüzdesine göre lot büyüklüğü hesaplar."""

    info = mt5.symbol_info(config.symbol)
    if info is None or stop_distance <= 0:
        return config.min_volume

    tick_size = info.trade_tick_size or info.point
    tick_value = info.trade_tick_value or info.tick_value

    if tick_size <= 0 or tick_value <= 0:
        logging.warning("Geçersiz tick bilgisi; minimum volume kullanılacak")
        return config.min_volume

    stop_ticks = stop_distance / tick_size
    loss_per_lot = stop_ticks * tick_value
    if loss_per_lot <= 0:
        return config.min_volume

    risk_amount = balance * config.risk_per_trade
    lot = risk_amount / loss_per_lot
    normalized = max(
        config.min_volume,
        min(config.max_volume, math.floor(lot / config.volume_step) * config.volume_step),
    )
    return round(normalized, 2)


def build_signal(df: pd.DataFrame, config: StrategyConfig) -> Optional[TradeSignal]:
    """Son mum verisi ile strateji kriterlerine göre sinyal üretir."""

    last = df.iloc[-1]
    prev = df.iloc[-2]

    symbol_info = mt5.symbol_info_tick(config.symbol)
    if symbol_info is None:
        logging.warning("Sembol tick bilgisi alınamadı")
        return None

    spread_points = (symbol_info.ask - symbol_info.bid) / pip_value(config.symbol)
    if spread_points > config.max_spread_points:
        logging.info("Spread yüksek: %.2f", spread_points)
        return None

    conditions_long = [
        last[f"ema_{config.ema_fast}"] > last[f"ema_{config.ema_slow}"] > last[f"ema_{config.ema_trend}"],
        prev[f"ema_{config.ema_fast}"] <= prev[f"ema_{config.ema_slow}"],
        last["rsi"] > config.rsi_oversold,
        last["close"] > last["vwap"],
    ]

    conditions_short = [
        last[f"ema_{config.ema_fast}"] < last[f"ema_{config.ema_slow}"] < last[f"ema_{config.ema_trend}"],
        prev[f"ema_{config.ema_fast}"] >= prev[f"ema_{config.ema_slow}"],
        last["rsi"] < 100 - config.rsi_overbought,
        last["close"] < last["vwap"],
    ]

    balance = mt5.account_info().balance if mt5.account_info() else 0

    if all(conditions_long):
        entry = symbol_info.ask
        stop = entry - config.atr_multiplier * last["atr"]
        target = entry + config.min_rr_ratio * (entry - stop)
        volume = calculate_volume(balance, entry - stop, config)
        return TradeSignal("buy", entry, stop, target, volume, "EMA-RSI Long")

    if all(conditions_short):
        entry = symbol_info.bid
        stop = entry + config.atr_multiplier * last["atr"]
        target = entry - config.min_rr_ratio * (stop - entry)
        volume = calculate_volume(balance, stop - entry, config)
        return TradeSignal("sell", entry, stop, target, volume, "EMA-RSI Short")

    return None


def place_order(signal: TradeSignal, config: StrategyConfig) -> Optional[int]:
    """Örnek bir market emir açılışı. Gerçek hesapta test etmeden kullanmayın."""

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": config.symbol,
        "volume": signal.volume,
        "type": mt5.ORDER_TYPE_BUY if signal.direction == "buy" else mt5.ORDER_TYPE_SELL,
        "price": signal.entry,
        "sl": signal.stop_loss,
        "tp": signal.take_profit,
        "deviation": 20,
        "magic": 445566,
        "comment": signal.comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    result = mt5.order_send(request)
    if result is None:
        logging.error("Emir gönderilemedi: %s", mt5.last_error())
        return None

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error("Emir retcode %s: %s", result.retcode, result)
        return None

    logging.info("Emir açıldı: ticket=%s", result.order)
    return result.order


def load_credentials_from_env() -> Optional[AccountCredentials]:
    """Ortam değişkenlerinden hesap bilgilerini alır."""

    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")

    if not all([login, password, server]):
        return None

    try:
        return AccountCredentials(login=int(login), password=password, server=server)
    except ValueError as exc:
        raise ValueError("MT5_LOGIN sayısal bir değer olmalı") from exc


def run_loop(config: StrategyConfig) -> None:
    """Sürekli çalışan strateji döngüsü."""

    logging.info("Scalping stratejisi başlatıldı | Symbol=%s", config.symbol)

    while True:
        try:
            df = fetch_rates(config)
            df = compute_indicators(df, config)
            signal = build_signal(df, config)

            if signal:
                logging.info(
                    "Sinyal: %s | entry=%.3f sl=%.3f tp=%.3f vol=%.2f | %s",
                    signal.direction.upper(),
                    signal.entry,
                    signal.stop_loss,
                    signal.take_profit,
                    signal.volume,
                    signal.comment,
                )

                # İsteğe bağlı emir açılışı
                # place_order(signal, config)
            else:
                logging.debug("Sinyal yok")

        except Exception as exc:  # pylint: disable=broad-exception-caught
            logging.exception("Strateji döngüsünde hata: %s", exc)

        time.sleep(config.refresh_seconds)


def main() -> None:
    initialize_logging()
    config = StrategyConfig()
    credentials = load_credentials_from_env()

    try:
        initialize_terminal(credentials)
        run_loop(config)
    except KeyboardInterrupt:
        logging.info("Kullanıcı tarafından durduruldu")
    finally:
        shutdown_terminal()


if __name__ == "__main__":
    main()

