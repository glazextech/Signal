"""XAUUSD MetaTrader 5 scalping aracı (gerçek zamanlı).

Kurulum:
    pip install MetaTrader5 pandas numpy
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple

import MetaTrader5 as mt5
import numpy as np
import pandas as pd


@dataclass
class StrategyConfig:
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
    poll_interval: float = 1.0


@dataclass
class MT5LoginCredentials:
    account: Optional[int] = None
    password: Optional[str] = None
    server: Optional[str] = None
    path: Optional[str] = None


@dataclass
class Signal:
    symbol: str
    direction: str
    timestamp: datetime
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class TradeState:
    symbol: str
    direction: str
    entry_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    opened_at: datetime
    signal_timestamp: datetime
    exit_price: Optional[float] = None
    closed_at: Optional[datetime] = None
    exit_reason: Optional[str] = None

    @classmethod
    def from_signal(cls, signal: Signal, now: datetime) -> "TradeState":
        return cls(
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=signal.price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            opened_at=now,
            signal_timestamp=signal.timestamp,
        )

    def maybe_close(self, tick_price: float, tick: mt5.Tick, now: datetime, signal: Signal) -> None:
        exit_reason, exit_price = self._evaluate_exit(tick_price, tick, signal)
        if exit_reason is None or exit_price is None:
            return
        self.exit_price = exit_price
        self.closed_at = now
        self.exit_reason = exit_reason

    def _evaluate_exit(
        self, tick_price: float, tick: mt5.Tick, signal: Signal
    ) -> Tuple[Optional[str], Optional[float]]:
        bid = getattr(tick, "bid", None)
        ask = getattr(tick, "ask", None)
        if self.direction == "BUY":
            exit_price = _safe_price(bid, tick_price)
            if self.stop_loss is not None and exit_price <= self.stop_loss:
                return "STOP_LOSS", exit_price
            if self.take_profit is not None and exit_price >= self.take_profit:
                return "TAKE_PROFIT", exit_price
            if signal.direction == "SELL":
                return "OPPOSITE_SIGNAL", exit_price
        else:
            exit_price = _safe_price(ask, tick_price)
            if self.stop_loss is not None and exit_price >= self.stop_loss:
                return "STOP_LOSS", exit_price
            if self.take_profit is not None and exit_price <= self.take_profit:
                return "TAKE_PROFIT", exit_price
            if signal.direction == "BUY":
                return "OPPOSITE_SIGNAL", exit_price
        return None, None

    def is_closed(self) -> bool:
        return self.closed_at is not None

    def pnl(self) -> Optional[float]:
        if self.exit_price is None:
            return None
        if self.direction == "BUY":
            return self.exit_price - self.entry_price
        return self.entry_price - self.exit_price

    def summary(self) -> str:
        if not self.is_closed():
            return (
                f"{self.direction} {self.symbol} | entry {self.entry_price:.3f} | "
                f"SL {self.stop_loss} | TP {self.take_profit} | opened {self.opened_at.isoformat()}"
            )
        pnl = self.pnl()
        pnl_str = f"{pnl:+.3f}" if pnl is not None else "NA"
        duration = (self.closed_at - self.opened_at) if self.closed_at else None
        duration_str = str(duration) if duration else "NA"
        return (
            f"{self.direction} {self.symbol} | entry {self.entry_price:.3f} -> exit {self.exit_price:.3f} | "
            f"PnL {pnl_str} | reason {self.exit_reason} | duration {duration_str}"
        )


def _safe_price(value: Optional[float], fallback: float) -> float:
    if value is not None and value > 0:
        return float(value)
    return float(fallback)


def initialize_mt5(credentials: Optional[MT5LoginCredentials] = None) -> None:
    init_kwargs = {}
    if credentials and credentials.path:
        init_kwargs["path"] = credentials.path
    if not mt5.initialize(**init_kwargs):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    if credentials and credentials.account is not None:
        authorized = mt5.login(
            login=credentials.account,
            password=credentials.password,
            server=credentials.server,
        )
        if not authorized:
            raise RuntimeError(f"Login failed: {mt5.last_error()}")


def shutdown_mt5() -> None:
    mt5.shutdown()


def ensure_symbol(symbol: str) -> None:
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"Sembol bulunamadı: {symbol}")
    if not info.visible and not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"Sembol seçilemedi: {symbol}")


def _rates_to_dataframe(rates: np.ndarray) -> pd.DataFrame:
    if rates is None or len(rates) == 0:
        raise RuntimeError("MetaTrader 5 veri döndürmedi.")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    return df[["open", "high", "low", "close", "tick_volume"]]


def fetch_rates(symbol: str, timeframe: int, count: int) -> pd.DataFrame:
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
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
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
        df["ema_fast"] = compute_ema(df["close"], max(1, int(self.config.ema_fast_period / 2)))
        df["ema_slow"] = compute_ema(df["close"], self.config.ema_fast_period)
        df["rsi"] = compute_rsi(df["close"], max(7, int(self.config.rsi_period / 2)))
        df["atr"] = compute_atr(df, max(10, int(self.config.atr_period / 2)))
        df["vwap"] = compute_vwap(df)
        return df.dropna()

    def generate_signal(self) -> Signal:
        trend_df = self._prepare_trend_dataframe()
        entry_df = self._prepare_entry_dataframe()
        if trend_df.empty or entry_df.empty:
            raise RuntimeError("Yeterli veri alınamadı.")
        trend_row = trend_df.iloc[-1]
        entry_row = entry_df.iloc[-1]
        trend_bias = "BULLISH" if trend_row["ema_fast"] > trend_row["ema_slow"] else "BEARISH"
        atr_value = entry_row["atr"]
        if np.isnan(atr_value) or atr_value == 0:
            raise RuntimeError("ATR hesaplanamadı.")
        price = float(entry_row["close"])
        direction = "FLAT"
        stop_loss = None
        take_profit = None
        if entry_row["tick_volume"] < self.config.min_tick_volume:
            return Signal(self.config.symbol, direction, entry_row.name.to_pydatetime(), price)
        if (
            trend_bias == "BULLISH"
            and entry_row["ema_fast"] > entry_row["ema_slow"]
            and entry_row["rsi"] > 55
            and price > float(entry_row["vwap"])
        ):
            direction = "BUY"
            stop_loss = price - self.config.atr_multiplier * atr_value
            take_profit = price + self.config.atr_multiplier * 1.5 * atr_value
        elif (
            trend_bias == "BEARISH"
            and entry_row["ema_fast"] < entry_row["ema_slow"]
            and entry_row["rsi"] < 45
            and price < float(entry_row["vwap"])
        ):
            direction = "SELL"
            stop_loss = price + self.config.atr_multiplier * atr_value
            take_profit = price - self.config.atr_multiplier * 1.5 * atr_value
        timestamp = entry_row.name.to_pydatetime().astimezone(timezone.utc)
        return Signal(
            symbol=self.config.symbol,
            direction=direction,
            timestamp=timestamp,
            price=price,
            stop_loss=float(stop_loss) if stop_loss is not None else None,
            take_profit=float(take_profit) if take_profit is not None else None,
        )


def get_latest_price(symbol: str) -> Tuple[float, mt5.Tick]:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"Tick alınamadı: {symbol}")
    price_candidates = [getattr(tick, "last", None), getattr(tick, "bid", None), getattr(tick, "ask", None)]
    price = next((p for p in price_candidates if p is not None and p > 0), None)
    if price is None:
        raise RuntimeError("Geçerli fiyat bulunamadı.")
    return float(price), tick


def load_credentials_from_env() -> Optional[MT5LoginCredentials]:
    account = os.getenv("MT5_ACCOUNT")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    path = os.getenv("MT5_PATH")
    if not any([account, password, server, path]):
        return None
    account_int = int(account) if account else None
    return MT5LoginCredentials(account=account_int, password=password, server=server, path=path)


def run_live_loop(strategy: ScalpingStrategy) -> None:
    ensure_symbol(strategy.config.symbol)
    open_trade: Optional[TradeState] = None
    last_signal_signature: Optional[Tuple[str, datetime]] = None
    while True:
        loop_start = datetime.now(timezone.utc)
        try:
            price, tick = get_latest_price(strategy.config.symbol)
            bid = getattr(tick, "bid", float("nan"))
            ask = getattr(tick, "ask", float("nan"))
            spread = ask - bid if all(np.isfinite([bid, ask])) else float("nan")
            logging.info(
                "Fiyat %.3f | bid %.3f | ask %.3f | spread %.3f",
                price,
                bid,
                ask,
                spread,
            )

            signal = strategy.generate_signal()
            logging.info(
                "Sinyal %s | fiyat %.3f | SL %s | TP %s | bar %s",
                signal.direction,
                signal.price,
                f"{signal.stop_loss:.3f}" if signal.stop_loss is not None else "NA",
                f"{signal.take_profit:.3f}" if signal.take_profit is not None else "NA",
                signal.timestamp.isoformat(),
            )

            if open_trade is not None and not open_trade.is_closed():
                open_trade.maybe_close(price, tick, loop_start, signal)
                if open_trade.is_closed():
                    logging.info("İşlem kapandı: %s", open_trade.summary())
                    open_trade = None

            signal_key = (signal.direction, signal.timestamp)
            if (
                open_trade is None
                and signal.direction in {"BUY", "SELL"}
                and signal_key != last_signal_signature
            ):
                open_trade = TradeState.from_signal(signal, loop_start)
                logging.info("Yeni işlem açıldı: %s", open_trade.summary())

            last_signal_signature = signal_key

        except KeyboardInterrupt:
            raise
        except Exception:
            logging.exception("Döngü hatası")
        time.sleep(strategy.config.poll_interval)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    credentials = load_credentials_from_env()
    initialize_mt5(credentials)
    strategy = ScalpingStrategy(StrategyConfig())
    try:
        run_live_loop(strategy)
    except KeyboardInterrupt:
        logging.info("Kullanıcı tarafından durduruldu.")
    finally:
        if mt5.initialize():
            shutdown_mt5()
        else:
            shutdown_mt5()


if __name__ == "__main__":
    main()
