"""MetaTrader 5 XAUUSD scalping helper with confidence filtering.

Bu modül, MetaTrader 5 terminalinden veri çekerek XAUUSD (Altın) için
çoklu zaman ölçekli bir scalping stratejisi uygular. Strateji, trend ve
momentum filtrelerini birleştirerek yalnızca güvenilir (high-conviction)
sinyallere odaklanırken, piyasadan tamamen kopmadan fırsat kovalar.

Kurulum:
    pip install MetaTrader5 pandas numpy

Not:
    - MT5 terminali açık ve hesabınızda giriş yapılmış olmalıdır.
    - Kod, MT5 API aracılığıyla emir gönderebilir. Önce demo hesapta test edin.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple

import MetaTrader5 as mt5
import numpy as np
import pandas as pd


_LOGGER = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Strateji ayarları."""

    symbol: str = "GOLD#"
    trend_timeframe: int = mt5.TIMEFRAME_M5
    entry_timeframe: int = mt5.TIMEFRAME_M1
    trend_lookback: int = 600
    entry_lookback: int = 400
    ema_fast_period: int = 21
    ema_slow_period: int = 55
    rsi_period: int = 14
    atr_period: int = 14
    stop_loss_pips: float = 10.0
    take_profit_pips: float = 1.25
    min_tick_volume: int = 60
    volume_lookback: int = 120
    volume_quantile: float = 0.65
    ema_slope_period: int = 3
    ema_slope_threshold: float = 0.05
    ema_alignment_threshold: float = 0.08
    rsi_entry_buffer: float = 5.0
    trend_rsi_buffer: float = 3.0
    atr_pct_min: float = 0.0006
    atr_pct_max: float = 0.0035
    vwap_distance_max_atr: float = 1.2
    min_confidence: float = 0.65
    cooldown_bars: int = 3
    max_spread_points: float = 45.0
    candle_body_ratio_min: float = 0.28
    wick_ratio_max: float = 0.55
    swing_window: int = 30
    swing_bias_buy: float = 0.55
    swing_bias_sell: float = 0.45


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
    reasons: Tuple[str, ...] = ()


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
    rs = avg_gain / avg_loss.replace(0, np.nan)
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

    def _prepare_trend_dataframe(self) -> pd.DataFrame:
        df = fetch_rates(self.config.symbol, self.config.trend_timeframe, self.config.trend_lookback)
        df["ema_fast"] = compute_ema(df["close"], self.config.ema_fast_period)
        df["ema_slow"] = compute_ema(df["close"], self.config.ema_slow_period)
        df["rsi"] = compute_rsi(df["close"], self.config.rsi_period)
        df["atr"] = compute_atr(df, self.config.atr_period)
        df = df.dropna()
        if df.empty:
            raise RuntimeError("Trend verisi hesaplanamadı.")
        return df

    def _prepare_entry_dataframe(self) -> pd.DataFrame:
        df = fetch_rates(self.config.symbol, self.config.entry_timeframe, self.config.entry_lookback)
        df["ema_fast"] = compute_ema(df["close"], int(self.config.ema_fast_period / 2))
        df["ema_slow"] = compute_ema(df["close"], self.config.ema_fast_period)
        df["rsi"] = compute_rsi(df["close"], max(7, int(self.config.rsi_period / 2)))
        df["atr"] = compute_atr(df, max(10, int(self.config.atr_period / 2)))
        df["vwap"] = compute_vwap(df)

        slope_period = max(1, self.config.ema_slope_period)
        df["ema_fast_slope"] = df["ema_fast"].diff(slope_period)
        df["ema_slow_slope"] = df["ema_slow"].diff(slope_period)
        df["ema_diff"] = df["ema_fast"] - df["ema_slow"]
        df["ema_diff_delta"] = df["ema_diff"].diff()
        df["rsi_delta"] = df["rsi"].diff()
        df["atr_pct"] = df["atr"] / df["close"].replace(0, np.nan)

        df["candle_body"] = (df["close"] - df["open"]).abs()
        df["candle_range"] = df["high"] - df["low"]
        df["candle_range"] = df["candle_range"].replace(0, np.nan)
        df["candle_body_ratio"] = df["candle_body"] / df["candle_range"]

        upper_anchor = df[["open", "close"]].max(axis=1)
        lower_anchor = df[["open", "close"]].min(axis=1)
        df["upper_wick"] = (df["high"] - upper_anchor).clip(lower=0)
        df["lower_wick"] = (lower_anchor - df["low"]).clip(lower=0)
        df["upper_wick_ratio"] = df["upper_wick"] / df["candle_range"]
        df["lower_wick_ratio"] = df["lower_wick"] / df["candle_range"]

        df = df.dropna()
        if df.empty:
            raise RuntimeError("Giriş verisi hesaplanamadı.")
        return df

    def _evaluate_direction(
        self,
        direction: str,
        entry_row: pd.Series,
        previous_row: pd.Series,
        trend_row: pd.Series,
        tick,
        volume_floor: float,
        pip_value: float,
        swing_high: float,
        swing_low: float,
    ) -> Tuple[Optional[dict], Tuple[str, ...]]:
        direction_sign = 1 if direction == "BUY" else -1

        atr_value = float(entry_row["atr"])
        if np.isnan(atr_value) or atr_value <= 0:
            return None, (f"{direction}: ATR geçersiz",)

        price = tick.ask if direction == "BUY" else tick.bid
        if price is None or price <= 0:
            return None, (f"{direction}: Tick fiyatı geçersiz",)

        trend_gap = (trend_row["ema_fast"] - trend_row["ema_slow"]) * direction_sign
        entry_gap = (entry_row["ema_fast"] - entry_row["ema_slow"]) * direction_sign

        gating_reasons = []
        if trend_gap <= 0:
            gating_reasons.append(
                f"Trend uyumsuz: EMA farkı {trend_gap:+.3f}"
            )
        if entry_gap <= 0:
            gating_reasons.append(
                f"Giriş EMA uyumsuz: fark {entry_gap:+.3f}"
            )
        if gating_reasons:
            return None, tuple(gating_reasons)

        conditions = []

        def add_condition(name: str, passed: bool, weight: float, detail: str) -> None:
            conditions.append((name, bool(passed), float(weight), detail))

        add_condition(
            "Trend bias",
            trend_gap > self.config.ema_alignment_threshold,
            2.5,
            f"Trend EMA farkı {trend_gap:+.3f}",
        )

        add_condition(
            "Entry EMA alignment",
            entry_gap > self.config.ema_alignment_threshold / 2,
            1.5,
            f"Giriş EMA farkı {entry_gap:+.3f}",
        )

        fast_slope = float(entry_row["ema_fast_slope"]) if pd.notna(entry_row["ema_fast_slope"]) else 0.0
        slow_slope = float(entry_row["ema_slow_slope"]) if pd.notna(entry_row["ema_slow_slope"]) else 0.0
        ema_momentum = float(entry_row["ema_diff_delta"]) if pd.notna(entry_row["ema_diff_delta"]) else 0.0
        prev_entry_gap_val = float(previous_row["ema_diff"]) if pd.notna(previous_row["ema_diff"]) else 0.0

        fast_slope *= direction_sign
        slow_slope *= direction_sign
        ema_momentum *= direction_sign
        prev_entry_gap = prev_entry_gap_val * direction_sign

        add_condition(
            "EMA fast slope",
            fast_slope > self.config.ema_slope_threshold,
            1.0,
            f"Hızlı EMA eğimi {fast_slope:+.3f}",
        )

        add_condition(
            "EMA slow slope",
            slow_slope > 0,
            0.6,
            f"Yavaş EMA eğimi {slow_slope:+.3f}",
        )

        add_condition(
            "EMA momentum",
            ema_momentum > 0,
            0.7,
            f"EMA diff Δ {ema_momentum:+.4f}",
        )

        add_condition(
            "EMA persistence",
            prev_entry_gap > 0,
            0.5,
            f"Önceki EMA farkı {prev_entry_gap:+.3f}",
        )

        rsi_value = float(entry_row["rsi"])
        rsi_delta = (entry_row["rsi_delta"] or 0.0) * direction_sign
        add_condition(
            "RSI level",
            (rsi_value - 50) * direction_sign > self.config.rsi_entry_buffer,
            1.0,
            f"RSI {rsi_value:.1f}",
        )

        add_condition(
            "RSI momentum",
            rsi_delta > 0,
            0.8,
            f"RSI Δ {rsi_delta:+.2f}",
        )

        trend_rsi = float(trend_row["rsi"]) if pd.notna(trend_row["rsi"]) else 50.0
        add_condition(
            "Trend RSI",
            (trend_rsi - 50) * direction_sign > self.config.trend_rsi_buffer,
            1.2,
            f"Trend RSI {trend_rsi:.1f}",
        )

        tick_volume = float(entry_row["tick_volume"])
        add_condition(
            "Volume boost",
            tick_volume >= volume_floor,
            0.8,
            f"Hacim {tick_volume:.0f} / eşik {volume_floor:.0f}",
        )

        atr_pct = float(entry_row["atr_pct"])
        add_condition(
            "Volatility window",
            self.config.atr_pct_min <= atr_pct <= self.config.atr_pct_max,
            0.7,
            f"ATR% {atr_pct:.4f}",
        )

        body_ratio = float(entry_row["candle_body_ratio"]) if pd.notna(entry_row["candle_body_ratio"]) else 0.0
        candle_direction = (entry_row["close"] - entry_row["open"]) * direction_sign
        add_condition(
            "Candle body",
            candle_direction > 0 and body_ratio >= self.config.candle_body_ratio_min,
            0.9,
            f"Gövde oranı {body_ratio:.2f}",
        )

        upper_wick_ratio = float(entry_row["upper_wick_ratio"]) if pd.notna(entry_row["upper_wick_ratio"]) else 0.0
        lower_wick_ratio = float(entry_row["lower_wick_ratio"]) if pd.notna(entry_row["lower_wick_ratio"]) else 0.0
        wick_ratio = upper_wick_ratio if direction == "BUY" else lower_wick_ratio
        add_condition(
            "Wick control",
            wick_ratio <= self.config.wick_ratio_max,
            0.6,
            f"Fitil oranı {wick_ratio:.2f}",
        )

        swing_span = swing_high - swing_low
        if swing_span <= 0:
            swing_position = 0.5
        else:
            swing_position = (price - swing_low) / swing_span

        if direction == "BUY":
            swing_pass = swing_position >= self.config.swing_bias_buy
            swing_detail = f"Swing pozisyonu {swing_position:.2f} (>= {self.config.swing_bias_buy:.2f})"
        else:
            swing_pass = swing_position <= self.config.swing_bias_sell
            swing_detail = f"Swing pozisyonu {swing_position:.2f} (<= {self.config.swing_bias_sell:.2f})"

        add_condition(
            "Swing bias",
            swing_pass,
            0.9,
            swing_detail,
        )

        vwap = float(entry_row["vwap"])
        distance_atr = abs(price - vwap) / atr_value if atr_value else np.inf
        vwap_bias = (price - vwap) * direction_sign
        add_condition(
            "VWAP bias",
            vwap_bias >= 0 and distance_atr <= self.config.vwap_distance_max_atr,
            0.9,
            f"VWAP farkı {vwap_bias:+.3f}, ATRx {distance_atr:.2f}",
        )

        total_weight = sum(weight for _, _, weight, _ in conditions)
        satisfied_weight = sum(weight for _, passed, weight, _ in conditions if passed)
        confidence = satisfied_weight / total_weight if total_weight else 0.0

        reasons = tuple(
            f"{'✔' if passed else '✘'} {name}: {detail}"
            for name, passed, _, detail in conditions
        )

        sl_offset = self.config.stop_loss_pips * pip_value
        tp_offset = self.config.take_profit_pips * pip_value
        stop_loss = price - direction_sign * sl_offset
        take_profit = price + direction_sign * tp_offset

        return (
            {
                "direction": direction,
                "price": float(price),
                "stop_loss": float(stop_loss),
                "take_profit": float(take_profit),
                "confidence": float(confidence),
                "reasons": reasons,
            },
            tuple(),
        )

    def generate_signal(self) -> Signal:
        trend_df = self._prepare_trend_dataframe()
        entry_df = self._prepare_entry_dataframe()

        if trend_df.empty or entry_df.empty:
            raise RuntimeError("Gerekli veri hazırlanamadı.")

        trend_row = trend_df.iloc[-1]
        entry_row = entry_df.iloc[-1]
        previous_row = entry_df.iloc[-2]

        tick = mt5.symbol_info_tick(self.config.symbol)
        if not tick:
            raise RuntimeError("Anlık tick verisi alınamadı.")

        symbol_info = mt5.symbol_info(self.config.symbol)
        if not symbol_info or symbol_info.point == 0:
            raise RuntimeError("Sembol bilgisi eksik veya geçersiz.")

        spread_points = (tick.ask - tick.bid) / symbol_info.point
        if spread_points > self.config.max_spread_points:
            reasons = (
                f"Spread {spread_points:.1f}p > limit {self.config.max_spread_points:.1f}p",
            )
            timestamp = entry_row.name.to_pydatetime().astimezone(timezone.utc)
            return Signal(
                symbol=self.config.symbol,
                direction="FLAT",
                timestamp=timestamp,
                price=float(entry_row["close"]),
                confidence=0.0,
                reasons=reasons,
            )

        pip_value = symbol_info.point
        if pip_value <= 0:
            raise RuntimeError("Geçersiz point değeri alındı.")

        swing_window = max(5, self.config.swing_window)
        recent_slice = entry_df.tail(swing_window)
        swing_high = float(recent_slice["high"].max())
        swing_low = float(recent_slice["low"].min())

        volume_series = entry_df["tick_volume"].tail(max(self.config.volume_lookback, 10))
        dynamic_threshold = volume_series.quantile(self.config.volume_quantile)
        if np.isnan(dynamic_threshold):
            dynamic_threshold = self.config.min_tick_volume
        volume_floor = max(self.config.min_tick_volume, float(dynamic_threshold))

        evaluations = []
        fallback_reasons = []

        for direction in ("BUY", "SELL"):
            evaluation, gating = self._evaluate_direction(
                direction,
                entry_row,
                previous_row,
                trend_row,
                tick,
                volume_floor,
                pip_value,
                swing_high,
                swing_low,
            )
            if evaluation:
                evaluations.append(evaluation)
            else:
                fallback_reasons.extend(f"{direction}: {reason}" for reason in gating)

        timestamp = entry_row.name.to_pydatetime().astimezone(timezone.utc)

        if not evaluations:
            reasons = tuple(fallback_reasons) or ("Uygun yönlü trend filtresi bulunamadı.",)
            return Signal(
                symbol=self.config.symbol,
                direction="FLAT",
                timestamp=timestamp,
                price=float(entry_row["close"]),
                confidence=0.0,
                reasons=reasons,
            )

        best = max(evaluations, key=lambda item: item["confidence"])

        if best["confidence"] < self.config.min_confidence:
            reasons = best["reasons"] + (f"Güven eşiği {self.config.min_confidence:.0%} üzeri bekleniyor.",)
            return Signal(
                symbol=self.config.symbol,
                direction="FLAT",
                timestamp=timestamp,
                price=float(best["price"]),
                confidence=float(best["confidence"]),
                reasons=reasons,
            )

        if self._last_signal_time is not None:
            elapsed_minutes = (timestamp - self._last_signal_time).total_seconds() / 60.0
            if elapsed_minutes < self.config.cooldown_bars:
                reasons = best["reasons"] + (
                    f"⏳ Cooldown aktif: {elapsed_minutes:.1f} dk < {self.config.cooldown_bars} dk",
                )
                return Signal(
                    symbol=self.config.symbol,
                    direction="FLAT",
                    timestamp=timestamp,
                    price=float(best["price"]),
                    confidence=float(best["confidence"]),
                    reasons=reasons,
                )

        self._last_signal_time = timestamp

        return Signal(
            symbol=self.config.symbol,
            direction=best["direction"],
            timestamp=timestamp,
            price=float(best["price"]),
            stop_loss=float(best["stop_loss"]),
            take_profit=float(best["take_profit"]),
            confidence=float(best["confidence"]),
            reasons=best["reasons"],
        )


def example_usage(counter: int) -> None:
    logging.basicConfig(level=logging.INFO)

    try:
        initialize_mt5()  # Manuel giriş varsa None, None, None
        strategy = ScalpingStrategy(StrategyConfig())
        signal = strategy.generate_signal()

        def print_reasons(title: str, reasons: Tuple[str, ...]) -> None:
            if not reasons:
                return
            print(title)
            for reason in reasons:
                print(f"    - {reason}")

        if signal.direction == "BUY":
            counter += 1
            print(f"[{counter}] BUY Sinyali ({signal.confidence:.0%}) Fiyat: {signal.price:.2f}")
            print_reasons("Filtreler:", signal.reasons)

            positions = mt5.positions_get(symbol=signal.symbol)
            if positions and len(positions) >= 3:
                print("Açık pozisyon limiti dolu → Yeni işlem açılmıyor.")
                return

            tick = mt5.symbol_info_tick(signal.symbol)
            if not tick:
                print("Tick alınamadı.")
                return

            price = tick.ask
            lot = 0.25
            deviation = 20

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": signal.symbol,
                "volume": lot,
                "type": mt5.ORDER_TYPE_BUY,
                "price": price,
                "deviation": deviation,
                "magic": 12345,
                "comment": "auto BUY",
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            print("BUY result:", result)

            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                ticket = result.order
                print(f"BUY açıldı! Ticket: {ticket}")

                tp_price = signal.take_profit if signal.take_profit else price + 0.04
                sl_price = signal.stop_loss if signal.stop_loss else price - 0.04

                modify_request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": ticket,
                    "symbol": signal.symbol,
                    "tp": tp_price,
                    "sl": sl_price,
                }
                modify_result = mt5.order_send(modify_request)
                print("SL/TP eklendi:", modify_result)

        elif signal.direction == "SELL":
            counter += 1
            print(f"[{counter}] SELL Sinyali ({signal.confidence:.0%}) Fiyat: {signal.price:.2f}")
            print_reasons("Filtreler:", signal.reasons)

            positions = mt5.positions_get(symbol=signal.symbol)
            if positions and len(positions) >= 3:
                print("Açık pozisyon limiti dolu → İşlem açılmıyor.")
                return

            tick = mt5.symbol_info_tick(signal.symbol)
            if not tick:
                print("Tick alınamadı.")
                return

            price = tick.bid
            lot = 0.08
            deviation = 20

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": signal.symbol,
                "volume": lot,
                "type": mt5.ORDER_TYPE_SELL,
                "price": price,
                "deviation": deviation,
                "magic": 12345,
                "comment": "auto SELL",
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            print("SELL result:", result)

            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                ticket = result.order
                print(f"SELL açıldı! Ticket: {ticket}")

                tp_price = signal.take_profit if signal.take_profit else price - 0.04
                sl_price = signal.stop_loss if signal.stop_loss else price + 0.04

                modify_request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": ticket,
                    "symbol": signal.symbol,
                    "tp": tp_price,
                    "sl": sl_price,
                }
                modify_result = mt5.order_send(modify_request)
                print("SL/TP eklendi:", modify_result)

        else:
            print(f"[{counter}] Sinyal yok ({signal.confidence:.0%})")
            print_reasons("Gözlemler:", signal.reasons)

    except Exception as exc:  # noqa: BLE001 - MT5 hatalarını yüzeye taşı
        print("Hata:", exc)
    finally:
        shutdown_mt5()


if __name__ == "__main__":
    sayqac = 0
    print("------------------")
    while True:
        sayqac += 1
        example_usage(sayqac)
        time.sleep(1)
