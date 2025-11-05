"""XAUUSD scalping analiz aracı (MetaTrader5).

Bu modül MetaTrader 5 terminaline bağlanır, XAUUSD için son fiyat verilerini
indirir, hızlı teknik indikatörleri hesaplar ve manuel işlemleriniz için
öneri niteliğinde sinyal çıktısı sağlar. Herhangi bir otomatik emir gönderimi
yapmaz; işlemlerinizi terminal içinde manuel olarak açmanız beklenir.

Öne çıkanlar:
    * `--terminal-path` ile yerel MT5 terminalini otomatik başlatabilir.
    * Halihazırda giriş yapılmış MT5 oturumuyla (şifre girmeden) çalışır.
    * İsteğe bağlı `--account/--password/--server` argümanlarıyla API üzerinden
      yeniden giriş yapabilirsiniz.

Gereksinimler:
    pip install MetaTrader5 pandas numpy

Önemli:
    - MT5 terminali yüklü, algoritmik işleme izin verilmiş ve (şifresiz modda
      kullanacaksanız) broker hesabındaki oturumunuz açık olmalıdır.
    - Python mimarisi (32/64 bit) MT5 terminaliyle eşleşmelidir.
    - Kaldıraçlı ürünler yüksek risk taşır; önce demo hesapta test edin.
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
    symbol: str = "XAUUSD"
    timeframe: int = mt5.TIMEFRAME_M1
    lookback: int = 600  # fetch ~10 hours of 1-minute candles
    max_spread_points: float = 30.0
    risk_per_trade: float = 0.005  # 0.5% of equity
    atr_period: int = 14
    ema_fast_period: int = 9
    ema_slow_period: int = 21
    rsi_period: int = 14
    rsi_upper: float = 65.0
    rsi_lower: float = 35.0
    reward_risk_ratio: float = 1.5
    account: Optional[int] = None
    password: Optional[str] = None
    server: Optional[str] = None
    terminal_path: Optional[str] = None
    lot: float = 0.10  # fallback lot kullanıcının manuel değerlendirmesi için
    momentum_window: int = 3
    min_momentum: float = 0.05
    min_signal_strength: float = 1.0
    min_volume_ratio: float = 1.0
    stop_atr_multiplier: float = 1.5


# ---------------------------------------------------------------------------
# MT5 helpers
# ---------------------------------------------------------------------------


def initialize_mt5(config: StrategyConfig) -> None:
    """Initialise MT5 terminal, optionally auto-launching and logging in."""

    init_kwargs = {"path": config.terminal_path} if config.terminal_path else {}
    if not mt5.initialize(**init_kwargs):
        error = mt5.last_error()
        if error and error[0] == -6:
            raise RuntimeError(
                "MT5 initialize() yetkilendirme hatası (-6). Terminali manuel olarak açıp broker hesabınıza giriş yapın "
                "ve tekrar deneyin. Eğer terminali otomatik başlatmak istiyorsanız --terminal-path ile terminal64.exe yolunu "
                "verip hesabın giriş bilgilerinin terminalde kayıtlı olduğundan emin olun."
            )
        raise RuntimeError(f"MT5 initialize() failed, error code: {error}")

    if config.account and config.password and config.server:
        authorized = mt5.login(config.account, password=config.password, server=config.server)
        if not authorized:
            last_error = mt5.last_error()
            mt5.shutdown()
            raise RuntimeError(
                f"MT5 login failed (account={config.account}), error: {last_error}"
            )
        logging.info("Logged in to MT5 account %s via API", config.account)
    else:
        logging.info("Using existing MT5 terminal session (no credentials supplied)")


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

    symbol_info = mt5.symbol_info(config.symbol)
    if symbol_info is None:
        logging.warning("Symbol info for %s unavailable; cannot compute signal", config.symbol)
        return None

    spread_points = (latest["ask"] - latest["bid"]) / symbol_info.point
    if spread_points > config.max_spread_points:
        logging.info("Spread %.1f exceeds threshold %.1f, skipping signal.", spread_points, config.max_spread_points)
        return None

    # EMA crossover logic
    bullish_cross = previous["ema_fast"] <= previous["ema_slow"] and latest["ema_fast"] > latest["ema_slow"]
    bearish_cross = previous["ema_fast"] >= previous["ema_slow"] and latest["ema_fast"] < latest["ema_slow"]

    atr_points = float(latest["atr"])
    if np.isnan(atr_points) or atr_points <= 0:
        logging.debug("ATR invalid or zero (%.4f), skipping signal", atr_points)
        return None

    volume_ratio = 1.0
    if "tick_volume" in prices.columns:
        recent_volume = prices["tick_volume"].iloc[-20:]
        avg_volume = recent_volume.mean()
        if avg_volume and avg_volume > 0:
            volume_ratio = latest["tick_volume"] / avg_volume

    if volume_ratio < config.min_volume_ratio:
        logging.debug(
            "Volume ratio %.2f below threshold %.2f, skipping signal.",
            volume_ratio,
            config.min_volume_ratio,
        )
        return None

    momentum_window = min(config.momentum_window, len(prices) - 2)
    if momentum_window <= 0:
        logging.debug("Not enough data for momentum window=%s", config.momentum_window)
        return None

    reference_close = prices["close"].iloc[-(momentum_window + 1)]
    momentum = (latest["close"] - reference_close) / atr_points if atr_points else 0.0

    bullish_stack = latest["close"] > latest["ema_fast"] > latest["ema_slow"]
    bearish_stack = latest["close"] < latest["ema_fast"] < latest["ema_slow"]
    rsi_oversold = latest["rsi"] <= config.rsi_lower
    rsi_overbought = latest["rsi"] >= config.rsi_upper

    min_momentum = max(config.min_momentum, 1e-6)
    ema_gap_buy = max(0.0, (latest["ema_fast"] - latest["ema_slow"]) / atr_points)
    ema_gap_sell = max(0.0, (latest["ema_slow"] - latest["ema_fast"]) / atr_points)

    momentum_buy_component = max(0.0, momentum) / min_momentum
    momentum_sell_component = max(0.0, -momentum) / min_momentum
    rsi_buy_component = max(0.0, (config.rsi_lower - latest["rsi"])) / max(1.0, config.rsi_lower)
    rsi_sell_component = max(0.0, (latest["rsi"] - config.rsi_upper)) / max(1.0, 100 - config.rsi_upper)

    bullish_strength = 0.0
    if bullish_cross and bullish_stack and rsi_oversold:
        bullish_strength = ema_gap_buy + momentum_buy_component + rsi_buy_component

    bearish_strength = 0.0
    if bearish_cross and bearish_stack and rsi_overbought:
        bearish_strength = ema_gap_sell + momentum_sell_component + rsi_sell_component

    selected_signal: Optional[TradeSignal] = None
    selected_strength = 0.0

    if bearish_strength >= config.min_signal_strength and momentum <= -config.min_momentum:
        entry = latest["bid"]
        stop_loss = entry + config.stop_atr_multiplier * atr_points
        take_profit = entry - config.reward_risk_ratio * (stop_loss - entry)
        selected_signal = TradeSignal(
            direction="sell",
            timestamp=latest.name.to_pydatetime(),
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment=f"EMA bear cross + RSI overbought | strength={bearish_strength:.2f} | vol={volume_ratio:.2f}",
        )
        selected_strength = bearish_strength

    if (
        bullish_strength >= config.min_signal_strength
        and momentum >= config.min_momentum
        and bullish_strength > selected_strength
    ):
        entry = latest["ask"]
        stop_loss = entry - config.stop_atr_multiplier * atr_points
        take_profit = entry + config.reward_risk_ratio * (entry - stop_loss)
        selected_signal = TradeSignal(
            direction="buy",
            timestamp=latest.name.to_pydatetime(),
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            comment=f"EMA bull cross + RSI oversold | strength={bullish_strength:.2f} | vol={volume_ratio:.2f}",
        )

    return selected_signal


def format_signal(signal: TradeSignal, config: StrategyConfig) -> str:
    """İnsan tarafından okunabilir sinyal çıktısı üret."""

    direction = "AL" if signal.direction == "buy" else "SAT"
    lines = [
        f"Sinyal: {direction}",
        f"Zaman: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Giriş fiyatı: {signal.entry:.2f}",
        f"Stop-loss:   {signal.stop_loss:.2f}",
        f"Take-profit: {signal.take_profit:.2f}",
        f"Not: {signal.comment}",
    ]
    if config.lot:
        lines.append(f"Önerilen lot referansı (manuel değerlendirme): {config.lot:.2f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI utilities
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MetaTrader5 XAUUSD scalping signal generator")
    parser.add_argument("--account", type=int, help="Override account login (optional)")
    parser.add_argument("--password", type=str, help="Override account password (optional)")
    parser.add_argument("--server", type=str, help="Override trade server name (optional)")
    parser.add_argument(
        "--terminal-path",
        type=str,
        help="Absolute path to terminal64.exe (auto-launch MT5 if given)",
    )
    parser.add_argument("--lots", type=float, default=0.10, help="Manuel işlemde referans alacağınız lot değeri")
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
        terminal_path=args.terminal_path,
        lot=args.lots,
        max_spread_points=args.max_spread,
    )

    try:
        initialize_mt5(config)
        logging.info("MT5 terminal initialised")

        raw = fetch_rates(config)
        symbol_info = mt5.symbol_info_tick(config.symbol)
        if symbol_info is None:
            raise RuntimeError(f"Symbol tick info for {config.symbol} unavailable")

        raw["bid"] = symbol_info.bid
        raw["ask"] = symbol_info.ask

        enriched = compute_indicators(raw, config)
        signal = generate_signal(enriched, config)

        if signal:
            logging.info("Sinyal bulundu:\n%s", format_signal(signal, config))
        else:
            logging.info("No valid signal at %s", datetime.now(timezone.utc))

    finally:
        shutdown_mt5()


if __name__ == "__main__":
    main()

