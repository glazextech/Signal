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


_LOGGER = logging.getLogger(__name__)


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
        _LOGGER.info("Logged in to MT5 account %s via API", config.account)
    else:
        _LOGGER.info("Using existing MT5 terminal session (no credentials supplied)")


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
    if df.empty:
        raise RuntimeError(f"No rates returned for {config.symbol} (zero-length dataset)")

    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)
    return df


# ---------------------------------------------------------------------------
# Indicator engine
# ---------------------------------------------------------------------------


def compute_indicators(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Append EMA, RSI, ATR indicators to the price DataFrame."""

    prices = df.sort_index().copy()

    required_bars = max(config.ema_slow_period, config.rsi_period, config.atr_period) + 1
    if len(prices) < required_bars:
        raise ValueError(
            f"Indicator hesaplamaları için yeterli veri yok (gereken: {required_bars}, mevcut: {len(prices)})"
        )

    prices["ema_fast"] = prices["close"].ewm(span=config.ema_fast_period, adjust=False).mean()
    prices["ema_slow"] = prices["close"].ewm(span=config.ema_slow_period, adjust=False).mean()

    delta = prices["close"].diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1 / config.rsi_period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / config.rsi_period, adjust=False).mean()
    loss = loss.replace(0, np.nan)
    rs = gain / loss
    prices["rsi"] = 100 - (100 / (1 + rs))
    prices["rsi"] = prices["rsi"].clip(lower=0, upper=100)

    tr = np.maximum(
        prices["high"] - prices["low"],
        np.maximum(
            prices["high"] - prices["close"].shift(1),
            prices["close"].shift(1) - prices["low"],
        ),
    )
    prices["atr"] = tr.ewm(span=config.atr_period, adjust=False).mean()

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

    if len(prices.index) < 2:
        _LOGGER.debug("Not enough bars (%d) to generate a signal", len(prices))
        return None

    for column in ("ema_fast", "ema_slow", "rsi", "atr"):
        if column not in prices.columns:
            _LOGGER.debug("Column %s missing from price DataFrame; skipping signal", column)
            return None

    latest = prices.iloc[-1]
    previous = prices.iloc[-2]

    latest_bid = float(latest.get("bid", latest["close"]))
    latest_ask = float(latest.get("ask", latest["close"]))

    if any(np.isnan([latest_bid, latest_ask])):
        _LOGGER.debug("Bid/ask verisi eksik; sinyal üretimi atlandı")
        return None

    if pd.isna(latest["atr"]):
        _LOGGER.debug("ATR bilgisi mevcut değil; sinyal üretimi atlandı")
        return None

    indicator_slice = prices[["ema_fast", "ema_slow", "rsi", "atr"]].tail(2)
    if indicator_slice.isna().any().any():
        _LOGGER.debug("Gerekli indikatör değerleri NaN içeriyor; sinyal atlandı")
        return None

    symbol_info = mt5.symbol_info(config.symbol)
    if symbol_info is None:
        _LOGGER.warning("Symbol info for %s unavailable; cannot compute signal", config.symbol)
        return None

    spread_points = (latest_ask - latest_bid) / symbol_info.point
    if spread_points > config.max_spread_points:
        _LOGGER.info(
            "Spread %.1f exceeds threshold %.1f, skipping signal.",
            spread_points,
            config.max_spread_points,
        )
        return None

    # EMA crossover logic
    bullish_cross = previous["ema_fast"] <= previous["ema_slow"] and latest["ema_fast"] > latest["ema_slow"]
    bearish_cross = previous["ema_fast"] >= previous["ema_slow"] and latest["ema_fast"] < latest["ema_slow"]

    atr_points = latest["atr"]

    if bullish_cross and latest["rsi"] < config.rsi_upper:
        entry = latest_ask
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
        entry = latest_bid
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
        _LOGGER.info("MT5 terminal initialised")

        raw = fetch_rates(config)
        tick_info = mt5.symbol_info_tick(config.symbol)
        if tick_info is None:
            raise RuntimeError(f"Symbol tick info for {config.symbol} unavailable")

        raw = raw.copy()
        raw.loc[raw.index[-1], "bid"] = tick_info.bid
        raw.loc[raw.index[-1], "ask"] = tick_info.ask

        try:
            enriched = compute_indicators(raw, config)
        except ValueError as indicator_error:
            _LOGGER.warning("Indicator hesaplaması atlandı: %s", indicator_error)
            return

        signal = generate_signal(enriched, config)

        if signal:
            _LOGGER.info("Sinyal bulundu:\n%s", format_signal(signal, config))
        else:
            _LOGGER.info("No valid signal at %s", datetime.now(timezone.utc))

    finally:
        shutdown_mt5()


if __name__ == "__main__":
    main()

