"""Fixed-time trade signal generator leveraging MetaTrader 5 market data.

This module connects to the locally installed MetaTrader 5 terminal, streams
recent candles and live ticks for a chosen symbol, performs comprehensive
technical analysis, and emits auditable buy/sell recommendations. The script
never places trades automatically: it prints an OCO-style (one-cancels-other)
plan comprising entry, stop-loss, and target levels so a human trader can act
manually.

Highlights
~~~~~~~~~~
* Native MetaTrader 5 data feed (no web scraping) with automatic terminal
  initialisation and optional credential-based login.
* Indicator suite covering EMAs, RSI, MACD, momentum, rolling volatility, and
  ATR, implemented with :mod:`pandas`/:mod:`numpy` for determinism.
* Probabilistic signal scoring that converts indicator confluence into a
  calibrated logistic probability for long/short candidates.
* Risk management based on ATR-derived stop distance, configurable risk per
  trade, and maximum stake limits.
* Dynamic configuration reload from JSON/YAML so parameters can be tuned at
  runtime without restarting the script.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import signal
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import MetaTrader5 as mt5  # type: ignore
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Timeframe helpers
# ---------------------------------------------------------------------------


TIMEFRAME_ALIASES = {
    "1m": mt5.TIMEFRAME_M1,
    "m1": mt5.TIMEFRAME_M1,
    "60s": mt5.TIMEFRAME_M1,
    "5m": mt5.TIMEFRAME_M5,
    "m5": mt5.TIMEFRAME_M5,
    "15m": mt5.TIMEFRAME_M15,
    "m15": mt5.TIMEFRAME_M15,
    "30m": mt5.TIMEFRAME_M30,
    "m30": mt5.TIMEFRAME_M30,
    "1h": mt5.TIMEFRAME_H1,
    "h1": mt5.TIMEFRAME_H1,
    "4h": mt5.TIMEFRAME_H4,
    "h4": mt5.TIMEFRAME_H4,
    "1d": mt5.TIMEFRAME_D1,
    "d1": mt5.TIMEFRAME_D1,
}


def resolve_timeframe(label: str) -> int:
    """Convert human-readable timeframe (e.g. ``"1m"``) to MT5 enum."""

    normalised = label.strip().lower().replace(" ", "")
    if normalised not in TIMEFRAME_ALIASES:
        raise ValueError(f"Unsupported timeframe '{label}'. Supported keys: {sorted(TIMEFRAME_ALIASES)}")
    return TIMEFRAME_ALIASES[normalised]


# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class IndicatorSettings:
    """Indicator lookback configuration."""

    ema_fast_period: int = 9
    ema_slow_period: int = 21
    ema_trend_period: int = 55
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    momentum_period: int = 10
    volatility_period: int = 20
    atr_period: int = 14


@dataclass(slots=True)
class RiskSettings:
    """Risk management configuration."""

    account_balance: float = 1000.0
    risk_per_trade: float = 0.01
    max_trade_size: float = 100.0
    min_probability: float = 0.55
    atr_stop_multiplier: float = 1.8
    atr_target_multiplier: float = 2.7
    reward_risk_ratio: float = 1.5


@dataclass(slots=True)
class StrategyConfig:
    """Top-level strategy configuration."""

    symbol: str = "EURUSD"
    timeframe: str = "1m"
    history_candles: int = 600
    warmup_candles: int = 150
    loop_seconds: int = 30
    max_spread_points: float = 25.0
    account: Optional[int] = None
    password: Optional[str] = None
    server: Optional[str] = None
    terminal_path: Optional[str] = None
    config_path: Optional[Path] = None
    indicators: IndicatorSettings = field(default_factory=IndicatorSettings)
    risk: RiskSettings = field(default_factory=RiskSettings)

    @property
    def timeframe_id(self) -> int:
        return resolve_timeframe(self.timeframe)

    @staticmethod
    def from_mapping(mapping: Dict[str, Any]) -> "StrategyConfig":
        """Hydrate a :class:`StrategyConfig` from a nested mapping."""

        def filter_kwargs(cls: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
            allowed = set(cls.__dataclass_fields__.keys())  # type: ignore[attr-defined]
            return {k: v for k, v in payload.items() if k in allowed}

        indicator_payload = mapping.get("indicators", {}) or {}
        risk_payload = mapping.get("risk", {}) or {}

        indicators = IndicatorSettings(**filter_kwargs(IndicatorSettings, indicator_payload))
        risk = RiskSettings(**filter_kwargs(RiskSettings, risk_payload))

        core_allowed = set(StrategyConfig.__dataclass_fields__.keys()) - {"indicators", "risk"}
        core = {k: v for k, v in mapping.items() if k in core_allowed}
        if "config_path" in core and core["config_path"] is not None:
            core["config_path"] = Path(core["config_path"])
        return StrategyConfig(indicators=indicators, risk=risk, **core)  # type: ignore[arg-type]


class ConfigManager:
    """Load and optionally hot-reload strategy configuration."""

    def __init__(self, initial: StrategyConfig) -> None:
        self._config = initial
        self._config_mtime: Optional[float] = None
        if initial.config_path:
            mtime = self._get_mtime(initial.config_path)
            if mtime is not None:
                overrides = self._load_file(initial.config_path)
                self._config = StrategyConfig.from_mapping({**asdict(initial), **overrides})
                self._config_mtime = mtime
            else:
                logging.warning(
                    "Config file %s not found at startup; waiting for creation",
                    initial.config_path,
                )

    @property
    def config(self) -> StrategyConfig:
        return self._config

    def maybe_reload(self) -> StrategyConfig:
        path = self._config.config_path
        if not path:
            return self._config
        mtime = self._get_mtime(path)
        if mtime is None or mtime == self._config_mtime:
            return self._config
        logging.info("Reloading strategy configuration from %s", path)
        overrides = self._load_file(path)
        self._config = StrategyConfig.from_mapping({**asdict(self._config), **overrides})
        self._config_mtime = mtime
        return self._config

    @staticmethod
    def _load_file(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise RuntimeError("Missing dependency 'pyyaml'. Install it to use YAML configs.") from exc
            with path.open("r", encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _get_mtime(path: Path) -> Optional[float]:
        try:
            return path.stat().st_mtime
        except FileNotFoundError:
            return None


# ---------------------------------------------------------------------------
# MetaTrader 5 initialisation helpers
# ---------------------------------------------------------------------------


def initialize_mt5(config: StrategyConfig) -> None:
    """Initialise the MetaTrader 5 terminal and authenticate if required."""

    init_kwargs: Dict[str, Any] = {}
    if config.terminal_path:
        init_kwargs["path"] = str(config.terminal_path)
    if not mt5.initialize(**init_kwargs):
        error = mt5.last_error()
        raise RuntimeError(f"MT5 initialize() failed: {error}")

    if config.account and config.password and config.server:
        if not mt5.login(config.account, password=config.password, server=config.server):
            last_error = mt5.last_error()
            mt5.shutdown()
            raise RuntimeError(f"MT5 login failed for account {config.account}: {last_error}")
        logging.info("Logged in to MT5 account %s", config.account)
    else:
        logging.info("Using active MT5 terminal session (no credentials supplied)")

    if not mt5.symbol_select(config.symbol, True):
        mt5.shutdown()
        raise RuntimeError(f"Failed to select symbol {config.symbol}")


def shutdown_mt5() -> None:
    """Gracefully terminate the MT5 connection."""

    mt5.shutdown()


# ---------------------------------------------------------------------------
# Data acquisition layer
# ---------------------------------------------------------------------------


class MT5DataClient:
    """Fetch candles and ticks from MetaTrader 5."""

    def __init__(self, config: StrategyConfig) -> None:
        self._config = config
        self._timeframe = config.timeframe_id

    def refresh(self, config: StrategyConfig) -> None:
        self._config = config
        self._timeframe = config.timeframe_id
        if not mt5.symbol_select(config.symbol, True):
            raise RuntimeError(f"Failed to select symbol {config.symbol}")

    def fetch_candles(self, limit: Optional[int] = None) -> pd.DataFrame:
        count = limit or self._config.history_candles
        rates = mt5.copy_rates_from_pos(self._config.symbol, self._timeframe, 0, count)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No rates returned for {self._config.symbol}: {mt5.last_error()}")

        frame = pd.DataFrame(rates)
        frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
        frame = frame.set_index("time").rename(
            columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "tick_volume": "volume",
            }
        )
        frame = frame[["open", "high", "low", "close", "volume"]]
        return frame

    def get_tick_and_info(self) -> tuple[Any, Any]:
        symbol_info = mt5.symbol_info(self._config.symbol)
        if symbol_info is None:
            raise RuntimeError(f"Symbol info unavailable for {self._config.symbol}")
        tick = mt5.symbol_info_tick(self._config.symbol)
        if tick is None:
            raise RuntimeError(f"Tick info unavailable for {self._config.symbol}")
        return tick, symbol_info


# ---------------------------------------------------------------------------
# Indicator engine
# ---------------------------------------------------------------------------


class IndicatorEngine:
    """Compute technical indicators needed for the strategy."""

    def __init__(self, settings: IndicatorSettings) -> None:
        self._settings = settings

    def enrich(self, candles: pd.DataFrame) -> pd.DataFrame:
        data = candles.copy()
        closes = data["close"]
        settings = self._settings

        data["ema_fast"] = closes.ewm(span=settings.ema_fast_period, adjust=False).mean()
        data["ema_slow"] = closes.ewm(span=settings.ema_slow_period, adjust=False).mean()
        data["ema_trend"] = closes.ewm(span=settings.ema_trend_period, adjust=False).mean()

        delta = closes.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / settings.rsi_period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / settings.rsi_period, adjust=False).mean()
        rs = gain / loss.replace(to_replace=0, value=np.nan)
        data["rsi"] = 100 - (100 / (1 + rs))

        ema_fast = closes.ewm(span=settings.macd_fast, adjust=False).mean()
        ema_slow = closes.ewm(span=settings.macd_slow, adjust=False).mean()
        data["macd"] = ema_fast - ema_slow
        data["macd_signal"] = data["macd"].ewm(span=settings.macd_signal, adjust=False).mean()
        data["macd_hist"] = data["macd"] - data["macd_signal"]

        data["momentum"] = closes.pct_change(periods=settings.momentum_period)

        returns = closes.pct_change()
        data["volatility"] = returns.rolling(window=settings.volatility_period).std() * math.sqrt(
            settings.volatility_period
        )

        high_low = data["high"] - data["low"]
        high_close = (data["high"] - data["close"].shift()).abs()
        low_close = (data["low"] - data["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data["atr"] = tr.rolling(window=settings.atr_period, min_periods=1).mean()

        data.dropna(inplace=True)
        return data


# ---------------------------------------------------------------------------
# Risk management
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SignalCandidate:
    direction: str
    entry: float
    stop_loss: float
    take_profit: float
    probability: float
    score: float
    timestamp: pd.Timestamp
    notes: str
    spread_points: float


class RiskManager:
    """Convert signals into actionable trade plans respecting risk settings."""

    def __init__(self, settings: RiskSettings) -> None:
        self._settings = settings

    def position_size(self, entry: float, stop_loss: float) -> float:
        risk_settings = self._settings
        risk_amount = risk_settings.account_balance * risk_settings.risk_per_trade
        stop_distance = max(abs(entry - stop_loss), 1e-8)
        if not math.isfinite(stop_distance):
            return 0.0
        raw_size = risk_amount / stop_distance
        recommended = min(raw_size, risk_settings.max_trade_size)
        logging.debug(
            "Position sizing: risk=%s stop_distance=%s raw_size=%s recommended=%s",
            risk_amount,
            stop_distance,
            raw_size,
            recommended,
        )
        return max(recommended, 0.0)

    def is_probability_acceptable(self, probability: float) -> bool:
        return probability >= self._settings.min_probability


# ---------------------------------------------------------------------------
# Signal evaluation
# ---------------------------------------------------------------------------


class SignalEngine:
    """Generate probabilistic trade candidates from enriched data."""

    def __init__(self, config: StrategyConfig) -> None:
        self._config = config

    def evaluate(
        self,
        data: pd.DataFrame,
        tick: Any,
        symbol_info: Any,
        spread_points: float,
    ) -> Optional[SignalCandidate]:
        if data.empty:
            return None

        latest = data.iloc[-1]
        previous = data.iloc[-2] if len(data) > 1 else latest

        features = self._collect_features(latest, previous)
        prob_long, prob_short = self._score_probabilities(features)

        logging.debug(
            "Features: %s prob_long=%.3f prob_short=%.3f",
            features,
            prob_long,
            prob_short,
        )

        if prob_long < self._config.risk.min_probability and prob_short < self._config.risk.min_probability:
            return None

        direction = "buy" if prob_long >= prob_short else "sell"
        probability = max(prob_long, prob_short)
        atr = float(latest["atr"])
        close_price = float(latest["close"])

        tick_bid = float(getattr(tick, "bid", float("nan")))
        tick_ask = float(getattr(tick, "ask", float("nan")))

        if direction == "buy":
            entry = tick_ask if tick_ask > 0 else close_price
            stop_loss = entry - self._config.risk.atr_stop_multiplier * atr
            reward_rr = self._config.risk.reward_risk_ratio * (entry - stop_loss)
            reward_atr = self._config.risk.atr_target_multiplier * atr
            take_profit = entry + min(reward_rr, reward_atr)
            score = prob_long
        else:
            entry = tick_bid if tick_bid > 0 else close_price
            stop_loss = entry + self._config.risk.atr_stop_multiplier * atr
            reward_rr = self._config.risk.reward_risk_ratio * (stop_loss - entry)
            reward_atr = self._config.risk.atr_target_multiplier * atr
            take_profit = entry - min(reward_rr, reward_atr)
            score = prob_short

        stop_loss = float(stop_loss)
        take_profit = float(max(take_profit, 0.0))

        notes = self._format_notes(features, direction)
        return SignalCandidate(
            direction=direction,
            entry=float(entry),
            stop_loss=stop_loss,
            take_profit=take_profit,
            probability=probability,
            score=score,
            timestamp=latest.name,
            notes=notes,
            spread_points=spread_points,
        )

    @staticmethod
    def _collect_features(latest: pd.Series, previous: pd.Series) -> Dict[str, float]:
        cross_up = bool(latest["ema_fast"] > latest["ema_slow"]) and bool(
            previous["ema_fast"] <= previous["ema_slow"]
        )
        cross_down = bool(latest["ema_fast"] < latest["ema_slow"]) and bool(
            previous["ema_fast"] >= previous["ema_slow"]
        )
        ema_cross = 1.0 if cross_up else (-1.0 if cross_down else 0.0)

        return {
            "ema_gap": float(latest["ema_fast"] - latest["ema_slow"]),
            "ema_trend": float(latest["close"] - latest["ema_trend"]),
            "ema_cross": ema_cross,
            "rsi": float(latest["rsi"]),
            "macd_hist": float(latest["macd_hist"]),
            "momentum": float(latest["momentum"]),
            "volatility": float(latest["volatility"]),
            "atr": float(latest["atr"]),
        }

    def _score_probabilities(self, features: Dict[str, float]) -> tuple[float, float]:
        weight_long = {
            "ema_gap": 2.0,
            "ema_trend": 1.2,
            "ema_cross": 1.5,
            "rsi": -0.04,
            "macd_hist": 1.8,
            "momentum": 1.2,
            "volatility": -0.8,
        }
        weight_short = {
            "ema_gap": -2.0,
            "ema_trend": -1.2,
            "ema_cross": -1.5,
            "rsi": 0.04,
            "macd_hist": -1.8,
            "momentum": -1.2,
            "volatility": -0.8,
        }

        bias = -0.1

        def sigmoid(x: float) -> float:
            return 1 / (1 + math.exp(-x))

        long_score = bias
        short_score = bias
        normalisers = {
            "ema_gap": 1.0,
            "ema_trend": 1.0,
            "ema_cross": 1.0,
            "rsi": 50.0,
            "macd_hist": 0.0005,
            "momentum": 0.5,
            "volatility": 0.02,
        }

        for key, value in features.items():
            normalised = value / normalisers.get(key, 1.0)
            long_score += weight_long.get(key, 0.0) * normalised
            short_score += weight_short.get(key, 0.0) * normalised

        return sigmoid(long_score), sigmoid(short_score)

    @staticmethod
    def _format_notes(features: Dict[str, float], direction: str) -> str:
        parts = [
            f"EMA gap {features['ema_gap']:.5f}",
            f"RSI {features['rsi']:.2f}",
            f"MACD hist {features['macd_hist']:.5f}",
            f"Momentum {features['momentum']:.4f}",
            f"Volatility {features['volatility']:.4f}",
            f"ATR {features['atr']:.5f}",
        ]
        return f"{direction.upper()} bias | " + " | ".join(parts)


# ---------------------------------------------------------------------------
# Terminal reporting
# ---------------------------------------------------------------------------


class SignalPrinter:
    """Render signals to stdout in an auditable, OCO-style format."""

    def __init__(self) -> None:
        self._last_timestamp: Optional[pd.Timestamp] = None

    def emit(self, strategy: StrategyConfig, candidate: SignalCandidate, risk: RiskManager) -> None:
        if candidate.timestamp == self._last_timestamp:
            logging.debug("Signal already emitted for timestamp %s", candidate.timestamp)
            return

        size = risk.position_size(candidate.entry, candidate.stop_loss)
        if size <= 0:
            logging.info("Calculated position size is zero; skipping output")
            return

        lines = [
            "=" * 72,
            f"{candidate.timestamp.isoformat()} | {strategy.symbol} | {strategy.timeframe}",
            f"RECOMMENDATION: {candidate.direction.upper()} (prob. {candidate.probability:.2%}, score {candidate.score:.3f})",
            f"ENTRY @ {candidate.entry:.5f}",
            f"STOP  @ {candidate.stop_loss:.5f} ({strategy.risk.atr_stop_multiplier:.2f} x ATR)",
            f"TARGET@ {candidate.take_profit:.5f} (R/R {strategy.risk.reward_risk_ratio:.2f})",
            f"SIZE  ≈ {size:.2f} (risk {strategy.risk.risk_per_trade:.2%} of equity {strategy.risk.account_balance:.2f})",
            f"SPREAD≈ {candidate.spread_points:.1f} pts (limit {strategy.max_spread_points:.1f})",
            f"NOTES: {candidate.notes}",
            "=" * 72,
        ]
        print("\n".join(lines), flush=True)
        self._last_timestamp = candidate.timestamp


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_strategy(
    config_manager: ConfigManager,
    once: bool,
    stop_flag: Optional[Callable[[], bool]] = None,
) -> None:
    printer = SignalPrinter()
    active_config = config_manager.config
    indicator_engine = IndicatorEngine(active_config.indicators)
    signal_engine = SignalEngine(active_config)
    risk_manager = RiskManager(active_config.risk)
    data_client = MT5DataClient(active_config)

    while True:
        if stop_flag and stop_flag():
            break

        try:
            updated_config = config_manager.maybe_reload()
        except Exception as exc:
            logging.warning("Config reload failed: %s", exc)
            updated_config = active_config

        if updated_config != active_config:
            logging.info("Config change detected; rebuilding components")
            active_config = updated_config
            indicator_engine = IndicatorEngine(active_config.indicators)
            signal_engine = SignalEngine(active_config)
            risk_manager = RiskManager(active_config.risk)
            data_client.refresh(active_config)

        try:
            candles = data_client.fetch_candles(active_config.history_candles)
        except Exception as exc:
            logging.error("Failed to fetch candles: %s", exc)
            if once:
                break
            time.sleep(active_config.loop_seconds)
            continue

        enriched = indicator_engine.enrich(candles)
        if len(enriched) < active_config.warmup_candles:
            logging.debug(
                "Warmup incomplete (%s/%s candles)", len(enriched), active_config.warmup_candles
            )
            if once:
                break
            time.sleep(active_config.loop_seconds)
            continue

        try:
            tick, symbol_info = data_client.get_tick_and_info()
        except Exception as exc:
            logging.error("Failed to fetch tick data: %s", exc)
            if once:
                break
            time.sleep(active_config.loop_seconds)
            continue

        point = float(getattr(symbol_info, "point", 0.0) or 0.0)
        bid = float(getattr(tick, "bid", 0.0) or 0.0)
        ask = float(getattr(tick, "ask", 0.0) or 0.0)
        if point <= 0 or bid <= 0 or ask <= 0:
            logging.warning("Invalid tick data received; skipping cycle")
            if once:
                break
            time.sleep(active_config.loop_seconds)
            continue

        spread_points = (ask - bid) / point
        if spread_points > active_config.max_spread_points:
            logging.info(
                "Spread %.1f exceeds threshold %.1f; skipping",
                spread_points,
                active_config.max_spread_points,
            )
        else:
            candidate = signal_engine.evaluate(enriched, tick, symbol_info, spread_points)
            if candidate and risk_manager.is_probability_acceptable(candidate.probability):
                printer.emit(active_config, candidate, risk_manager)
            else:
                logging.info(
                    "No trade candidate (probability below %.2f)",
                    active_config.risk.min_probability,
                )

        if once:
            break
        time.sleep(active_config.loop_seconds)


# ---------------------------------------------------------------------------
# CLI handling
# ---------------------------------------------------------------------------


def parse_arguments(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MetaTrader5 fixed-time trade signal generator (no auto-trading)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbol", default="EURUSD", help="Instrument symbol as defined in MT5")
    parser.add_argument("--timeframe", default="1m", help="Timeframe (e.g. 1m, 5m, 15m, 1h)")
    parser.add_argument("--history", type=int, default=600, help="Number of candles to request each cycle")
    parser.add_argument("--warmup", type=int, default=150, help="Minimum candles required before scoring")
    parser.add_argument("--loop-seconds", type=int, default=30, help="Seconds between evaluations")
    parser.add_argument("--max-spread", type=float, default=25.0, help="Maximum spread (points) to allow a trade")
    parser.add_argument("--account", type=int, help="MT5 account number for API login")
    parser.add_argument("--password", type=str, help="MT5 account password")
    parser.add_argument("--server", type=str, help="MT5 trade server name")
    parser.add_argument("--terminal-path", type=Path, help="Path to terminal64.exe (optional auto launch)")
    parser.add_argument("--account-balance", type=float, default=1000.0, help="Account equity used for sizing")
    parser.add_argument("--risk-per-trade", type=float, default=0.01, help="Risk percentage per trade (0-1)")
    parser.add_argument("--max-trade-size", type=float, default=100.0, help="Upper cap on trade size units")
    parser.add_argument("--min-probability", type=float, default=0.55, help="Minimum probability to emit a signal")
    parser.add_argument("--config-path", type=Path, help="Optional JSON/YAML file for hot-reload configuration")
    parser.add_argument("--log-level", default="INFO", help="Logging verbosity (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--once", action="store_true", help="Run a single evaluation cycle and exit")
    return parser.parse_args(list(argv) if argv is not None else None)


def build_config_from_args(args: argparse.Namespace) -> StrategyConfig:
    base_config = StrategyConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        history_candles=args.history,
        warmup_candles=args.warmup,
        loop_seconds=args.loop_seconds,
        max_spread_points=args.max_spread,
        account=args.account,
        password=args.password,
        server=args.server,
        terminal_path=args.terminal_path,
        config_path=args.config_path,
    )

    risk = replace(
        base_config.risk,
        account_balance=args.account_balance,
        risk_per_trade=args.risk_per_trade,
        max_trade_size=args.max_trade_size,
        min_probability=args.min_probability,
    )

    return replace(base_config, risk=risk)


def configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_arguments(argv)
    configure_logging(args.log_level)
    base_config = build_config_from_args(args)
    config_manager = ConfigManager(base_config)

    if args.config_path and not args.config_path.exists():
        logging.warning("Config file %s does not exist yet; continuing with CLI parameters", args.config_path)

    try:
        initialize_mt5(config_manager.config)
    except Exception as exc:
        logging.exception("Failed to initialise MetaTrader 5: %s", exc)
        return 1

    stop_requested = False

    def _handle_signal(signum, frame):  # type: ignore[override]
        nonlocal stop_requested
        logging.info("Signal %s received; shutting down after current cycle", signum)
        stop_requested = True

    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    try:
        run_strategy(config_manager, once=args.once, stop_flag=lambda: stop_requested)
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        shutdown_mt5()

    if stop_requested:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

