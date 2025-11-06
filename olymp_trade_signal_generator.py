"""Fixed-time trading signal generator built on MetaTrader 5 market data.

This script connects to a locally installed MetaTrader 5 terminal, analyses
recent price action, and emits high-confidence CALL/PUT recommendations for
60-second fixed-time contracts. Signals are informational only; no trades are
executed automatically.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import signal
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

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
    "15m": mt5.TIMEFRAME_M15,
    "30m": mt5.TIMEFRAME_M30,
    "1h": mt5.TIMEFRAME_H1,
    "h1": mt5.TIMEFRAME_H1,
}


def resolve_timeframe(label: str) -> int:
    normalised = label.strip().lower()
    if normalised not in TIMEFRAME_ALIASES:
        raise ValueError(f"Unsupported timeframe '{label}'. Supported values: {sorted(TIMEFRAME_ALIASES)}")
    return TIMEFRAME_ALIASES[normalised]


# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class IndicatorSettings:
    ema_fast: int = 5
    ema_mid: int = 13
    ema_slow: int = 34
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    momentum_period_short: int = 3
    momentum_period_long: int = 8
    volatility_period: int = 20
    atr_period: int = 14
    stoch_k_period: int = 8
    stoch_d_period: int = 3
    stoch_smoothing: int = 3
    bollinger_period: int = 20
    bollinger_std: float = 2.0


@dataclass(slots=True)
class RiskSettings:
    account_balance: float = 1000.0
    risk_per_trade: float = 0.02
    max_trade_size: float = 100.0
    min_probability: float = 0.55
    max_consecutive_same_direction: int = 2


@dataclass(slots=True)
class TradeSettings:
    expiry_seconds: int = 60
    entry_window_seconds: int = 15
    min_confidence_edge: float = 0.07
    min_momentum: float = 0.0001
    max_spread_points: float = 25.0
    entry_buffer_points: float = 3.0


@dataclass(slots=True)
class StrategyConfig:
    symbol: str = "EURUSD"
    timeframe: str = "1m"
    history_candles: int = 600
    warmup_candles: int = 150
    loop_seconds: int = 30
    account: Optional[int] = None
    password: Optional[str] = None
    server: Optional[str] = None
    terminal_path: Optional[Path] = None
    config_path: Optional[Path] = None
    indicators: IndicatorSettings = field(default_factory=IndicatorSettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    trade: TradeSettings = field(default_factory=TradeSettings)

    @property
    def timeframe_id(self) -> int:
        return resolve_timeframe(self.timeframe)

    @staticmethod
    def from_mapping(mapping: Dict[str, Any]) -> "StrategyConfig":
        def filter_fields(cls: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
            allowed = set(cls.__dataclass_fields__.keys())  # type: ignore[attr-defined]
            return {k: v for k, v in payload.items() if k in allowed}

        indicators = IndicatorSettings(**filter_fields(IndicatorSettings, mapping.get("indicators", {}) or {}))
        risk = RiskSettings(**filter_fields(RiskSettings, mapping.get("risk", {}) or {}))
        trade = TradeSettings(**filter_fields(TradeSettings, mapping.get("trade", {}) or {}))

        core_allowed = set(StrategyConfig.__dataclass_fields__.keys()) - {"indicators", "risk", "trade"}
        core = {k: v for k, v in mapping.items() if k in core_allowed}
        if "config_path" in core and core["config_path"] is not None:
            core["config_path"] = Path(core["config_path"])
        return StrategyConfig(indicators=indicators, risk=risk, trade=trade, **core)  # type: ignore[arg-type]


class ConfigManager:
    """Load and hot-reload configuration from disk."""

    def __init__(self, initial: StrategyConfig) -> None:
        self._config = initial
        self._config_mtime: Optional[float] = None
        if initial.config_path:
            mtime = self._stat(initial.config_path)
            if mtime is not None:
                overrides = self._load(initial.config_path)
                self._config = StrategyConfig.from_mapping({**asdict(initial), **overrides})
                self._config_mtime = mtime
            else:
                logging.warning("Config file %s not found; watching for creation", initial.config_path)

    @property
    def config(self) -> StrategyConfig:
        return self._config

    def maybe_reload(self) -> StrategyConfig:
        path = self._config.config_path
        if not path:
            return self._config
        mtime = self._stat(path)
        if mtime is None or mtime == self._config_mtime:
            return self._config
        logging.info("Reloading strategy configuration from %s", path)
        overrides = self._load(path)
        self._config = StrategyConfig.from_mapping({**asdict(self._config), **overrides})
        self._config_mtime = mtime
        return self._config

    @staticmethod
    def _load(path: Path) -> Dict[str, Any]:
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
    def _stat(path: Path) -> Optional[float]:
        try:
            return path.stat().st_mtime
        except FileNotFoundError:
            return None


# ---------------------------------------------------------------------------
# MetaTrader 5 helpers
# ---------------------------------------------------------------------------


def initialize_mt5(config: StrategyConfig) -> None:
    init_kwargs: Dict[str, Any] = {}
    if config.terminal_path:
        init_kwargs["path"] = str(config.terminal_path)
    if not mt5.initialize(**init_kwargs):
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")

    if config.account and config.password and config.server:
        if not mt5.login(config.account, password=config.password, server=config.server):
            last_error = mt5.last_error()
            mt5.shutdown()
            raise RuntimeError(f"MT5 login failed for account {config.account}: {last_error}")
        logging.info("Logged in to MT5 account %s", config.account)
    else:
        logging.info("Using existing MT5 session; ensure terminal is authorised")

    if not mt5.symbol_select(config.symbol, True):
        mt5.shutdown()
        raise RuntimeError(f"Failed to select symbol {config.symbol}")


def shutdown_mt5() -> None:
    mt5.shutdown()


# ---------------------------------------------------------------------------
# Data acquisition layer
# ---------------------------------------------------------------------------


class MT5DataClient:
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

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time")
        df = df.rename(columns={"tick_volume": "volume"})
        return df[["open", "high", "low", "close", "volume"]]

    def get_tick_and_info(self) -> Tuple[Any, Any]:
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
    def __init__(self, settings: IndicatorSettings) -> None:
        self._settings = settings

    def enrich(self, candles: pd.DataFrame) -> pd.DataFrame:
        data = candles.copy()
        settings = self._settings

        closes = data["close"]
        data["ema_fast"] = closes.ewm(span=settings.ema_fast, adjust=False).mean()
        data["ema_mid"] = closes.ewm(span=settings.ema_mid, adjust=False).mean()
        data["ema_slow"] = closes.ewm(span=settings.ema_slow, adjust=False).mean()
        data["ema_fast_slope"] = data["ema_fast"].diff()
        data["ema_mid_slope"] = data["ema_mid"].diff()

        delta = closes.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / settings.rsi_period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / settings.rsi_period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        data["rsi"] = 100 - (100 / (1 + rs))

        ema_fast = closes.ewm(span=settings.macd_fast, adjust=False).mean()
        ema_slow = closes.ewm(span=settings.macd_slow, adjust=False).mean()
        data["macd"] = ema_fast - ema_slow
        data["macd_signal"] = data["macd"].ewm(span=settings.macd_signal, adjust=False).mean()
        data["macd_hist"] = data["macd"] - data["macd_signal"]

        data["momentum_short"] = closes.pct_change(settings.momentum_period_short)
        data["momentum_long"] = closes.pct_change(settings.momentum_period_long)

        returns = closes.pct_change()
        data["volatility"] = returns.rolling(window=settings.volatility_period).std()

        high_low = data["high"] - data["low"]
        high_close = (data["high"] - data["close"].shift()).abs()
        low_close = (data["low"] - data["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data["atr"] = tr.rolling(window=settings.atr_period, min_periods=1).mean()

        highest_high = data["high"].rolling(window=settings.stoch_k_period).max()
        lowest_low = data["low"].rolling(window=settings.stoch_k_period).min()
        stoch_raw = ((closes - lowest_low) / (highest_high - lowest_low)).replace([np.inf, -np.inf], np.nan)
        data["stoch_k"] = stoch_raw.rolling(window=settings.stoch_smoothing).mean() * 100
        data["stoch_d"] = data["stoch_k"].rolling(window=settings.stoch_d_period).mean()

        bollinger_mid = closes.rolling(window=settings.bollinger_period).mean()
        bollinger_std = closes.rolling(window=settings.bollinger_period).std(ddof=0)
        data["boll_mid"] = bollinger_mid
        data["boll_upper"] = bollinger_mid + settings.bollinger_std * bollinger_std
        data["boll_lower"] = bollinger_mid - settings.bollinger_std * bollinger_std

        body = data["close"] - data["open"]
        range_ = (data["high"] - data["low"]).replace(0, np.nan)
        data["body_relative"] = body / range_

        data.dropna(inplace=True)
        return data


# ---------------------------------------------------------------------------
# Signal evaluation components
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SignalCandidate:
    direction: str  # "call" or "put"
    confidence: float
    entry_price: float
    reference_price: float
    spread_points: float
    expiry_time: datetime
    entry_deadline: datetime
    invalidation_price: float
    momentum_short: float
    notes: List[str]
    timestamp: pd.Timestamp


class RiskManager:
    def __init__(self, settings: RiskSettings) -> None:
        self._settings = settings
        self._last_direction: Optional[str] = None
        self._same_direction_count = 0

    def stake_size(self, confidence: float) -> float:
        base = self._settings.account_balance * self._settings.risk_per_trade
        scaled = max(0.5, min(confidence, 0.95))
        weight = (scaled - 0.5) / 0.45
        stake = base * (0.75 + 0.5 * weight)
        return float(min(stake, self._settings.max_trade_size))

    def accept_direction(self, direction: str) -> bool:
        if direction != self._last_direction:
            self._last_direction = direction
            self._same_direction_count = 1
            return True
        self._same_direction_count += 1
        return self._same_direction_count <= self._settings.max_consecutive_same_direction

    def is_probability_acceptable(self, confidence: float) -> bool:
        return confidence >= self._settings.min_probability


class SignalEngine:
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
        prob_call, prob_put = self._score_probabilities(features)
        confidence = max(prob_call, prob_put)
        edge = confidence - 0.5

        logging.debug("Features: %s | prob_call=%.3f prob_put=%.3f", features, prob_call, prob_put)

        trade = self._config.trade
        if edge < trade.min_confidence_edge:
            return None

        if abs(features["momentum_short"]) < trade.min_momentum:
            return None

        direction = "call" if prob_call >= prob_put else "put"

        now = datetime.now(timezone.utc)
        expiry = now + timedelta(seconds=trade.expiry_seconds)
        entry_deadline = now + timedelta(seconds=trade.entry_window_seconds)

        point = float(getattr(symbol_info, "point", 0.0) or 0.0)
        bid = float(getattr(tick, "bid", 0.0) or 0.0)
        ask = float(getattr(tick, "ask", 0.0) or 0.0)
        if point <= 0 or bid <= 0 or ask <= 0:
            return None

        if direction == "call":
            entry_price = ask
            invalidation = entry_price - trade.entry_buffer_points * point
        else:
            entry_price = bid
            invalidation = entry_price + trade.entry_buffer_points * point

        notes = self._format_notes(features, direction)

        return SignalCandidate(
            direction=direction,
            confidence=confidence,
            entry_price=float(entry_price),
            reference_price=float(latest["close"]),
            spread_points=spread_points,
            expiry_time=expiry,
            entry_deadline=entry_deadline,
            invalidation_price=float(invalidation),
            momentum_short=float(features["momentum_short"]),
            notes=notes,
            timestamp=latest.name,
        )

    @staticmethod
    def _collect_features(latest: pd.Series, previous: pd.Series) -> Dict[str, float]:
        ema_fast_mid_gap = float(latest["ema_fast"] - latest["ema_mid"])
        ema_mid_slow_gap = float(latest["ema_mid"] - latest["ema_slow"])
        ema_fast_slope = float(latest["ema_fast_slope"])
        ema_mid_slope = float(latest["ema_mid_slope"])
        price_vs_ema_fast = float(latest["close"] - latest["ema_fast"])
        price_vs_boll_mid = float(latest["close"] - latest["boll_mid"])
        rsi_dev = float(latest["rsi"] - 50.0)
        stoch_diff = float(latest["stoch_k"] - latest["stoch_d"])
        macd_hist = float(latest["macd_hist"])
        momentum_short = float(latest["momentum_short"])
        momentum_long = float(latest["momentum_long"])
        volatility = float(latest["volatility"])
        atr = float(latest["atr"])
        body_relative = float(latest["body_relative"])
        upper_dist = float(latest["boll_upper"] - latest["close"])
        lower_dist = float(latest["close"] - latest["boll_lower"])

        return {
            "ema_fast_mid_gap": ema_fast_mid_gap,
            "ema_mid_slow_gap": ema_mid_slow_gap,
            "ema_fast_slope": ema_fast_slope,
            "ema_mid_slope": ema_mid_slope,
            "price_vs_ema_fast": price_vs_ema_fast,
            "price_vs_boll_mid": price_vs_boll_mid,
            "rsi_dev": rsi_dev,
            "stoch_diff": stoch_diff,
            "macd_hist": macd_hist,
            "momentum_short": momentum_short,
            "momentum_long": momentum_long,
            "volatility": volatility,
            "atr": atr,
            "body_relative": body_relative,
            "upper_dist": upper_dist,
            "lower_dist": lower_dist,
        }

    def _score_probabilities(self, f: Dict[str, float]) -> Tuple[float, float]:
        weight_call = {
            "ema_fast_mid_gap": 3.2,
            "ema_mid_slow_gap": 2.3,
            "ema_fast_slope": 1.8,
            "ema_mid_slope": 1.0,
            "price_vs_ema_fast": 2.1,
            "price_vs_boll_mid": 1.4,
            "rsi_dev": -0.05,
            "stoch_diff": 1.2,
            "macd_hist": 2.6,
            "momentum_short": 3.3,
            "momentum_long": 1.7,
            "volatility": -1.1,
            "body_relative": 0.9,
            "upper_dist": -0.4,
            "lower_dist": 0.7,
        }
        weight_put = {
            "ema_fast_mid_gap": -3.2,
            "ema_mid_slow_gap": -2.3,
            "ema_fast_slope": -1.8,
            "ema_mid_slope": -1.0,
            "price_vs_ema_fast": -2.1,
            "price_vs_boll_mid": -1.4,
            "rsi_dev": 0.05,
            "stoch_diff": -1.2,
            "macd_hist": -2.6,
            "momentum_short": -3.3,
            "momentum_long": -1.7,
            "volatility": -1.1,
            "body_relative": -0.9,
            "upper_dist": 0.7,
            "lower_dist": -0.4,
        }

        normalisers = {
            "ema_fast_mid_gap": 0.0015,
            "ema_mid_slow_gap": 0.0015,
            "ema_fast_slope": 0.0008,
            "ema_mid_slope": 0.0006,
            "price_vs_ema_fast": 0.0012,
            "price_vs_boll_mid": 0.0015,
            "rsi_dev": 20.0,
            "stoch_diff": 20.0,
            "macd_hist": 0.0004,
            "momentum_short": 0.0004,
            "momentum_long": 0.0004,
            "volatility": 0.002,
            "body_relative": 0.7,
            "upper_dist": 0.001,
            "lower_dist": 0.001,
        }

        bias = 0.05

        def sigmoid(x: float) -> float:
            return 1 / (1 + math.exp(-x))

        call_score = bias
        put_score = bias
        for key, value in f.items():
            norm = value / normalisers.get(key, 1.0)
            call_score += weight_call.get(key, 0.0) * norm
            put_score += weight_put.get(key, 0.0) * norm

        return sigmoid(call_score), sigmoid(put_score)

    @staticmethod
    def _format_notes(features: Dict[str, float], direction: str) -> List[str]:
        return [
            f"EMA gap fast-mid {features['ema_fast_mid_gap']*1e4:+.2f} pts",
            f"Momentum short {features['momentum_short']:+.5f}",
            f"Momentum long {features['momentum_long']:+.5f}",
            f"RSI dev {features['rsi_dev']:+.2f}",
            f"Stoch diff {features['stoch_diff']:+.2f}",
            f"MACD hist {features['macd_hist']:+.5f}",
            f"Volatility {features['volatility']:.5f}",
        ]


# ---------------------------------------------------------------------------
# Terminal reporting
# ---------------------------------------------------------------------------


class SignalPrinter:
    def __init__(self) -> None:
        self._last_timestamp: Optional[pd.Timestamp] = None

    def emit(
        self,
        strategy: StrategyConfig,
        candidate: SignalCandidate,
        risk: RiskManager,
        bid: float,
        ask: float,
    ) -> None:
        if candidate.timestamp == self._last_timestamp:
            logging.debug("Signal already emitted for %s", candidate.timestamp)
            return

        if not risk.accept_direction(candidate.direction):
            logging.info("Skipping signal due to consecutive %s limit", candidate.direction.upper())
            return

        stake = risk.stake_size(candidate.confidence)
        if stake <= 0:
            logging.info("Calculated stake is zero; skipping")
            return

        direction_label = "CALL (expect price higher)" if candidate.direction == "call" else "PUT (expect price lower)"
        lines = [
            "=" * 80,
            f"{candidate.timestamp.isoformat()} | {strategy.symbol} | expiry {strategy.trade.expiry_seconds}s",
            f"SIGNAL: {direction_label} | confidence {candidate.confidence:.2%} | spread {candidate.spread_points:.1f} pts",
            f"ENTRY price {'≥' if candidate.direction == 'call' else '≤'} {candidate.entry_price:.5f} | bid {bid:.5f} | ask {ask:.5f}",
            f"STAKE ≈ {stake:.2f} (risk {strategy.risk.risk_per_trade:.2%} of balance {strategy.risk.account_balance:.2f})",
            f"INVALIDATE if price {'<' if candidate.direction == 'call' else '>'} {candidate.invalidation_price:.5f}",
            f"ENTER by {candidate.entry_deadline.strftime('%H:%M:%S')} UTC | EXPIRY {candidate.expiry_time.strftime('%H:%M:%S')} UTC",
            "NOTES:",
        ]
        lines.extend(f" - {note}" for note in candidate.notes)
        lines.append("=" * 80)
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
    active_config = config_manager.config
    data_client = MT5DataClient(active_config)
    indicator_engine = IndicatorEngine(active_config.indicators)
    risk_manager = RiskManager(active_config.risk)
    signal_engine = SignalEngine(active_config)
    printer = SignalPrinter()

    while True:
        if stop_flag and stop_flag():
            break

        try:
            new_config = config_manager.maybe_reload()
        except Exception as exc:
            logging.warning("Config reload failed: %s", exc)
            new_config = active_config

        if new_config != active_config:
            logging.info("Config change detected; rebuilding components")
            active_config = new_config
            data_client.refresh(active_config)
            indicator_engine = IndicatorEngine(active_config.indicators)
            signal_engine = SignalEngine(active_config)
            risk_manager = RiskManager(active_config.risk)

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
            logging.debug("Invalid tick data; skipping cycle")
            if once:
                break
            time.sleep(active_config.loop_seconds)
            continue

        spread_points = (ask - bid) / point
        if spread_points > active_config.trade.max_spread_points:
            logging.info(
                "Spread %.1f exceeds threshold %.1f; skipping",
                spread_points,
                active_config.trade.max_spread_points,
            )
        else:
            candidate = signal_engine.evaluate(enriched, tick, symbol_info, spread_points)
            if candidate and risk_manager.is_probability_acceptable(candidate.confidence):
                printer.emit(active_config, candidate, risk_manager, bid, ask)
            else:
                logging.info(
                    "No trade candidate (confidence %.2f < %.2f)",
                    0.0 if candidate is None else candidate.confidence,
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
        description="MetaTrader5 fixed-time trading signal generator (no auto-trading)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbol", default="EURUSD", help="Instrument symbol in MT5")
    parser.add_argument("--timeframe", default="1m", help="Base candle timeframe (e.g. 1m, 5m)")
    parser.add_argument("--history", type=int, default=600, help="Number of candles per fetch")
    parser.add_argument("--warmup", type=int, default=150, help="Minimum candles before scoring")
    parser.add_argument("--loop-seconds", type=int, default=30, help="Seconds between evaluations")
    parser.add_argument("--account", type=int, help="MT5 account number (optional)")
    parser.add_argument("--password", type=str, help="MT5 account password (optional)")
    parser.add_argument("--server", type=str, help="MT5 trade server name (optional)")
    parser.add_argument("--terminal-path", type=Path, help="Path to terminal64.exe (optional)")
    parser.add_argument("--account-balance", type=float, default=1000.0, help="Balance used for stake sizing")
    parser.add_argument("--risk-per-trade", type=float, default=0.02, help="Risk per trade as fraction")
    parser.add_argument("--max-trade-size", type=float, default=100.0, help="Maximum stake size")
    parser.add_argument("--min-probability", type=float, default=0.55, help="Minimum confidence to emit a signal")
    parser.add_argument("--max-spread", type=float, default=25.0, help="Maximum spread in points")
    parser.add_argument("--expiry-seconds", type=int, default=60, help="Fixed-time expiry in seconds")
    parser.add_argument("--entry-window", type=int, default=15, help="Seconds allowed to enter after signal")
    parser.add_argument("--config-path", type=Path, help="Optional JSON/YAML config file")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--once", action="store_true", help="Run a single evaluation and exit")
    return parser.parse_args(list(argv) if argv is not None else None)


def build_config_from_args(args: argparse.Namespace) -> StrategyConfig:
    base = StrategyConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        history_candles=args.history,
        warmup_candles=args.warmup,
        loop_seconds=args.loop_seconds,
        account=args.account,
        password=args.password,
        server=args.server,
        terminal_path=args.terminal_path,
        config_path=args.config_path,
    )

    risk = replace(
        base.risk,
        account_balance=args.account_balance,
        risk_per_trade=args.risk_per_trade,
        max_trade_size=args.max_trade_size,
        min_probability=args.min_probability,
    )

    trade = replace(
        base.trade,
        expiry_seconds=args.expiry_seconds,
        entry_window_seconds=args.entry_window,
        max_spread_points=args.max_spread,
    )

    return replace(base, risk=risk, trade=trade)


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
    config = build_config_from_args(args)
    config_manager = ConfigManager(config)

    if args.config_path and not args.config_path.exists():
        logging.warning("Config file %s does not exist; continuing with CLI parameters", args.config_path)

    try:
        initialize_mt5(config_manager.config)
    except Exception as exc:
        logging.exception("Failed to initialise MetaTrader 5: %s", exc)
        return 1

    stop_requested = False

    def _signal_handler(signum: int, _: Any) -> None:
        nonlocal stop_requested
        logging.info("Signal %s received; preparing to shut down", signum)
        stop_requested = True

    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_handler)

    try:
        run_strategy(config_manager, once=args.once, stop_flag=lambda: stop_requested)
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        shutdown_mt5()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
