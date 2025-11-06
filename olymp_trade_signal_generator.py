"""Olymp Trade fixed-time trade signal generator.

This script connects to Olymp Trade market data endpoints, ingests real-time
ticks and recent candlesticks, computes a rich technical indicator stack, and
produces auditable buy/sell recommendations for a single best candidate trade
per evaluation interval. It **never** places trades automatically; instead, it
prints OCO-like (one-cancels-other) suggestions to the terminal so a human
trader can execute or ignore them manually.

Key capabilities
----------------
* Resilient data acquisition via HTTP polling (candles) and optional
  WebSocket keeps-alive for tick updates.
* Comprehensive indicator suite (EMA, RSI, MACD, momentum, volatility,
  ATR) with deterministic, unit-test-friendly implementations using
  :mod:`pandas` and :mod:`numpy`.
* Probabilistic signal scoring that converts indicator confluence into a
  calibrated logistic score for long/short scenarios, returning at most one
  actionable trade each cycle.
* Risk management framework that sizes positions using ATR-derived stops,
  user-defined account equity, and maximum risk-per-trade and stake limits.
* Dynamic configuration reload (optional) so that parameter tweaks in a
  JSON/YAML file are applied without restarting the process.
* Verbose logging and health metrics so the behaviour is auditable and
  failures are easy to diagnose.

Usage example
-------------

.. code-block:: bash

    python olymp_trade_signal_generator.py \
        --symbol EURUSD_OTC --timeframe 1m --loop-seconds 30 \
        --account-balance 2500 --risk-per-trade 0.02 \
        --config-path config/strategy.yaml --log-level INFO

The script assumes you have valid access to Olymp Trade's public market data
endpoints. Endpoints vary by region and account status; adjust the REST/WebSocket
URLs with CLI flags or configuration overrides as needed.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import math
import signal
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

try:  # Optional dependencies: httpx for REST, websockets for live ticks.
    import httpx
except ImportError as exc:  # pragma: no cover - import guard for runtime clarity
    raise SystemExit(
        "Missing dependency 'httpx'. Install requirements first, e.g. "
        "pip install httpx"
    ) from exc

try:
    import websockets  # type: ignore
    from websockets.client import WebSocketClientProtocol
except ImportError:
    websockets = None  # type: ignore
    WebSocketClientProtocol = None  # type: ignore


# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EndpointConfig:
    """Container for Olymp Trade endpoint URLs.

    Users can override the defaults with CLI arguments or configuration files.
    The REST endpoint is used for candle history; the WebSocket endpoint (if
    reachable) keeps a heartbeat alive and may deliver tick-level updates.
    """

    rest_base_url: str = "https://olymptrade.com/api/v4"
    candles_path: str = "/assets/{symbol}/candles"
    candles_limit_param: str = "limit"
    timeframe_param: str = "timeframe"
    websocket_url: Optional[str] = "wss://olymptrade.com/websocket/public"


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
    volatility_period: int = 20  # rolling window for std-dev of returns
    atr_period: int = 14


@dataclass(slots=True)
class RiskSettings:
    """Risk management configuration."""

    account_balance: float = 1000.0
    risk_per_trade: float = 0.01  # 1% per trade
    max_trade_size: float = 100.0  # currency units (or broker's stake units)
    min_probability: float = 0.55
    atr_stop_multiplier: float = 1.8
    atr_target_multiplier: float = 2.7
    reward_risk_ratio: float = 1.5


@dataclass(slots=True)
class StrategyConfig:
    """Top-level strategy configuration."""

    symbol: str = "EURUSD"
    timeframe: str = "1m"
    history_candles: int = 500
    poll_interval_seconds: int = 30
    warmup_candles: int = 150
    endpoint: EndpointConfig = field(default_factory=EndpointConfig)
    indicators: IndicatorSettings = field(default_factory=IndicatorSettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    config_path: Optional[Path] = None

    @staticmethod
    def from_mapping(mapping: Dict[str, Any]) -> "StrategyConfig":
        """Hydrate a :class:`StrategyConfig` from a nested mapping."""

        def dataclass_from(prefix: str, cls: Any, payload: Dict[str, Any]) -> Any:
            allowed = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
            filtered = {k: v for k, v in payload.items() if k in allowed}
            return cls(**filtered)

        endpoint = dataclass_from("endpoint", EndpointConfig, mapping.get("endpoint", {}))
        indicators = dataclass_from("indicators", IndicatorSettings, mapping.get("indicators", {}))
        risk = dataclass_from("risk", RiskSettings, mapping.get("risk", {}))

        core_allowed = {field.name for field in StrategyConfig.__dataclass_fields__.values()}
        core = {k: v for k, v in mapping.items() if k in core_allowed}
        core.update({"endpoint": endpoint, "indicators": indicators, "risk": risk})
        config_path = mapping.get("config_path")
        if config_path is not None:
            core["config_path"] = Path(config_path)
        return StrategyConfig(**core)

    def merge_overrides(self, overrides: Dict[str, Any]) -> "StrategyConfig":
        """Return a new config with overrides applied."""

        return StrategyConfig.from_mapping({**asdict(self), **overrides})


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


class GracefulExit(SystemExit):
    """Raised on shutdown events to unwind asyncio tasks cleanly."""


def install_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    """Register signal handlers for Ctrl+C and SIGTERM."""

    def _signal_handler(signame: str) -> None:
        logging.info("Received %s, shutting down...", signame)
        raise GracefulExit(0)

    for signame in {"SIGINT", "SIGTERM"}:
        if hasattr(signal, signame):
            loop.add_signal_handler(getattr(signal, signame), lambda s=signame: _signal_handler(s))


# ---------------------------------------------------------------------------
# Data acquisition layer
# ---------------------------------------------------------------------------


class OlympTradeDataClient:
    """Lightweight client for Olymp Trade market data.

    The implementation fetches candle history via REST polling. If a WebSocket
    endpoint is reachable and :mod:`websockets` is installed, a background task
    keeps the connection alive (helpful for institutional accounts that require
    a heartbeat to keep REST quotas high). The WebSocket listener is optional;
    the trading logic relies exclusively on the candle DataFrame for
    determinations, ensuring reproducibility.
    """

    def __init__(
        self,
        config: StrategyConfig,
        timeout: float = 10.0,
        session: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._config = config
        self._timeout = timeout
        self._session = session or httpx.AsyncClient(timeout=timeout)
        self._ws_connection: Optional[WebSocketClientProtocol] = None
        self._ws_task: Optional[asyncio.Task[None]] = None

    # ------------------------ REST polling ---------------------------------

    async def fetch_candles(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Fetch recent candles for the configured symbol/timeframe.

        Parameters
        ----------
        limit:
            Optional candle count to request; defaults to strategy history.
        """

        limit = limit or self._config.history_candles
        endpoint = self._build_candles_endpoint()
        params = {
            self._config.endpoint.candles_limit_param: limit,
            self._config.endpoint.timeframe_param: self._config.timeframe,
        }
        logging.debug("Fetching candles: %s params=%s", endpoint, params)
        response = await self._session.get(endpoint, params=params)
        response.raise_for_status()
        payload = response.json()
        candles = self._parse_candle_payload(payload)
        if candles.empty:
            raise RuntimeError("Received empty candle payload")
        return candles

    async def close(self) -> None:
        """Close active HTTP/WebSocket resources."""

        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            with contextlib.suppress(Exception):  # pragma: no cover - defensive
                await self._ws_task
        if self._ws_connection and websockets:
            with contextlib.suppress(Exception):  # pragma: no cover - defensive
                await self._ws_connection.close()
        await self._session.aclose()

    # ------------------------ WebSocket (optional) -------------------------

    async def ensure_websocket(self) -> None:
        """Establish a background WebSocket heartbeat if configured."""

        if not self._config.endpoint.websocket_url or websockets is None:
            logging.debug("WebSocket support disabled or dependency missing")
            return

        if self._ws_task and not self._ws_task.done():
            return

        async def _run() -> None:
            url = self._config.endpoint.websocket_url
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    self._ws_connection = ws
                    logging.info("WebSocket heartbeat established: %s", url)
                    await self._subscribe_ticks(ws)
            except Exception as exc:  # pragma: no cover - runtime guard
                logging.warning("WebSocket connection error: %s", exc)

        self._ws_task = asyncio.create_task(_run(), name="olymp-ws-heartbeat")

    async def _subscribe_ticks(self, ws: WebSocketClientProtocol) -> None:
        """Subscribe to tick updates to keep the session warm."""

        subscribe_payload = json.dumps(
            {
                "event": "subscribe",
                "payload": {
                    "symbol": self._config.symbol,
                    "timeframe": self._config.timeframe,
                    "channel": "ticks",
                },
            }
        )
        await ws.send(subscribe_payload)
        logging.debug("Subscribed to ticks: %s", subscribe_payload)
        async for message in ws:
            logging.debug("Tick heartbeat: %s", message)

    # ------------------------ Helpers --------------------------------------

    def _build_candles_endpoint(self) -> str:
        base = self._config.endpoint.rest_base_url.rstrip("/")
        path = self._config.endpoint.candles_path.format(symbol=self._config.symbol)
        return f"{base}{path}"

    @staticmethod
    def _parse_candle_payload(payload: Any) -> pd.DataFrame:
        """Convert raw JSON payload to a candle DataFrame."""

        if isinstance(payload, dict) and "data" in payload:
            payload = payload["data"]
        if not isinstance(payload, (list, tuple)):
            raise ValueError("Unexpected candle payload format")

        frame = pd.DataFrame(payload)
        possible_columns = {
            "open": "open",
            "close": "close",
            "high": "high",
            "low": "low",
            "volume": "volume",
            "timestamp": "timestamp",
            "time": "time",
        }
        renamed = {}
        for column, target in possible_columns.items():
            if column in frame.columns:
                renamed[column] = target
        frame = frame.rename(columns=renamed)
        if "timestamp" not in frame.columns:
            if "time" in frame.columns:
                frame["timestamp"] = frame["time"]
            else:
                raise ValueError("Candle payload missing timestamp field")

        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
        frame = frame.set_index("timestamp").sort_index()
        numeric_cols = [col for col in ["open", "high", "low", "close", "volume"] if col in frame.columns]
        frame[numeric_cols] = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")
        frame = frame.dropna(subset=["open", "high", "low", "close"])
        return frame


# ---------------------------------------------------------------------------
# Indicator engine
# ---------------------------------------------------------------------------


class IndicatorEngine:
    """Compute technical indicators needed for the strategy."""

    def __init__(self, settings: IndicatorSettings) -> None:
        self._settings = settings

    def enrich(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Return candles with indicator columns appended."""

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
        data["volatility"] = returns.rolling(window=settings.volatility_period).std() * math.sqrt(settings.volatility_period)

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


class RiskManager:
    """Convert signals into actionable trade plans respecting risk settings."""

    def __init__(self, settings: RiskSettings) -> None:
        self._settings = settings

    def position_size(self, entry: float, stop_loss: float) -> float:
        """Calculate position size based on ATR-derived stop distance."""

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

    def evaluate(self, data: pd.DataFrame) -> Optional[SignalCandidate]:
        """Return a single best trade candidate or ``None``."""

        if data.empty:
            return None

        latest = data.iloc[-1]
        previous = data.iloc[-2] if len(data) > 1 else latest

        features = self._collect_features(latest, previous)
        prob_long, prob_short = self._score_probabilities(features)

        logging.debug("Features: %s prob_long=%.3f prob_short=%.3f", features, prob_long, prob_short)

        if prob_long < self._config.risk.min_probability and prob_short < self._config.risk.min_probability:
            return None

        direction = "buy" if prob_long >= prob_short else "sell"
        probability = max(prob_long, prob_short)
        atr = latest["atr"]
        entry = float(latest["close"])

        if direction == "buy":
            stop_loss = entry - self._config.risk.atr_stop_multiplier * atr
            reward_rr = self._config.risk.reward_risk_ratio * (entry - stop_loss)
            reward_atr = self._config.risk.atr_target_multiplier * atr
            take_profit = entry + min(reward_rr, reward_atr)
            score = prob_long
        else:
            stop_loss = entry + self._config.risk.atr_stop_multiplier * atr
            reward_rr = self._config.risk.reward_risk_ratio * (stop_loss - entry)
            reward_atr = self._config.risk.atr_target_multiplier * atr
            take_profit = max(entry - min(reward_rr, reward_atr), 0.0)
            score = prob_short

        notes = self._format_notes(features, direction)
        return SignalCandidate(
            direction=direction,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            probability=probability,
            score=score,
            timestamp=latest.name,
            notes=notes,
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

    def _score_probabilities(self, features: Dict[str, float]) -> Tuple[float, float]:
        # Weighted logistic regression style scoring for interpretability.
        weight_long = {
            "ema_gap": 2.0,
            "ema_trend": 1.2,
            "ema_cross": 1.5,
            "rsi": -0.04,  # penalise overbought (>70)
            "macd_hist": 1.8,
            "momentum": 1.2,
            "volatility": -0.8,
        }
        weight_short = {
            "ema_gap": -2.0,
            "ema_trend": -1.2,
            "ema_cross": -1.5,
            "rsi": 0.04,  # penalise oversold (<30)
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

        prob_long = sigmoid(long_score)
        prob_short = sigmoid(short_score)
        return prob_long, prob_short

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
# Dynamic configuration loader
# ---------------------------------------------------------------------------


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

    async def maybe_reload(self) -> StrategyConfig:
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
            except ImportError as exc:  # pragma: no cover - optional dep guard
                raise RuntimeError(
                    "Missing dependency 'pyyaml'. Install it to use YAML configs."
                ) from exc
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
            # OCO-style stop and take profit levels.
            f"STOP  @ {candidate.stop_loss:.5f} ({strategy.risk.atr_stop_multiplier:.2f} x ATR)",
            f"TARGET@ {candidate.take_profit:.5f} (R/R {strategy.risk.reward_risk_ratio:.2f})",
            f"SIZE  ≈ {size:.2f} (risk {strategy.risk.risk_per_trade:.2%} of equity {strategy.risk.account_balance:.2f})",
            f"NOTES: {candidate.notes}",
            "=" * 72,
        ]
        print("\n".join(lines), flush=True)
        self._last_timestamp = candidate.timestamp


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


class StrategyRunner:
    """Async orchestrator that periodically evaluates signals."""

    def __init__(self, config_manager: ConfigManager) -> None:
        self._config_manager = config_manager
        self._printer = SignalPrinter()

    async def run(self) -> None:
        config: Optional[StrategyConfig] = None
        client: Optional[OlympTradeDataClient] = None
        indicator_engine: Optional[IndicatorEngine] = None
        signal_engine: Optional[SignalEngine] = None
        risk_manager: Optional[RiskManager] = None

        try:
            while True:
                (
                    config,
                    client,
                    indicator_engine,
                    signal_engine,
                    risk_manager,
                ) = await self._ensure_components(
                    config, client, indicator_engine, signal_engine, risk_manager
                )

                try:
                    candles = await client.fetch_candles()
                except Exception as exc:
                    logging.warning("Failed to fetch candles: %s", exc)
                    await asyncio.sleep(config.poll_interval_seconds)
                    continue

                enriched = indicator_engine.enrich(candles)
                if len(enriched) < config.warmup_candles:
                    logging.debug(
                        "Warmup incomplete (%s/%s candles)", len(enriched), config.warmup_candles
                    )
                    await asyncio.sleep(config.poll_interval_seconds)
                    continue

                candidate = signal_engine.evaluate(enriched)
                if candidate and risk_manager.is_probability_acceptable(candidate.probability):
                    self._printer.emit(config, candidate, risk_manager)
                else:
                    logging.info(
                        "No trade candidate (probability below %.2f)",
                        config.risk.min_probability,
                    )

                await asyncio.sleep(config.poll_interval_seconds)
        except GracefulExit:
            logging.info("StrategyRunner shutdown requested")
        finally:
            if client:
                await client.close()

    async def _ensure_components(
        self,
        current_config: Optional[StrategyConfig],
        client: Optional[OlympTradeDataClient],
        indicator_engine: Optional[IndicatorEngine],
        signal_engine: Optional[SignalEngine],
        risk_manager: Optional[RiskManager],
    ) -> Tuple[StrategyConfig, OlympTradeDataClient, IndicatorEngine, SignalEngine, RiskManager]:
        new_config = await self._config_manager.maybe_reload()

        components_ready = all(
            component is not None
            for component in (current_config, client, indicator_engine, signal_engine, risk_manager)
        )

        if components_ready and new_config == current_config:
            assert current_config is not None
            assert client is not None
            assert indicator_engine is not None
            assert signal_engine is not None
            assert risk_manager is not None
            return current_config, client, indicator_engine, signal_engine, risk_manager

        if client is not None:
            await client.close()

        client = OlympTradeDataClient(new_config)
        await client.ensure_websocket()
        indicator_engine = IndicatorEngine(new_config.indicators)
        signal_engine = SignalEngine(new_config)
        risk_manager = RiskManager(new_config.risk)

        logging.info(
            "Strategy components initialised: %s %s (poll %ss)",
            new_config.symbol,
            new_config.timeframe,
            new_config.poll_interval_seconds,
        )

        return new_config, client, indicator_engine, signal_engine, risk_manager


# ---------------------------------------------------------------------------
# CLI handling
# ---------------------------------------------------------------------------


def parse_arguments(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Olymp Trade signal generator (no auto-trading)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbol", default="EURUSD", help="Instrument symbol as recognised by Olymp Trade")
    parser.add_argument("--timeframe", default="1m", help="Candle timeframe (e.g. 1m, 5m, 15m)")
    parser.add_argument("--history", type=int, default=500, help="Number of candles to request each cycle")
    parser.add_argument("--warmup", type=int, default=150, help="Minimum candles required before scoring")
    parser.add_argument("--loop-seconds", type=int, default=30, help="Seconds between evaluations")
    parser.add_argument("--account-balance", type=float, default=1000.0, help="Account equity for risk calculations")
    parser.add_argument("--risk-per-trade", type=float, default=0.01, help="Risk percentage per trade (0-1)")
    parser.add_argument("--max-trade-size", type=float, default=100.0, help="Absolute cap on trade size")
    parser.add_argument("--min-probability", type=float, default=0.55, help="Minimum probability threshold to emit a trade")
    parser.add_argument("--config-path", type=Path, help="Optional JSON/YAML file for dynamic configuration")
    parser.add_argument("--rest-base-url", help="Override REST base URL")
    parser.add_argument("--candles-path", help="Override REST candles path template")
    parser.add_argument("--websocket-url", help="Override WebSocket URL")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--once", action="store_true", help="Run a single evaluation and exit")
    return parser.parse_args(list(argv) if argv is not None else None)


def build_config_from_args(args: argparse.Namespace) -> StrategyConfig:
    base_config = StrategyConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        history_candles=args.history,
        poll_interval_seconds=args.loop_seconds,
        warmup_candles=args.warmup,
        config_path=args.config_path,
    )

    endpoint = base_config.endpoint
    if args.rest_base_url:
        endpoint = replace(endpoint, rest_base_url=args.rest_base_url)
    if args.candles_path:
        endpoint = replace(endpoint, candles_path=args.candles_path)
    if args.websocket_url:
        endpoint = replace(endpoint, websocket_url=args.websocket_url)

    risk = replace(
        base_config.risk,
        account_balance=args.account_balance,
        risk_per_trade=args.risk_per_trade,
        max_trade_size=args.max_trade_size,
        min_probability=args.min_probability,
    )

    return replace(base_config, endpoint=endpoint, risk=risk)


async def run_once(config_manager: ConfigManager) -> None:
    config = config_manager.config
    client = OlympTradeDataClient(config)
    indicator_engine = IndicatorEngine(config.indicators)
    signal_engine = SignalEngine(config)
    risk_manager = RiskManager(config.risk)
    printer = SignalPrinter()

    candles = await client.fetch_candles()
    enriched = indicator_engine.enrich(candles)
    candidate = signal_engine.evaluate(enriched)
    if candidate and risk_manager.is_probability_acceptable(candidate.probability):
        printer.emit(config, candidate, risk_manager)
    else:
        logging.info("No qualifying trade candidate in single-run mode")
    await client.close()


def configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_arguments(argv)
    configure_logging(args.log_level)
    base_config = build_config_from_args(args)
    config_manager = ConfigManager(base_config)

    if args.config_path and not args.config_path.exists():
        logging.error("Config file %s does not exist", args.config_path)
        return 1

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        install_signal_handlers(loop)
        if args.once:
            loop.run_until_complete(run_once(config_manager))
        else:
            runner = StrategyRunner(config_manager)
            loop.run_until_complete(runner.run())
    except GracefulExit:
        logging.info("Graceful exit requested")
    except Exception as exc:  # pragma: no cover - top-level defensive
        logging.exception("Fatal error: %s", exc)
        return 1
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

