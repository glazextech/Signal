"""Binance spot market research and OCO signal assistant.

This script inspects every USDT spot pair on Binance, evaluates them with a
layered indicator stack, and proposes a best-effort OCO order idea based on
the strongest probabilistic edge it can detect.

Design goals
------------
 * Fetch fresh ticker, order book, and kline data via the public REST API.
 * Filter symbols only for liquidity so even high-priced majors remain in
   scope, while still avoiding untradeable books.
* Compute layered technical signals (EMA, RSI, MACD, momentum, volatility)
  and convert them into a probabilistic confidence score for a >=5% move.
* Respect configurable risk limits so the suggested position size never
  exceeds the user's comfort zone.
* Provide verbose logging that documents every decision and intermediate
  calculation, making the analysis auditable and easy to iterate on.

Usage
-----
    python binance_bot.py --budget 250 --max-risk 0.12 --verbose

Dependencies
------------
    pip install requests pandas numpy

IMPORTANT: Binance applies rate limits. This script paces order book
requests, but for production you may need API keys, caching, or backoff.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------


@dataclass
class BudgetConfig:
    """Describe trader capital and risk appetite for position sizing."""

    total_budget: float = 200.0
    max_risk_per_trade: float = 0.10  # allocate at most 10% of capital per idea
    quote_asset: str = "USDT"  # concentrate on liquid USD-quoted markets


@dataclass
class IndicatorParams:
    """Collect indicator lookback windows for easy experimentation."""

    ema_fast: int = 12
    ema_slow: int = 26
    ema_signal: int = 9
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    momentum_period: int = 10
    atr_period: int = 14
    volatility_period: int = 20
    trend_strength_threshold: float = 0.003  # tune after observing live output


@dataclass
class RiskParams:
    """Liquidity and execution safety thresholds."""

    min_quote_volume: float = 1_000_000.0  # 24h quote volume filter
    min_order_book_notional: float = 10_000.0  # depth check to avoid illiquid books
    order_book_limit: int = 20
    liquidity_depth_levels: int = 5
    min_profit_target: float = 0.05  # ensure >=5% upside target
    stop_loss_atr_multiplier: float = 1.5
    take_profit_atr_multiplier: float = 3.0
    max_candidates: Optional[int] = None  # analyze all unless user caps it


@dataclass
class MarketSnapshot:
    """Bundle essential market data for one symbol."""

    symbol: str
    last_price: float
    volume_quote: float
    volume_base: float
    best_bid: float
    best_ask: float
    bids: List[Tuple[float, float]]
    asks: List[Tuple[float, float]]


@dataclass
class AnalysisResult:
    """Hold indicator outputs and the final probability estimate."""

    symbol: str
    buy_price: float
    take_profit: float
    stop_loss: float
    probability: float
    position_notional: float
    quantity: float
    reasoning: str
    metrics: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


BINANCE_API = "https://api.binance.com"


def _make_session(timeout: float = 10.0) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "CursorBot/1.0"})
    session.request = _wrap_with_timeout(session.request, timeout)
    return session


def _wrap_with_timeout(func, timeout: float):  # type: ignore[override]
    def _wrapped(method: str, url: str, **kwargs):
        kwargs.setdefault("timeout", timeout)
        return func(method, url, **kwargs)

    return _wrapped


def fetch_ticker_stats(session: requests.Session, quote_asset: str) -> List[Dict[str, str]]:
    """Fetch 24h stats once; reuse for price + volume based filtering."""

    logging.debug("Requesting 24h ticker stats from Binance")
    response = session.get(f"{BINANCE_API}/api/v3/ticker/24hr")
    response.raise_for_status()
    stats = response.json()
    logging.info("Fetched %d ticker entries", len(stats))
    filtered = [s for s in stats if s["symbol"].endswith(quote_asset)]
    logging.debug("Filtered %d %s pairs", len(filtered), quote_asset)
    return filtered


def fetch_order_book(
    session: requests.Session,
    symbol: str,
    depth: int,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Get current order book for a symbol."""

    logging.debug("Fetching order book for %s", symbol)
    response = session.get(
        f"{BINANCE_API}/api/v3/depth",
        params={"symbol": symbol, "limit": depth},
    )
    response.raise_for_status()
    payload = response.json()
    bids = [(float(price), float(qty)) for price, qty in payload.get("bids", [])]
    asks = [(float(price), float(qty)) for price, qty in payload.get("asks", [])]
    return bids, asks


def fetch_klines(
    session: requests.Session,
    symbol: str,
    interval: str = "5m",
    limit: int = 500,
) -> pd.DataFrame:
    """Pull kline history for deeper technical analysis."""

    logging.debug("Fetching %s klines for %s", interval, symbol)
    response = session.get(
        f"{BINANCE_API}/api/v3/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
    )
    response.raise_for_status()
    raw = response.json()
    df = pd.DataFrame(
        raw,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )
    if df.empty:
        raise ValueError(f"Empty kline frame for {symbol}")

    df[["open", "high", "low", "close", "volume"]] = df[[
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    return df


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------


def fetch_data(
    session: requests.Session,
    budget: BudgetConfig,
    risk: RiskParams,
) -> List[MarketSnapshot]:
    """Collect market snapshots for liquid symbols that pass exchange filters."""

    stats = fetch_ticker_stats(session, budget.quote_asset)

    eligible: List[Dict[str, str]] = []
    for entry in stats:
        try:
            price = float(entry["lastPrice"])
            quote_volume = float(entry["quoteVolume"])
            base_volume = float(entry["volume"])
        except (KeyError, ValueError) as exc:
            logging.debug("Skipping %s due to parsing error: %s", entry.get("symbol"), exc)
            continue

        symbol = entry["symbol"]

        if price <= 0:
            logging.debug("Skipping %s due to non-positive price", symbol)
            continue

        if quote_volume < risk.min_quote_volume:
            logging.debug(
                "Skipping %s due to low 24h quote volume: %.2f < %.2f",
                symbol,
                quote_volume,
                risk.min_quote_volume,
            )
            continue

        eligible.append(entry)

    eligible = sorted(
        eligible,
        key=lambda item: float(item["quoteVolume"]),
        reverse=True,
    )

    total_candidates = len(eligible)
    if risk.max_candidates is not None and total_candidates > risk.max_candidates:
        logging.info(
            "Candidate list capped from %d to %d by max_candidates",
            total_candidates,
            risk.max_candidates,
        )
        eligible = eligible[: risk.max_candidates]

    logging.info("Eligible liquid shortlist: %d symbols", len(eligible))

    snapshots: List[MarketSnapshot] = []
    for entry in eligible:
        symbol = entry["symbol"]

        try:
            bids, asks = fetch_order_book(session, symbol, risk.order_book_limit)
        except requests.HTTPError as exc:
            logging.warning("Order book request failed for %s: %s", symbol, exc)
            continue

        if not bids or not asks:
            logging.debug("Skipping %s due to empty order book", symbol)
            continue

        depth_notional = _order_book_depth_value(bids, asks, risk.liquidity_depth_levels)
        if depth_notional < risk.min_order_book_notional:
            logging.debug(
                "Skipping %s; depth notional %.2f below threshold %.2f",
                symbol,
                depth_notional,
                risk.min_order_book_notional,
            )
            continue

        last_price = float(entry["lastPrice"])
        volume_quote = float(entry["quoteVolume"])
        volume_base = float(entry["volume"])

        snapshot = MarketSnapshot(
            symbol=symbol,
            last_price=last_price,
            volume_quote=volume_quote,
            volume_base=volume_base,
            best_bid=bids[0][0],
            best_ask=asks[0][0],
            bids=bids,
            asks=asks,
        )

        snapshots.append(snapshot)

        # NOTE: If this loop grows large we may need adaptive sleeping to respect Binance rate limits.
        time.sleep(0.1)  # friendly pacing; tune for production throughput

    logging.info("Prepared %d market snapshots after depth filter", len(snapshots))
    return snapshots


def _order_book_depth_value(
    bids: Iterable[Tuple[float, float]],
    asks: Iterable[Tuple[float, float]],
    levels: int,
) -> float:
    """Estimate combined notional available near top-of-book."""

    total = 0.0
    for side in (bids, asks):
        for idx, (price, qty) in enumerate(side):
            if idx >= levels:
                break
            total += price * qty
    return total


# ---------------------------------------------------------------------------
# Technical analysis engine
# ---------------------------------------------------------------------------


def analyze_coin(
    session: requests.Session,
    snapshot: MarketSnapshot,
    budget: BudgetConfig,
    indicators: IndicatorParams,
    risk: RiskParams,
) -> Optional[AnalysisResult]:
    """Produce indicator metrics and an OCO suggestion for a symbol."""

    try:
        df = fetch_klines(session, snapshot.symbol)
    except (requests.HTTPError, ValueError) as exc:
        logging.warning("Skipping %s: failed to load klines (%s)", snapshot.symbol, exc)
        return None

    enriched = _compute_indicators(df, indicators)
    latest = enriched.iloc[-1]
    prev = enriched.iloc[-2]

    buy_price = snapshot.best_ask

    atr = latest["atr"]
    stop_loss = max(buy_price - risk.stop_loss_atr_multiplier * atr, 0.0)
    take_profit_raw = buy_price + risk.take_profit_atr_multiplier * atr
    take_profit = max(take_profit_raw, buy_price * (1 + risk.min_profit_target))

    metrics = {
        "ema_fast": latest["ema_fast"],
        "ema_slow": latest["ema_slow"],
        "ema_signal": latest["ema_signal"],
        "rsi": latest["rsi"],
        "macd": latest["macd"],
        "macd_signal": latest["macd_signal"],
        "macd_hist": latest["macd_hist"],
        "momentum": latest["momentum"],
        "trend_strength": latest["trend_strength"],
        "volatility": latest["volatility"],
        "atr": atr,
    }

    probability, reasoning = _score_probability(latest, prev, indicators)

    if probability < 0.0:
        logging.debug(
            "Probability guard triggered for %s; computed %.2f",
            snapshot.symbol,
            probability,
        )
        return None

    max_trade_notional = budget.total_budget * budget.max_risk_per_trade
    position_notional = min(max_trade_notional, budget.total_budget)
    quantity = position_notional / buy_price if buy_price else 0.0

    # NOTE: Double-check exchange minimum quantities before live deployment.
    if quantity * buy_price < 5.0:
        logging.debug(
            "Skipping %s because position size %.2f is too small for meaningful trading",
            snapshot.symbol,
            quantity * buy_price,
        )
        return None

    result = AnalysisResult(
        symbol=snapshot.symbol,
        buy_price=buy_price,
        take_profit=take_profit,
        stop_loss=stop_loss,
        probability=probability,
        position_notional=position_notional,
        quantity=quantity,
        reasoning=reasoning,
        metrics=metrics,
    )

    logging.debug("Analysis result for %s: %s", snapshot.symbol, result)
    return result


def _compute_indicators(df: pd.DataFrame, params: IndicatorParams) -> pd.DataFrame:
    """Append indicator columns; keep logic vectorized for speed."""

    prices = df.copy()

    prices["ema_fast"] = prices["close"].ewm(span=params.ema_fast, adjust=False).mean()
    prices["ema_slow"] = prices["close"].ewm(span=params.ema_slow, adjust=False).mean()
    prices["ema_signal"] = prices["close"].ewm(span=params.ema_signal, adjust=False).mean()

    delta = prices["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / params.rsi_period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / params.rsi_period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    prices["rsi"] = 100 - (100 / (1 + rs))

    ema_fast = prices["close"].ewm(span=params.macd_fast, adjust=False).mean()
    ema_slow = prices["close"].ewm(span=params.macd_slow, adjust=False).mean()
    prices["macd"] = ema_fast - ema_slow
    prices["macd_signal"] = prices["macd"].ewm(span=params.macd_signal, adjust=False).mean()
    prices["macd_hist"] = prices["macd"] - prices["macd_signal"]

    prices["momentum"] = prices["close"] - prices["close"].shift(params.momentum_period)

    tr = np.maximum(
        prices["high"] - prices["low"],
        np.maximum(
            prices["high"] - prices["close"].shift(1),
            prices["close"].shift(1) - prices["low"],
        ),
    )
    prices["atr"] = tr.rolling(window=params.atr_period, min_periods=1).mean()

    prices["volatility"] = (
        prices["close"].pct_change().rolling(window=params.volatility_period).std()
    ).fillna(0)

    prices["trend_strength"] = (
        (prices["ema_fast"] - prices["ema_slow"]).abs() / prices["close"]
    ).fillna(0)

    return prices.dropna().copy()


def _score_probability(
    latest: pd.Series,
    previous: pd.Series,
    params: IndicatorParams,
) -> Tuple[float, str]:
    """Translate indicator relationships into a coarse probability metric."""

    score = 0.0
    score_components: List[str] = []
    weight_total = 0.0

    def add_score(condition: bool, weight: float, note: str) -> None:
        nonlocal score, weight_total
        weight_total += weight
        if condition:
            score += weight
            score_components.append(f"+ {note}")
        else:
            score_components.append(f"- {note}")

    add_score(latest["ema_fast"] > latest["ema_slow"], 1.0, "EMA bullish alignment")
    add_score(latest["macd"] > latest["macd_signal"], 1.0, "MACD above signal")
    add_score(latest["macd_hist"] > previous["macd_hist"], 0.5, "MACD histogram rising")
    add_score(45 < latest["rsi"] < 65, 0.5, "RSI mid-zone momentum")
    add_score(latest["momentum"] > 0, 0.5, "Positive momentum")
    add_score(
        latest["trend_strength"] > params.trend_strength_threshold,
        0.5,
        "Trend strength above threshold",
    )
    add_score(latest["volatility"] < 0.05, 0.5, "Volatility contained (<5%)")

    if weight_total == 0:
        return 0.0, "No scoring weights applied"

    normalized = score / weight_total
    probability = max(0.0, min(0.95, 0.40 + normalized * 0.45))

    reasoning = ", ".join(score_components)
    return probability, reasoning


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def generate_oco_signal(result: AnalysisResult) -> str:
    """Format an OCO suggestion for terminal display."""

    prob_percent = result.probability * 100
    lines = [
        f"Symbol: {result.symbol}",
        f"Buy price (limit leg): {result.buy_price:.6f}",
        f"Take-profit (OCO limit): {result.take_profit:.6f}",
        f"Stop-loss (OCO stop): {result.stop_loss:.6f}",
        f"Position notional: {result.position_notional:.2f}",
        f"Quantity: {result.quantity:.6f}",
        f"Success probability: {prob_percent:.1f}%",
        f"Rationale: {result.reasoning}",
    ]

    metric_order = [
        "rsi",
        "macd",
        "macd_hist",
        "momentum",
        "trend_strength",
        "volatility",
        "atr",
    ]
    metric_lines = []
    for key in metric_order:
        value = result.metrics.get(key)
        if value is None:
            continue
        metric_lines.append(f"  {key}: {value:.6f}")

    if metric_lines:
        lines.append("Key metrics:")
        lines.extend(metric_lines)

    return "\n".join(lines)


def log_results(results: List[AnalysisResult]) -> None:
    """Print a sorted summary of all analyzed symbols."""

    if not results:
        logging.warning("No eligible coins found under the current constraints")
        return

    results_sorted = sorted(results, key=lambda r: r.probability, reverse=True)

    logging.info("Top %d candidates: %s", len(results_sorted), [r.symbol for r in results_sorted])

    best = results_sorted[0]
    logging.info("Best candidate %s with %.1f%% probability", best.symbol, best.probability * 100)

    print("\n=== OCO SIGNAL SUGGESTION ===")
    print(generate_oco_signal(best))

    print("\n--- Additional Candidates ---")
    for result in results_sorted[1:5]:
        print(f"{result.symbol}: {result.probability * 100:.1f}% | TP: {result.take_profit:.6f} | SL: {result.stop_loss:.6f}")


# ---------------------------------------------------------------------------
# CLI + orchestration
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binance all-market screener with OCO output")
    parser.add_argument("--budget", type=float, default=200.0, help="Total capital in quote asset for position sizing")
    parser.add_argument("--max-risk", type=float, default=0.10, help="Max fraction of budget to risk per trade")
    parser.add_argument("--min-volume", type=float, default=1_000_000.0, help="Minimum 24h quote volume")
    parser.add_argument("--min-depth", type=float, default=10_000.0, help="Minimum combined order book notional")
    parser.add_argument("--quote", type=str, default="USDT", help="Quote asset to analyse (default: USDT)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--max-candidates", type=int, default=None, help="Optional cap on number of symbols to analyse")
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    budget = BudgetConfig(total_budget=args.budget, max_risk_per_trade=args.max_risk, quote_asset=args.quote)
    indicators = IndicatorParams()
    risk = RiskParams(
        min_quote_volume=args.min_volume,
        min_order_book_notional=args.min_depth,
        max_candidates=args.max_candidates,
    )

    logging.info(
        "Starting Binance spot scan across all %s markets with capital %.2f",
        budget.quote_asset,
        budget.total_budget,
    )
    logging.debug("Budget config: %s", budget)
    logging.debug("Risk params: %s", risk)
    logging.debug("Indicator params: %s", indicators)

    session = _make_session()

    try:
        snapshots = fetch_data(session, budget, risk)
        results: List[AnalysisResult] = []
        for snapshot in snapshots:
            logging.info("Analysing %s (price %.4f)", snapshot.symbol, snapshot.last_price)
            result = analyze_coin(session, snapshot, budget, indicators, risk)
            if result:
                results.append(result)

        log_results(results)

    finally:
        session.close()
        logging.debug("HTTP session closed")


if __name__ == "__main__":
    main()

