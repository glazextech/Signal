"""Disciplined intraday scalping trading robot for MetaTrader 5.

The robot implements a dual-timeframe (M15 trend filter, M1 execution)
scalping strategy with strict risk controls, dynamic position sizing, CSV
logging, and an optional historical backtest mode. It uses only synchronous
MetaTrader5 API calls.

WARNING: Leveraged trading carries significant risk. Test thoroughly on a demo
account and ensure compliance with broker and regulatory requirements before
deploying live capital.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import MetaTrader5 as mt5  # type: ignore
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants & configuration
# ---------------------------------------------------------------------------


TIMEFRAME_ALIASES: Dict[str, int] = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
}

MAGIC_NUMBER = 20251106


@dataclass
class StrategyParameters:
    """Static inputs defined at start-up."""

    symbols: List[str] = field(default_factory=lambda: ["XAUUSD", "EURUSD"])
    analysis_tf: int = mt5.TIMEFRAME_M15
    entry_tf: int = mt5.TIMEFRAME_M1
    ema_fast_period: int = 8
    ema_slow_period: int = 50
    rsi_period: int = 14
    rsi_entry_long: float = 40.0
    rsi_entry_short: float = 60.0
    rsi_momentum_long: float = 45.0
    rsi_momentum_short: float = 55.0
    atr_period: int = 14
    atr_multiplier: float = 1.2
    reward_risk_ratio: float = 1.5
    risk_percent: float = 2.0
    max_spread_points: float = 35.0
    max_open_trades_per_symbol: int = 1
    max_trades_per_day: int = 6
    daily_loss_limit_pct: float = 8.0
    max_consecutive_losses: int = 3
    loss_cooldown_minutes: int = 60
    lookback_bars_analysis: int = 300
    lookback_bars_entry: int = 1500
    slippage_points: float = 30.0
    trade_log_path: Path = field(default_factory=lambda: Path("trades_log.csv"))
    backtest: bool = False
    backtest_days: int = 5
    poll_interval_seconds: int = 15
    terminal_path: Optional[str] = None


@dataclass
class SymbolDailyStats:
    """Aggregated intraday performance for each symbol."""

    trades_today: int = 0
    net_profit: float = 0.0
    consecutive_losses: int = 0
    last_loss_time: Optional[datetime] = None
    cooldown_until: Optional[datetime] = None


@dataclass
class StrategyState:
    symbol_stats: Dict[str, SymbolDailyStats] = field(default_factory=dict)
    logged_deals: set[int] = field(default_factory=set)


@dataclass
class TradeDecision:
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    atr_points: float
    spread_points: float


@dataclass
class BacktestTrade:
    symbol: str
    direction: str
    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    lot: float
    atr_points: float
    spread_points: float


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def today_midnight_utc() -> datetime:
    now = utcnow()
    return datetime(now.year, now.month, now.day, tzinfo=timezone.utc)


def parse_timeframe(alias: str, default: int) -> int:
    if not alias:
        return default
    key = alias.strip().upper()
    if key not in TIMEFRAME_ALIASES:
        valid = ", ".join(TIMEFRAME_ALIASES)
        raise ValueError(f"Unsupported timeframe '{alias}'. Valid options: {valid}")
    return TIMEFRAME_ALIASES[key]


def timeframe_to_str(timeframe: int) -> str:
    reverse = {value: key for key, value in TIMEFRAME_ALIASES.items()}
    return reverse.get(timeframe, str(timeframe))


def ensure_trade_log(path: Path) -> None:
    if not path.exists():
        with path.open("w", newline="") as file:
            csv.writer(file).writerow(
                [
                    "timestamp",
                    "event_type",
                    "mode",
                    "symbol",
                    "direction",
                    "price",
                    "stop_loss",
                    "take_profit",
                    "lot",
                    "balance",
                    "profit",
                    "spread_points",
                    "atr_points",
                    "comment",
                ]
            )


def append_trade_log(
    params: StrategyParameters,
    *,
    event_type: str,
    mode: str,
    symbol: str,
    direction: str,
    price: float,
    stop_loss: float,
    take_profit: float,
    lot: float,
    balance: float,
    profit: Optional[float],
    spread_points: Optional[float],
    atr_points: Optional[float],
    comment: str,
) -> None:
    with params.trade_log_path.open("a", newline="") as file:
        csv.writer(file).writerow(
            [
                utcnow().isoformat(),
                event_type,
                mode,
                symbol,
                direction,
                f"{price:.5f}",
                f"{stop_loss:.5f}",
                f"{take_profit:.5f}",
                f"{lot:.2f}",
                f"{balance:.2f}",
                "" if profit is None else f"{profit:.2f}",
                "" if spread_points is None else f"{spread_points:.2f}",
                "" if atr_points is None else f"{atr_points:.5f}",
                comment,
            ]
        )


def initialize_mt5(params: StrategyParameters) -> None:
    init_kwargs = {"path": params.terminal_path} if params.terminal_path else {}
    if not mt5.initialize(**init_kwargs):
        raise RuntimeError(f"MetaTrader5 initialize failed: {mt5.last_error()}")

    account_info = mt5.account_info()
    if account_info is None:
        raise RuntimeError("Failed to obtain account info; ensure terminal is logged in.")

    logging.info(
        "Connected to MT5 | login=%s | leverage=%s | balance=%.2f",
        account_info.login,
        account_info.leverage,
        account_info.balance,
    )

    for symbol in params.symbols:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Unable to select symbol '{symbol}'. Ensure it is visible in Market Watch.")


def shutdown_mt5() -> None:
    mt5.shutdown()


def rates_to_dataframe(rates) -> pd.DataFrame:
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    return df


def fetch_rates(symbol: str, timeframe: int, count: int) -> pd.DataFrame:
    raw = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"No rates for {symbol} ({timeframe_to_str(timeframe)}) | error={mt5.last_error()}")
    return rates_to_dataframe(raw)


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.bfill().fillna(50)


def calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr.bfill()


def enrich_entry_dataframe(df: pd.DataFrame, params: StrategyParameters) -> pd.DataFrame:
    enriched = df.copy()
    enriched["ema_fast"] = calculate_ema(enriched["close"], params.ema_fast_period)
    enriched["ema_slow"] = calculate_ema(enriched["close"], params.ema_slow_period)
    enriched["rsi"] = calculate_rsi(enriched["close"], params.rsi_period)
    enriched["atr"] = calculate_atr(enriched, params.atr_period)
    return enriched.dropna()


def enrich_analysis_dataframe(df: pd.DataFrame, params: StrategyParameters) -> pd.DataFrame:
    enriched = df.copy()
    enriched["ema_fast"] = calculate_ema(enriched["close"], params.ema_fast_period)
    enriched["ema_slow"] = calculate_ema(enriched["close"], params.ema_slow_period)
    return enriched.dropna()


def determine_trend(df: pd.DataFrame) -> Optional[str]:
    if df.empty:
        return None
    latest = df.iloc[-1]
    if latest["ema_fast"] > latest["ema_slow"]:
        return "buy"
    if latest["ema_fast"] < latest["ema_slow"]:
        return "sell"
    return None


def normalize_price(price: float, point: float) -> float:
    return round(price / point) * point


def normalize_volume(volume: float, symbol_info) -> float:
    min_volume = symbol_info.volume_min
    max_volume = symbol_info.volume_max
    step = symbol_info.volume_step
    if volume < min_volume:
        return 0.0
    steps = math.floor((volume - min_volume) / step + 1e-9)
    normalized = min_volume + steps * step
    return min(max(normalized, min_volume), max_volume)


def value_per_point(symbol_info) -> float:
    tick_size = symbol_info.trade_tick_size or symbol_info.point
    tick_value = symbol_info.trade_tick_value
    if not tick_value:
        tick_value = symbol_info.point * symbol_info.trade_contract_size
    if tick_size == 0:
        return 0.0
    return tick_value / tick_size


def calculate_lot_size(
    symbol_info,
    balance: float,
    params: StrategyParameters,
    stop_distance_points: float,
) -> Tuple[float, Dict[str, float]]:
    val_per_point = value_per_point(symbol_info)
    risk_amount = balance * (params.risk_percent / 100)
    risk_per_lot = val_per_point * stop_distance_points

    debug = {
        "balance": balance,
        "risk_amount": risk_amount,
        "stop_distance_points": stop_distance_points,
        "value_per_point": val_per_point,
        "risk_per_lot": risk_per_lot,
    }

    if risk_per_lot <= 0 or risk_amount <= 0:
        return 0.0, debug

    raw_volume = risk_amount / risk_per_lot
    debug["raw_volume"] = raw_volume

    normalized = normalize_volume(raw_volume, symbol_info)
    debug["normalized_volume"] = normalized

    return normalized, debug


def check_spread(symbol_info, params: StrategyParameters, tick) -> Tuple[bool, float]:
    spread_points = (tick.ask - tick.bid) / symbol_info.point
    return spread_points <= params.max_spread_points, spread_points


def positions_open(symbol: str) -> int:
    positions = mt5.positions_get(symbol=symbol)
    return len(positions) if positions else 0


def compute_daily_metrics(params: StrategyParameters, state: StrategyState) -> Tuple[Dict[str, SymbolDailyStats], float]:
    start = today_midnight_utc()
    deals = mt5.history_deals_get(start, utcnow()) or []

    stats_map: Dict[str, SymbolDailyStats] = {symbol: SymbolDailyStats() for symbol in params.symbols}
    total_profit = 0.0

    for deal in deals:
        if deal.entry != mt5.DEAL_ENTRY_OUT:
            continue
        symbol = getattr(deal, "symbol", "")
        if symbol not in stats_map:
            continue
        stats = stats_map[symbol]
        stats.trades_today += 1
        stats.net_profit += deal.profit
        total_profit += deal.profit

    for symbol in params.symbols:
        stats = stats_map[symbol]
        symbol_deals = [d for d in deals if getattr(d, "symbol", "") == symbol and d.entry == mt5.DEAL_ENTRY_OUT]
        symbol_deals.sort(key=lambda d: d.time)
        consecutive_losses = 0
        last_loss_time = None
        for deal in symbol_deals:
            if deal.profit < 0:
                consecutive_losses += 1
                last_loss_time = datetime.fromtimestamp(deal.time, tz=timezone.utc)
            elif deal.profit > 0:
                consecutive_losses = 0
        stats.consecutive_losses = consecutive_losses
        stats.last_loss_time = last_loss_time
        if consecutive_losses >= params.max_consecutive_losses and last_loss_time is not None:
            stats.cooldown_until = last_loss_time + timedelta(minutes=params.loss_cooldown_minutes)
        else:
            stats.cooldown_until = None

    # Log exits not previously recorded
    new_exits = [d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT and d.ticket not in state.logged_deals]
    if new_exits:
        account_info = mt5.account_info()
        balance_snapshot = account_info.balance if account_info else 0.0
        for deal in new_exits:
            append_trade_log(
                params,
                event_type="EXIT",
                mode="LIVE",
                symbol=deal.symbol,
                direction="buy" if deal.type == mt5.DEAL_TYPE_BUY else "sell",
                price=deal.price,
                stop_loss=deal.price,
                take_profit=deal.price,
                lot=deal.volume,
                balance=balance_snapshot,
                profit=deal.profit,
                spread_points=None,
                atr_points=None,
                comment=f"deal={deal.ticket}; order={deal.order}",
            )
    state.logged_deals.update(d.ticket for d in deals if d.entry == mt5.DEAL_ENTRY_OUT)

    return stats_map, total_profit


def loss_limit_breached(total_profit: float, params: StrategyParameters) -> bool:
    account_info = mt5.account_info()
    if account_info is None:
        return True
    threshold = -abs(account_info.balance * (params.daily_loss_limit_pct / 100))
    return total_profit <= threshold


def build_trade_decision(symbol: str, params: StrategyParameters, symbol_info) -> Optional[TradeDecision]:
    analysis_df = enrich_analysis_dataframe(
        fetch_rates(symbol, params.analysis_tf, params.lookback_bars_analysis + params.ema_slow_period),
        params,
    )
    trend = determine_trend(analysis_df)
    if trend is None:
        logging.debug("%s trend ambiguous; skipping", symbol)
        return None

    entry_df = enrich_entry_dataframe(
        fetch_rates(symbol, params.entry_tf, params.lookback_bars_entry + params.ema_slow_period),
        params,
    )
    if len(entry_df) < 2:
        logging.debug("%s insufficient entry data; skipping", symbol)
        return None

    prev = entry_df.iloc[-2]
    last = entry_df.iloc[-1]

    bullish_candle = last["close"] > last["open"]
    bearish_candle = last["close"] < last["open"]
    rsi_cross_up = prev["rsi"] < params.rsi_entry_long and last["rsi"] >= params.rsi_entry_long
    rsi_cross_down = prev["rsi"] > params.rsi_entry_short and last["rsi"] <= params.rsi_entry_short
    price_above_fast = last["close"] > last["ema_fast"]
    price_below_fast = last["close"] < last["ema_fast"]

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logging.warning("Tick data missing for %s", symbol)
        return None

    spread_ok, spread_points = check_spread(symbol_info, params, tick)
    if not spread_ok:
        logging.info("%s spread %.2f > %.2f; skipping", symbol, spread_points, params.max_spread_points)
        return None

    atr_points = last["atr"]
    if atr_points <= 0:
        logging.debug("%s ATR non-positive; skipping", symbol)
        return None

    if trend == "buy" and bullish_candle and rsi_cross_up and price_above_fast and last["rsi"] <= params.rsi_momentum_long:
        entry_price = tick.ask
        stop_loss = entry_price - params.atr_multiplier * atr_points
        take_profit = entry_price + params.reward_risk_ratio * (entry_price - stop_loss)
        return TradeDecision("buy", entry_price, stop_loss, take_profit, atr_points, spread_points)

    if trend == "sell" and bearish_candle and rsi_cross_down and price_below_fast and last["rsi"] >= params.rsi_momentum_short:
        entry_price = tick.bid
        stop_loss = entry_price + params.atr_multiplier * atr_points
        take_profit = entry_price - params.reward_risk_ratio * (stop_loss - entry_price)
        return TradeDecision("sell", entry_price, stop_loss, take_profit, atr_points, spread_points)

    logging.debug("%s no qualifying setup", symbol)
    return None


def submit_order(
    symbol: str,
    decision: TradeDecision,
    params: StrategyParameters,
    symbol_info,
    account_balance: float,
) -> None:
    stop_distance_points = abs(decision.entry_price - decision.stop_loss) / symbol_info.point
    lot, debug = calculate_lot_size(symbol_info, account_balance, params, stop_distance_points)
    logging.info(
        "%s lot sizing | stop_pts=%.2f | val_per_point=%.5f | risk=%.2f | raw=%.3f | normalized=%.3f",
        symbol,
        debug.get("stop_distance_points", 0.0),
        debug.get("value_per_point", 0.0),
        debug.get("risk_amount", 0.0),
        debug.get("raw_volume", 0.0),
        debug.get("normalized_volume", 0.0),
    )

    if lot <= 0:
        logging.warning("%s lot size below broker minimum; trade skipped", symbol)
        return

    entry_price = normalize_price(decision.entry_price, symbol_info.point)
    stop_loss = normalize_price(decision.stop_loss, symbol_info.point)
    take_profit = normalize_price(decision.take_profit, symbol_info.point)

    order_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY if decision.direction == "buy" else mt5.ORDER_TYPE_SELL,
        "price": entry_price,
        "sl": stop_loss,
        "tp": take_profit,
        "deviation": int(params.slippage_points),
        "magic": MAGIC_NUMBER,
        "comment": "scalp_bot",
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    check = mt5.order_check(order_request)
    if check is None or check.retcode != mt5.TRADE_RETCODE_DONE:
        logging.warning("%s order_check rejected: %s", symbol, getattr(check, "comment", "unknown"))
        return

    result = mt5.order_send(order_request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error("%s order_send failed: %s", symbol, getattr(result, "comment", "unknown"))
        return

    logging.info(
        "%s %s order placed | ticket=%s | entry=%.5f | sl=%.5f | tp=%.5f | lot=%.2f | spread=%.2f",
        symbol,
        decision.direction.upper(),
        result.order,
        entry_price,
        stop_loss,
        take_profit,
        lot,
        decision.spread_points,
    )

    append_trade_log(
        params,
        event_type="ENTRY",
        mode="LIVE",
        symbol=symbol,
        direction=decision.direction,
        price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        lot=lot,
        balance=account_balance,
        profit=None,
        spread_points=decision.spread_points,
        atr_points=decision.atr_points,
        comment=f"ticket={result.order}",
    )


def run_live_trading(params: StrategyParameters, state: StrategyState) -> None:
    logging.info(
        "Live trading started | symbols=%s | analysis_tf=%s | entry_tf=%s",
        ",".join(params.symbols),
        timeframe_to_str(params.analysis_tf),
        timeframe_to_str(params.entry_tf),
    )

    ensure_trade_log(params.trade_log_path)

    while True:
        try:
            account_info = mt5.account_info()
            if account_info is None:
                raise RuntimeError("Account info unavailable; aborting loop")

            symbol_stats, total_profit = compute_daily_metrics(params, state)
            state.symbol_stats = symbol_stats

            if loss_limit_breached(total_profit, params):
                logging.warning("Daily loss limit reached (%.2f). Standing aside until tomorrow.", total_profit)
                time.sleep(params.poll_interval_seconds)
                continue

            for symbol in params.symbols:
                stats = symbol_stats[symbol]

                if stats.cooldown_until and utcnow() < stats.cooldown_until:
                    remaining = max((stats.cooldown_until - utcnow()).total_seconds() / 60, 0)
                    logging.info("%s cooldown active for %.1f minutes", symbol, remaining)
                    continue

                if stats.trades_today >= params.max_trades_per_day:
                    logging.info("%s max trades per day reached (%d)", symbol, params.max_trades_per_day)
                    continue

                if positions_open(symbol) >= params.max_open_trades_per_symbol:
                    logging.debug("%s has open position; awaiting exit", symbol)
                    continue

                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    logging.warning("Symbol info missing for %s", symbol)
                    continue

                decision = build_trade_decision(symbol, params, symbol_info)
                if decision is None:
                    continue

                submit_order(symbol, decision, params, symbol_info, account_info.balance)

            time.sleep(params.poll_interval_seconds)

        except KeyboardInterrupt:
            logging.info("Interrupted by user; stopping live loop")
            break
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("Unexpected live trading error: %s", exc)
            time.sleep(params.poll_interval_seconds)


def run_backtest(params: StrategyParameters) -> None:
    logging.info("Running backtest for last %d days", params.backtest_days)
    ensure_trade_log(params.trade_log_path)

    account_info = mt5.account_info()
    if account_info is None:
        raise RuntimeError("Account info required for backtesting")

    start_balance = account_info.balance
    equity = start_balance
    equity_curve: List[Tuple[datetime, float]] = [(utcnow(), equity)]
    trade_count = 0
    win_count = 0
    loss_count = 0

    minutes_per_day = 24 * 60
    bars_needed = params.backtest_days * minutes_per_day + params.lookback_bars_entry

    for symbol in params.symbols:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logging.warning("Skipping %s in backtest; symbol info unavailable", symbol)
            continue

        entry_df = enrich_entry_dataframe(fetch_rates(symbol, params.entry_tf, bars_needed), params)
        analysis_df = enrich_analysis_dataframe(fetch_rates(symbol, params.analysis_tf, bars_needed // 3), params)
        trend_series = (analysis_df["ema_fast"] > analysis_df["ema_slow"]).map(lambda flag: "buy" if flag else "sell")
        trend_series = trend_series.reindex(entry_df.index, method="ffill")
        entry_df = entry_df.join(trend_series.rename("trend"))

        current_position: Optional[BacktestTrade] = None

        for idx in range(1, len(entry_df)):
            row = entry_df.iloc[idx]
            prev = entry_df.iloc[idx - 1]
            trend = row.get("trend")
            if trend not in ("buy", "sell"):
                continue

            if current_position is None:
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    continue
                spread_ok, spread_points = check_spread(symbol_info, params, tick)
                if not spread_ok:
                    continue

                bullish_candle = row["close"] > row["open"]
                bearish_candle = row["close"] < row["open"]
                rsi_cross_up = prev["rsi"] < params.rsi_entry_long and row["rsi"] >= params.rsi_entry_long
                rsi_cross_down = prev["rsi"] > params.rsi_entry_short and row["rsi"] <= params.rsi_entry_short
                price_above_fast = row["close"] > row["ema_fast"]
                price_below_fast = row["close"] < row["ema_fast"]
                atr_points = row["atr"]
                if atr_points <= 0:
                    continue

                if trend == "buy" and bullish_candle and rsi_cross_up and price_above_fast and row["rsi"] <= params.rsi_momentum_long:
                    entry_price = row["close"]
                    stop_loss = entry_price - params.atr_multiplier * atr_points
                    take_profit = entry_price + params.reward_risk_ratio * (entry_price - stop_loss)
                    stop_points = abs(entry_price - stop_loss) / symbol_info.point
                    lot, _ = calculate_lot_size(symbol_info, equity, params, stop_points)
                    if lot > 0:
                        current_position = BacktestTrade(symbol, "buy", row.name, entry_price, stop_loss, take_profit, lot, atr_points, spread_points)
                        append_trade_log(
                            params,
                            event_type="ENTRY",
                            mode="BACKTEST",
                            symbol=symbol,
                            direction="buy",
                            price=entry_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            lot=lot,
                            balance=equity,
                            profit=None,
                            spread_points=spread_points,
                            atr_points=atr_points,
                            comment="backtest entry",
                        )
                elif trend == "sell" and bearish_candle and rsi_cross_down and price_below_fast and row["rsi"] >= params.rsi_momentum_short:
                    entry_price = row["close"]
                    stop_loss = entry_price + params.atr_multiplier * row["atr"]
                    take_profit = entry_price - params.reward_risk_ratio * (stop_loss - entry_price)
                    stop_points = abs(entry_price - stop_loss) / symbol_info.point
                    lot, _ = calculate_lot_size(symbol_info, equity, params, stop_points)
                    if lot > 0:
                        current_position = BacktestTrade(symbol, "sell", row.name, entry_price, stop_loss, take_profit, lot, row["atr"], spread_points)
                        append_trade_log(
                            params,
                            event_type="ENTRY",
                            mode="BACKTEST",
                            symbol=symbol,
                            direction="sell",
                            price=entry_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            lot=lot,
                            balance=equity,
                            profit=None,
                            spread_points=spread_points,
                            atr_points=row["atr"],
                            comment="backtest entry",
                        )
            else:
                high = row["high"]
                low = row["low"]
                exit_price = None
                exit_reason = ""

                if current_position.direction == "buy":
                    if low <= current_position.stop_loss:
                        exit_price = current_position.stop_loss
                        exit_reason = "SL"
                    if high >= current_position.take_profit and exit_price is None:
                        exit_price = current_position.take_profit
                        exit_reason = "TP"
                else:
                    if high >= current_position.stop_loss:
                        exit_price = current_position.stop_loss
                        exit_reason = "SL"
                    if low <= current_position.take_profit and exit_price is None:
                        exit_price = current_position.take_profit
                        exit_reason = "TP"

                if exit_price is not None:
                    direction_factor = 1 if current_position.direction == "buy" else -1
                    pnl = (
                        (exit_price - current_position.entry_price)
                        * direction_factor
                        * value_per_point(symbol_info)
                        * current_position.lot
                    )
                    equity += pnl
                    equity_curve.append((row.name, equity))
                    trade_count += 1
                    if pnl > 0:
                        win_count += 1
                    else:
                        loss_count += 1
                    append_trade_log(
                        params,
                        event_type="EXIT",
                        mode="BACKTEST",
                        symbol=current_position.symbol,
                        direction=current_position.direction,
                        price=exit_price,
                        stop_loss=current_position.stop_loss,
                        take_profit=current_position.take_profit,
                        lot=current_position.lot,
                        balance=equity,
                        profit=pnl,
                        spread_points=current_position.spread_points,
                        atr_points=current_position.atr_points,
                        comment=f"{exit_reason} | duration={(row.name - current_position.entry_time)}",
                    )
                    current_position = None

    if trade_count == 0:
        logging.info("Backtest finished: no trades triggered.")
        return

    max_equity = max(value for _, value in equity_curve)
    min_equity = min(value for _, value in equity_curve)
    max_drawdown = 0.0
    peak = equity_curve[0][1]
    for _, eq in equity_curve:
        if eq > peak:
            peak = eq
        drawdown = (eq - peak) / peak if peak else 0.0
        max_drawdown = min(max_drawdown, drawdown)

    logging.info("Backtest summary")
    logging.info("  Start balance : %.2f", start_balance)
    logging.info("  End balance   : %.2f", equity)
    logging.info("  Net PnL       : %.2f", equity - start_balance)
    logging.info("  Max drawdown  : %.2f%%", max_drawdown * 100)
    logging.info("  Trades        : %d (wins=%d, losses=%d, win_rate=%.1f%%)", trade_count, win_count, loss_count, (win_count / trade_count * 100) if trade_count else 0)
    logging.info("  Equity range  : %.2f -> %.2f", min_equity, max_equity)


def parse_arguments() -> StrategyParameters:
    parser = argparse.ArgumentParser(description="Intraday scalping trading robot for MetaTrader 5")
    parser.add_argument("--symbols", type=str, default="XAUUSD,EURUSD", help="Comma-separated symbols list")
    parser.add_argument("--analysis-tf", type=str, default="M15", help="Trend timeframe (e.g. M15)")
    parser.add_argument("--entry-tf", type=str, default="M1", help="Execution timeframe (e.g. M1)")
    parser.add_argument("--risk", type=float, default=2.0, help="Risk per trade as %% of balance")
    parser.add_argument("--max-spread", type=float, default=35.0, help="Maximum acceptable spread in points")
    parser.add_argument("--daily-loss", type=float, default=8.0, help="Daily loss stop as %% of balance")
    parser.add_argument("--max-trades", type=int, default=6, help="Maximum trades per day per symbol")
    parser.add_argument("--cooldown-losses", type=int, default=3, help="Consecutive losses before cooldown")
    parser.add_argument("--cooldown-min", type=int, default=60, help="Cooldown duration in minutes")
    parser.add_argument("--backtest", action="store_true", help="Enable backtest mode")
    parser.add_argument("--backtest-days", type=int, default=5, help="Number of days to include in backtest")
    parser.add_argument("--terminal-path", type=str, default=None, help="Optional path to terminal64.exe")
    parser.add_argument("--poll", type=int, default=15, help="Polling interval in seconds for live trading")
    parser.add_argument("--log-path", type=str, default="trades_log.csv", help="CSV log file path")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    try:
        analysis_tf = parse_timeframe(args.analysis_tf, mt5.TIMEFRAME_M15)
        entry_tf = parse_timeframe(args.entry_tf, mt5.TIMEFRAME_M1)
    except ValueError as exc:
        parser.error(str(exc))

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        parser.error("At least one symbol must be specified")

    params = StrategyParameters(
        symbols=symbols,
        analysis_tf=analysis_tf,
        entry_tf=entry_tf,
        risk_percent=args.risk,
        max_spread_points=args.max_spread,
        daily_loss_limit_pct=args.daily_loss,
        max_trades_per_day=args.max_trades,
        max_consecutive_losses=args.cooldown_losses,
        loss_cooldown_minutes=args.cooldown_min,
        backtest=args.backtest,
        backtest_days=args.backtest_days,
        terminal_path=args.terminal_path,
        poll_interval_seconds=args.poll,
        trade_log_path=Path(args.log_path),
    )

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    return params


def main() -> None:
    params = parse_arguments()
    state = StrategyState()

    try:
        initialize_mt5(params)
        logging.info("Strategy initialised | mode=%s", "BACKTEST" if params.backtest else "LIVE")
        logging.info("Parameters: risk=%.2f%%, rr=%.2f, atr_k=%.2f", params.risk_percent, params.reward_risk_ratio, params.atr_multiplier)

        if params.backtest:
            run_backtest(params)
        else:
            run_live_trading(params, state)

    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Fatal error: %s", exc)
        sys.exit(1)
    finally:
        shutdown_mt5()
        logging.info("MT5 connection closed")


if __name__ == "__main__":
    main()

