# Binance Spot Signal Bot

This project contains a Python script (`bot.py`) that connects to Binance's
public REST API, evaluates a moving-average/RSI/ATR strategy across a basket of
liquid spot markets, and reports the highest-confidence trading signal
including entry, stop-loss, and take-profit levels.

## Strategy Overview

- Trend: bullish or bearish 50/200 EMA crossover on the hourly chart.
- Momentum filter: RSI(14) must be above 55 for long setups or below 45 for
  short setups.
- Risk management: stop-loss at 1 × ATR(14), take-profit at 1.5 × ATR(14) from
  the proposed entry, yielding a 1:1.5 risk-to-reward ratio.

## Prerequisites

- Python 3.9+
- Dependencies: `pandas`, `requests`

Install dependencies:

```bash
pip install pandas requests
```

## Usage

```bash
python bot.py
```

The script prints either the best available signal (symbol, direction, entry,
stop-loss, take-profit, risk/reward, and indicator values) or informs you that
no qualifying signals were found.

You can extend the universe of symbols or adjust the timeframe by editing the
`symbols` list and `interval` variable inside `bot.py`.