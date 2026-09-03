"""Buy-and-hold benchmark helpers for honest strategy evaluation."""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd


def buy_and_hold_return(
    prices: pd.Series,
    initial_cash: float = 100_000.0,
    commission_rate: float = 0.001,
) -> Dict[str, float]:
    """
    Simulate buying max shares at first bar and holding to last bar.
    """
    clean = prices.dropna()
    if len(clean) < 2:
        return {
            "buy_hold_return": 0.0,
            "final_value": initial_cash,
            "shares": 0,
        }

    start_price = float(clean.iloc[0])
    end_price = float(clean.iloc[-1])
    if start_price <= 0:
        return {
            "buy_hold_return": 0.0,
            "final_value": initial_cash,
            "shares": 0,
        }

    # Account for entry commission (one-way), same rate as MockTrader default
    affordable = int(initial_cash / (start_price * (1 + commission_rate)))
    cost = affordable * start_price * (1 + commission_rate)
    cash_left = initial_cash - cost
    final_value = cash_left + affordable * end_price
    total_return = ((final_value - initial_cash) / initial_cash) * 100 if initial_cash else 0.0

    return {
        "buy_hold_return": total_return,
        "final_value": final_value,
        "shares": affordable,
        "start_price": start_price,
        "end_price": end_price,
    }


def align_buy_hold_curve(
    timestamps: pd.Series,
    prices: pd.Series,
    initial_cash: float = 100_000.0,
    commission_rate: float = 0.001,
) -> Optional[pd.DataFrame]:
    """Equity curve for buy-and-hold on the same timestamps."""
    df = pd.DataFrame({"timestamp": pd.to_datetime(timestamps), "price": prices}).dropna()
    if len(df) < 2:
        return None

    start_price = float(df["price"].iloc[0])
    shares = int(initial_cash / (start_price * (1 + commission_rate)))
    cost = shares * start_price * (1 + commission_rate)
    cash_left = initial_cash - cost
    df = df.copy()
    df["value"] = cash_left + shares * df["price"]
    return df[["timestamp", "value"]]
