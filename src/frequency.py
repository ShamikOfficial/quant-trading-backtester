"""Bar-frequency helpers for risk metric annualization."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def infer_periods_per_year(timestamps: pd.Series) -> int:
    """
    Infer annualization factor from bar spacing.

    Returns:
        Approximate bars per trading year (e.g. 252 daily, ~78k minute).
    """
    ts = pd.to_datetime(timestamps, errors="coerce").dropna().sort_values()
    if len(ts) < 3:
        return 252

    median_delta = ts.diff().dropna().median()
    if pd.isna(median_delta):
        return 252

    seconds = float(median_delta.total_seconds())
    if seconds <= 0:
        return 252

    # ~6.5 trading hours/day * 252 trading days
    trading_seconds_per_year = 252 * 6.5 * 3600

    if seconds < 90:
        # ~1-minute bars
        return max(int(trading_seconds_per_year / seconds), 1)
    if seconds < 3600:
        # intraday multi-minute
        return max(int(trading_seconds_per_year / seconds), 1)
    if seconds < 86400 * 1.5:
        # daily
        return 252
    if seconds < 86400 * 8:
        # weekly
        return 52
    return 12


def format_frequency_label(periods_per_year: int) -> str:
    if periods_per_year >= 50_000:
        return "1-minute"
    if periods_per_year >= 5_000:
        return "intraday"
    if periods_per_year >= 200:
        return "daily"
    if periods_per_year >= 40:
        return "weekly"
    return "monthly"


def resolve_periods_per_year(
    portfolio_history: list,
    override: Optional[int] = None,
) -> int:
    if override is not None and override > 0:
        return override
    if not portfolio_history:
        return 252
    ts = pd.Series([row.get("timestamp") for row in portfolio_history])
    return infer_periods_per_year(ts)
