"""Public Yahoo Finance daily data loader (no API key required)."""

from __future__ import annotations

from typing import List, Optional

import pandas as pd


DEFAULT_DEMO_TICKERS = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]


def fetch_daily_ohlcv(
    tickers: Optional[List[str]] = None,
    period: str = "2y",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Download daily OHLCV via yfinance and normalize to project schema.

    Columns: ticker, datetime_et, open, high, low, close, volume
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required for the demo pipeline. Install with: pip install yfinance"
        ) from exc

    tickers = tickers or DEFAULT_DEMO_TICKERS
    frames: List[pd.DataFrame] = []

    for ticker in tickers:
        raw = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=auto_adjust,
            progress=False,
            threads=False,
        )
        if raw is None or raw.empty:
            print(f"Warning: no yfinance data for {ticker}")
            continue

        # yfinance may return MultiIndex columns for single tickers in newer versions
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]

        df = raw.reset_index()
        date_col = "Date" if "Date" in df.columns else df.columns[0]
        df = df.rename(
            columns={
                date_col: "datetime_et",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df["ticker"] = ticker
        df["datetime_et"] = pd.to_datetime(df["datetime_et"]).dt.tz_localize(None)
        keep = ["ticker", "datetime_et", "open", "high", "low", "close", "volume"]
        frames.append(df[keep].dropna(subset=["close"]))

    if not frames:
        raise RuntimeError(
            "Failed to download any market data from yfinance. Check network connectivity."
        )

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["ticker", "datetime_et"]).reset_index(drop=True)
    return out


def add_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the same core technical features used by the Polygon pipeline."""
    from src.strategies.technical_indicators import (
        calculate_ema,
        calculate_rsi,
        calculate_sma,
    )
    import numpy as np

    parts = []
    for ticker, group in df.groupby("ticker", sort=False):
        g = group.sort_values("datetime_et").copy()
        g["sma_20"] = calculate_sma(g["close"], 20)
        g["sma_50"] = calculate_sma(g["close"], 50)
        g["ema_12"] = calculate_ema(g["close"], 12)
        g["ema_26"] = calculate_ema(g["close"], 26)
        g["rsi_14"] = calculate_rsi(g["close"], 14)
        g["returns"] = g["close"].pct_change()
        g["log_returns"] = np.log(g["close"] / g["close"].shift(1))
        parts.append(g)

    return pd.concat(parts, ignore_index=True)
