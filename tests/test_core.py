"""Smoke tests for core metrics, simulator, and feature hygiene."""

import numpy as np
import pandas as pd

from src.benchmark import buy_and_hold_return
from src.columns import feature_columns
from src.evaluator import (
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_total_return,
)
from src.frequency import infer_periods_per_year
from src.simulator import MockTrader
from src.strategies.technical_indicators import calculate_ema, calculate_rsi, calculate_sma


def test_ema_rsi_sma_run():
    s = pd.Series(np.linspace(100, 120, 60))
    assert len(calculate_sma(s, 10).dropna()) > 0
    assert len(calculate_ema(s, 10).dropna()) > 0
    assert calculate_rsi(s, 14).iloc[-1] >= 0


def test_feature_columns_excludes_metadata():
    cols = ["close", "ticker", "datetime_et", "rsi_14", "returns", "volume"]
    feats = feature_columns(cols)
    assert "rsi_14" in feats
    assert "returns" in feats
    assert "ticker" not in feats
    assert "close" not in feats


def test_infer_daily_frequency():
    ts = pd.date_range("2024-01-01", periods=40, freq="B")
    assert infer_periods_per_year(pd.Series(ts)) == 252


def test_buy_and_hold_positive_on_uptrend():
    prices = pd.Series([100.0, 110.0, 120.0])
    result = buy_and_hold_return(prices, initial_cash=10_000.0, commission_rate=0.0)
    assert result["buy_hold_return"] > 0


def test_simulator_buy_and_sell():
    trader = MockTrader(initial_cash=10_000.0, commission_rate=0.0)
    buy = trader.execute_buy_order("AAPL", 10, 100.0)
    assert buy["status"] == "FILLED"
    assert trader.get_cash_balance() == 9_000.0
    sell = trader.execute_sell_order("AAPL", 10, 110.0)
    assert sell["status"] == "FILLED"
    assert abs(trader.get_cash_balance() - 10_100.0) < 1e-6


def test_sharpe_and_drawdown():
    returns = pd.Series([0.01, -0.005, 0.02, 0.0, -0.01])
    sharpe = calculate_sharpe_ratio(returns, periods_per_year=252)
    assert isinstance(sharpe, float)
    values = pd.Series([100, 110, 105, 120, 90])
    dd, _, _ = calculate_max_drawdown(values)
    assert dd > 0
    assert calculate_total_return(100, 110) == 10.0
