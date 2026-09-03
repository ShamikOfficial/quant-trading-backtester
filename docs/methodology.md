# Methodology

This project is designed as a **research-to-simulation** pipeline, not a live trading system.

## Design principles

1. **Chronological splits only** — no random shuffling of time series.
2. **Embargo / purge gap** — a buffer between train and test reduces label leakage from overlapping forward returns.
3. **Held-out evaluation** — the demo backtests only on the post-embargo test window.
4. **Honest baseline** — every strategy report includes **buy-and-hold** on the same window with the same commission rate.
5. **Frequency-aware risk metrics** — Sharpe / Sortino / volatility annualization uses inferred bar frequency (daily ≈ 252; minute bars scale to trading-year minutes).
6. **Feature hygiene** — model features exclude metadata (`ticker`, timestamps) and raw OHLC columns used only for execution.

## Signal → Order → Execute

```text
Market data
    → Feature engineering (returns, SMA/EMA ratios, RSI flags, volatility, lags)
    → XGBoost predicts next-bar return
    → Confidence gate (threshold + scaled |prediction|)
    → Position sizing (cash / inventory constrained)
    → MockTrader fills with commission
    → Portfolio snapshots → performance report
```

## What this is / is not

| Is | Is not |
|----|--------|
| Reproducible research backtester | Live brokerage integration |
| ML + technical baseline playground | Guaranteed alpha |
| Portfolio-friendly demo via yfinance | Substitute for institutional OMS / risk systems |

## Limitations (call these out in interviews)

- Point-in-time corporate actions / survivorship depend on the data vendor.
- Minute-bar Polygon path needs an API key and rate-limit handling.
- Transaction costs are a flat commission rate (no slippage model beyond that).
- Default XGBoost hyperparameters are intentionally simple for clarity.

## Reproducing the demo numbers in the README

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py demo --plot
```

Results are written to `docs/sample_results/latest_demo_summary.json`.
