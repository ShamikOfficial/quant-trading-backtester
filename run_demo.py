"""
Zero-config demo pipeline for recruiters and contributors.

Downloads daily OHLCV via yfinance (no API key), trains XGBoost with a
chronological train/test split + embargo, then backtests ONLY on the held-out
window vs buy-and-hold.

Usage:
    python run_demo.py
    python run_demo.py --tickers AAPL MSFT --period 3y --plot
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.benchmark import buy_and_hold_return
from src.evaluator import generate_performance_report, plot_performance_metrics
from src.simulator import MockTrader
from src.strategies.ml_models import MLStrategy, XGBoostModel, train_model_on_processed_data
from src.yfinance_loader import DEFAULT_DEMO_TICKERS, add_daily_features, fetch_daily_ohlcv


def _ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def prepare_demo_dataset(
    tickers: List[str],
    period: str,
    sample_dir: Path,
) -> Path:
    print("=" * 70)
    print("DEMO: downloading daily market data (yfinance, no API key)")
    print("=" * 70)
    raw = fetch_daily_ohlcv(tickers=tickers, period=period)
    print(f"Downloaded {len(raw):,} rows across {raw['ticker'].nunique()} tickers")

    featured = add_daily_features(raw)
    stamp = datetime.now().strftime("%Y%m%d")
    out = sample_dir / f"demo_daily_{stamp}.csv"
    featured.to_csv(out, index=False)
    print(f"Saved featured dataset -> {out}")
    return out


def train_demo_models(
    processed_file: Path,
    tickers: List[str],
    model_dir: Path,
    lookback_window: int,
    embargo_periods: int,
) -> Dict[str, Dict[str, Any]]:
    print("\n" + "=" * 70)
    print("DEMO: training XGBoost models (chronological split + embargo)")
    print("=" * 70)
    _ensure_dirs(model_dir)
    results: Dict[str, Dict[str, Any]] = {}

    for ticker in tickers:
        print(f"\n--- {ticker} ---")
        try:
            model_path = model_dir / f"model_{ticker}.pkl"
            _, metrics = train_model_on_processed_data(
                processed_file=str(processed_file),
                ticker=ticker,
                lookback_window=lookback_window,
                train_test_split=0.7,
                embargo_periods=embargo_periods,
                model_params={"n_estimators": 120, "max_depth": 4, "learning_rate": 0.05},
                save_model_path=str(model_path),
            )
            results[ticker] = metrics
        except Exception as exc:
            print(f"ERROR training {ticker}: {exc}")
            results[ticker] = {"error": str(exc)}

    return results


def backtest_held_out(
    processed_file: Path,
    ticker: str,
    metrics: Dict[str, Any],
    model_dir: Path,
    initial_cash: float,
    min_confidence: float,
    lookback_window: int,
) -> Optional[Dict[str, Any]]:
    """Backtest strictly on the held-out window from training metrics."""
    if "error" in metrics or not metrics.get("test_start"):
        print(f"Skipping backtest for {ticker}: no held-out window")
        return None

    model_path = model_dir / f"model_{ticker}.pkl"
    if not model_path.exists():
        print(f"Skipping backtest for {ticker}: missing model file")
        return None

    model = XGBoostModel()
    model.load_model(str(model_path))

    data = pd.read_csv(processed_file)
    data["datetime_et"] = pd.to_datetime(data["datetime_et"])
    data = data[data["ticker"] == ticker].sort_values("datetime_et").reset_index(drop=True)

    start = pd.to_datetime(metrics["test_start"])
    end = pd.to_datetime(metrics["test_end"])
    # Include lookback history before test window so features are warm
    hist_start = start - pd.Timedelta(days=lookback_window * 3)
    window = data[(data["datetime_et"] >= hist_start) & (data["datetime_et"] <= end)].copy()
    test_mask_start = start

    if len(window) < lookback_window + 20:
        print(f"Skipping {ticker}: insufficient held-out rows ({len(window)})")
        return None

    strategy = MLStrategy(
        name=f"XGBoost_{ticker}",
        parameters={
            "lookback_window": lookback_window,
            "min_confidence": min_confidence,
            "bar_type": "daily",
        },
    )
    strategy.model = model
    strategy.model_trained = True
    strategy.training_data = pd.DataFrame()

    trader = MockTrader(initial_cash=initial_cash, commission_rate=0.001)
    chunk = lookback_window + 30

    print(f"\nBacktesting {ticker} on held-out window {start.date()} -> {end.date()} ...")

    for i in range(chunk, len(window)):
        ts = window.iloc[i]["datetime_et"]
        if ts < test_mask_start:
            continue

        current_data = window.iloc[i - chunk : i + 1]
        price = float(window.iloc[i]["close"])
        position = trader.get_position(ticker)
        current_position = {
            "symbol": ticker,
            "quantity": position.quantity if position else 0,
            "avg_cost": position.avg_cost if position else 0.0,
        }

        signal = strategy.generate_signal(current_data.copy(), current_position)
        if signal["action"] != "HOLD" and signal.get("confidence", 0) >= min_confidence:
            order = strategy.create_order(
                signal,
                {
                    "symbol": ticker,
                    "current_price": price,
                    "cash": trader.get_cash_balance(),
                    "positions": {ticker: current_position},
                },
            )
            if order:
                trader.execute_order(order)

        trader.update_prices({ticker: price})
        trader.record_portfolio_snapshot(ts, {ticker: price})

    if not trader.portfolio_history:
        print(f"No portfolio history for {ticker}")
        return None

    # Buy-and-hold on the same evaluation prices
    eval_prices = window[window["datetime_et"] >= test_mask_start]["close"]
    bh = buy_and_hold_return(eval_prices, initial_cash=initial_cash, commission_rate=0.001)

    report = generate_performance_report(
        trader,
        trader.portfolio_history,
        periods_per_year=252,
        benchmark_total_return=bh["buy_hold_return"],
    )
    report["ticker"] = ticker
    report["evaluation_start"] = str(start.date())
    report["evaluation_end"] = str(end.date())
    report["test_direction_hit_rate"] = metrics.get("test_direction_hit_rate")
    report["test_r2"] = metrics.get("test_r2")
    report["n_train"] = metrics.get("n_train")
    report["n_test"] = metrics.get("n_test")

    return {
        "report": report,
        "portfolio_history": trader.portfolio_history,
        "buy_hold": bh,
    }


def print_summary(rows: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 70)
    print("HELD-OUT BACKTEST SUMMARY (vs buy-and-hold)")
    print("=" * 70)
    header = f"{'Ticker':<8} {'Strategy%':>10} {'BuyHold%':>10} {'Excess%':>10} {'Sharpe':>8} {'MaxDD%':>8} {'HitRate':>8} {'Trades':>7}"
    print(header)
    print("-" * len(header))
    for row in rows:
        r = row["report"]
        print(
            f"{r.get('ticker', '?'):<8} "
            f"{r.get('total_return', 0):>9.2f}% "
            f"{r.get('buy_hold_return', 0):>9.2f}% "
            f"{r.get('excess_return_vs_buy_hold', 0):>9.2f}% "
            f"{r.get('sharpe_ratio', 0):>8.3f} "
            f"{r.get('max_drawdown', 0):>7.2f}% "
            f"{(r.get('test_direction_hit_rate') or 0)*100:>7.1f}% "
            f"{r.get('total_trades', 0):>7d}"
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the zero-config quant trading demo (yfinance → train → held-out backtest)."
    )
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_DEMO_TICKERS)
    parser.add_argument("--period", default="2y", help="yfinance period (e.g. 1y, 2y, 5y)")
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--lookback-window", type=int, default=10)
    parser.add_argument("--embargo-periods", type=int, default=5, help="Purge gap between train and test")
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--sample-dir", default="data/sample")
    parser.add_argument("--model-dir", default="models/demo")
    parser.add_argument("--output-dir", default="docs/sample_results")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--processed-file",
        default=None,
        help="Reuse an existing demo CSV instead of re-downloading",
    )
    args = parser.parse_args(argv)

    sample_dir = Path(args.sample_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    _ensure_dirs(sample_dir, model_dir, output_dir)

    try:
        if args.processed_file:
            processed_file = Path(args.processed_file)
            if not processed_file.exists():
                print(f"ERROR: file not found: {processed_file}", file=sys.stderr)
                return 1
        else:
            processed_file = prepare_demo_dataset(args.tickers, args.period, sample_dir)

        # Discover tickers present in file if user passed a custom CSV
        present = sorted(pd.read_csv(processed_file, usecols=["ticker"])["ticker"].unique())
        tickers = [t for t in args.tickers if t in present] or present

        train_metrics = train_demo_models(
            processed_file=processed_file,
            tickers=tickers,
            model_dir=model_dir,
            lookback_window=args.lookback_window,
            embargo_periods=args.embargo_periods,
        )

        results = []
        for ticker in tickers:
            metrics = train_metrics.get(ticker, {})
            result = backtest_held_out(
                processed_file=processed_file,
                ticker=ticker,
                metrics=metrics,
                model_dir=model_dir,
                initial_cash=args.initial_cash,
                min_confidence=args.min_confidence,
                lookback_window=args.lookback_window,
            )
            if result:
                results.append(result)
                if args.plot:
                    plot_path = output_dir / f"demo_{ticker}_equity.png"
                    plot_performance_metrics(result["portfolio_history"], save_path=str(plot_path))

        if not results:
            print("ERROR: no successful backtests. Check data download / training logs.", file=sys.stderr)
            return 1

        print_summary(results)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = output_dir / f"demo_summary_{stamp}.json"
        payload = {
            "generated_at": datetime.now().isoformat(),
            "processed_file": str(processed_file),
            "tickers": tickers,
            "period": args.period,
            "methodology": {
                "data": "yfinance daily OHLCV",
                "split": "chronological 70/30 with embargo",
                "evaluation": "backtest only on held-out window",
                "benchmark": "buy-and-hold same window + 10bps commission",
            },
            "training": train_metrics,
            "backtests": {r["report"]["ticker"]: r["report"] for r in results},
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

        # Stable pointer for README
        latest = output_dir / "latest_demo_summary.json"
        with open(latest, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

        print(f"\nSaved results -> {summary_path}")
        print(f"Latest pointer -> {latest}")
        print("\nDemo complete. This is the recruiter-friendly path (no Polygon key required).")
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
