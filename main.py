"""
Quant Trading Backtester — unified CLI entrypoint.

Examples:
  python main.py demo
  python main.py demo --tickers AAPL MSFT --plot
  python main.py collect --help
  python main.py train --help
  python main.py backtest --help
"""

from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="quant-trading-backtester",
        description="End-to-end quant trading pipeline: data → features → XGBoost → backtest.",
    )
    sub = parser.add_subparsers(dest="command")

    demo = sub.add_parser("demo", help="Zero-config demo (yfinance, no API key)")
    demo.add_argument("--tickers", nargs="+", default=None)
    demo.add_argument("--period", default="2y")
    demo.add_argument("--plot", action="store_true")
    demo.add_argument("--min-confidence", type=float, default=0.35)
    demo.add_argument("--embargo-periods", type=int, default=5)
    demo.add_argument("--processed-file", default=None)

    sub.add_parser("collect", help="Polygon/API data collection (see run_data_collection.py)")
    sub.add_parser("train", help="Train models (see run_ml_training.py)")
    sub.add_parser("backtest", help="Run backtests (see run_backtest.py)")

    args, rest = parser.parse_known_args(argv)

    if args.command is None:
        parser.print_help()
        print("\nTip: start with  python main.py demo")
        return 0

    if args.command == "demo":
        from run_demo import main as demo_main

        demo_argv = []
        if args.tickers:
            demo_argv += ["--tickers", *args.tickers]
        demo_argv += ["--period", args.period]
        demo_argv += ["--min-confidence", str(args.min_confidence)]
        demo_argv += ["--embargo-periods", str(args.embargo_periods)]
        if args.plot:
            demo_argv.append("--plot")
        if args.processed_file:
            demo_argv += ["--processed-file", args.processed_file]
        demo_argv += rest
        return demo_main(demo_argv)

    if args.command == "collect":
        from run_data_collection import main as collect_main

        sys.argv = ["run_data_collection.py", *rest]
        collect_main()
        return 0

    if args.command == "train":
        from run_ml_training import main as train_main

        sys.argv = ["run_ml_training.py", *rest]
        train_main()
        return 0

    if args.command == "backtest":
        from run_backtest import main as backtest_main

        sys.argv = ["run_backtest.py", *rest]
        backtest_main()
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
