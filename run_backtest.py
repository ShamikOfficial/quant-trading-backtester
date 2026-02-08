# Backtesting script using saved ML models

import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import json

from src.simulator import MockTrader
from src.evaluator import generate_performance_report, plot_performance_metrics
from src.strategies.ml_models import XGBoostModel, load_processed_data, find_processed_file
from src.strategies.ml_models import MLStrategy


def load_model_for_ticker(ticker: str, model_dir: str = "models") -> Optional[XGBoostModel]:
    # Load trained model for a ticker
    model_path = Path(model_dir) / f"model_{ticker}.pkl"
    if not model_path.exists():
        return None
    
    model = XGBoostModel()
    model.load_model(str(model_path))
    return model


def run_backtest_ml(processed_file: str,
                    ticker: str,
                    initial_cash: float = 100000.0,
                    model_dir: str = "models",
                    lookback_window: int = 10,
                    min_confidence: float = 0.6,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> Dict:
    # Run backtest using saved ML model
    print(f"\n{'='*60}")
    print(f"Running Backtest for {ticker}")
    print(f"{'='*60}")
    
    # Load model
    print(f"Loading model for {ticker}...")
    model = load_model_for_ticker(ticker, model_dir)
    if model is None:
        raise FileNotFoundError(f"Model not found for {ticker}. Please train the model first.")
    
    print(f"Model loaded successfully ({len(model.feature_names)} features)")
    
    # Load data
    print(f"Loading processed data...")
    data = load_processed_data(processed_file, ticker=ticker)
    
    if len(data) == 0:
        raise ValueError(f"No data found for {ticker}")
    
    # Filter by date range if provided
    if start_date:
        data = data[data['datetime_et'] >= pd.to_datetime(start_date)]
    if end_date:
        data = data[data['datetime_et'] <= pd.to_datetime(end_date)]
    
    if len(data) == 0:
        raise ValueError("No data in specified date range")
    
    print(f"Data loaded: {len(data):,} rows")
    print(f"Date range: {data['datetime_et'].min()} to {data['datetime_et'].max()}")
    
    # Initialize strategy
    strategy = MLStrategy(
        name=f"MLStrategy_{ticker}",
        parameters={
            'lookback_window': lookback_window,
            'min_confidence': min_confidence
        }
    )
    strategy.model = model
    strategy.model_trained = True
    # Set training_data to prevent retraining (use a dummy value)
    strategy.training_data = pd.DataFrame()  # Empty DataFrame prevents retraining check
    
    # Initialize simulator
    trader = MockTrader(initial_cash=initial_cash)
    
    # Run backtest
    print(f"\nRunning backtest simulation...")
    print(f"Processing {len(data)} time periods...")
    
    # Process data in chunks (sliding window)
    chunk_size = lookback_window + 50  # Need enough data for features
    
    # Optimization: process every N minutes to speed up (adjust step_size as needed)
    # step_size = 1 means every minute, step_size = 5 means every 5 minutes
    # Use larger step size for faster processing (max 5000 iterations)
    step_size = max(1, int(len(data) / 5000))  # Auto-adjust: max 5k iterations
    snapshot_frequency = max(1, step_size * 10)  # Record snapshot every 10 steps
    
    print(f"Using step size: {step_size} (processing every {step_size} minute(s))")
    print(f"Snapshot frequency: every {snapshot_frequency} periods")
    
    for i in range(chunk_size, len(data), step_size):
        # Get current data window
        current_data = data.iloc[i-chunk_size:i+1]
        current_row = data.iloc[i]
        current_price = current_row['close']
        timestamp = current_row['datetime_et']
        
        # Get current position
        position = trader.get_position(ticker)
        current_position = {
            'symbol': ticker,
            'quantity': position.quantity if position else 0,
            'avg_cost': position.avg_cost if position else 0.0
        }
        
        # Generate signal
        signal = strategy.generate_signal(current_data.copy(), current_position)
        
        # Create order if signal is strong enough
        if signal['action'] != 'HOLD' and signal.get('confidence', 0) >= min_confidence:
            portfolio_state = {
                'symbol': ticker,
                'current_price': current_price,
                'cash': trader.get_cash_balance(),
                'positions': {ticker: current_position}
            }
            
            order = strategy.create_order(signal, portfolio_state)
            
            if order:
                # Execute order
                result = trader.execute_order(order)
                if result['status'] == 'FILLED':
                    print(f"{timestamp}: {order['action']} {order['quantity']} shares @ ${order['price']:.2f}")
        
        # Update prices
        trader.update_prices({ticker: current_price})
        
        # Record snapshot periodically (not every iteration)
        if (i - chunk_size) % snapshot_frequency == 0:
            trader.record_portfolio_snapshot(timestamp, {ticker: current_price})
        
        # Progress indicator
        if (i - chunk_size) % (1000 * step_size) == 0:
            portfolio_value = trader.get_portfolio_value({ticker: current_price})
            print(f"  Processed {i+1}/{len(data)} periods | Portfolio Value: ${portfolio_value:,.2f}")
    
    # Record final snapshot
    final_timestamp = data.iloc[-1]['datetime_et']
    final_price = data.iloc[-1]['close']
    trader.record_portfolio_snapshot(final_timestamp, {ticker: final_price})
    
    # Generate performance report
    print(f"\nGenerating performance report...")
    report = generate_performance_report(trader, trader.portfolio_history)
    
    # Add additional info
    report['ticker'] = ticker
    report['strategy'] = strategy.name
    report['data_periods'] = len(data)
    report['trades_executed'] = len(trader.get_trade_history())
    
    return {
        'trader': trader,
        'strategy': strategy,
        'report': report,
        'portfolio_history': trader.portfolio_history
    }


def run_multi_ticker_backtest(processed_file: str,
                              tickers: List[str],
                              initial_cash: float = 100000.0,
                              model_dir: str = "models",
                              **kwargs) -> Dict:
    # Run backtest for multiple tickers
    results = {}
    
    for ticker in tickers:
        try:
            result = run_backtest_ml(
                processed_file=processed_file,
                ticker=ticker,
                initial_cash=initial_cash,
                model_dir=model_dir,
                **kwargs
            )
            results[ticker] = result
        except Exception as e:
            print(f"Error backtesting {ticker}: {e}")
            results[ticker] = {'error': str(e)}
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run backtest using saved ML models')
    
    parser.add_argument('--processed-file', type=str, required=True,
                       help='Path to processed CSV file')
    parser.add_argument('--ticker', type=str, default=None,
                       help='Single ticker to backtest')
    parser.add_argument('--tickers', nargs='+', default=None,
                       help='Multiple tickers to backtest')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory containing trained models')
    parser.add_argument('--initial-cash', type=float, default=100000.0,
                       help='Starting capital')
    parser.add_argument('--lookback-window', type=int, default=10,
                       help='Lookback window for features')
    parser.add_argument('--min-confidence', type=float, default=0.6,
                       help='Minimum confidence threshold for trades')
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--output-dir', type=str, default='backtest_results',
                       help='Directory to save results')
    parser.add_argument('--plot', action='store_true',
                       help='Generate performance plots')
    
    args = parser.parse_args()
    
    # Handle glob patterns
    try:
        if '*' in args.processed_file or '?' in args.processed_file:
            args.processed_file = find_processed_file(args.processed_file)
            print(f"Using file: {args.processed_file}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Determine tickers
    if args.ticker:
        tickers = [args.ticker]
    elif args.tickers:
        tickers = args.tickers
    else:
        # Find all available models
        model_dir_path = Path(args.model_dir)
        if not model_dir_path.exists():
            print(f"Error: Model directory not found: {model_dir_path}")
            return
        
        model_files = list(model_dir_path.glob("model_*.pkl"))
        if not model_files:
            print(f"No model files found in {model_dir_path}")
            return
        
        tickers = [m.stem.replace("model_", "") for m in model_files]
        print(f"Found {len(tickers)} models. Backtesting all...")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run backtest(s)
    if len(tickers) == 1:
        # Single ticker
        result = run_backtest_ml(
            processed_file=args.processed_file,
            ticker=tickers[0],
            initial_cash=args.initial_cash,
            model_dir=args.model_dir,
            lookback_window=args.lookback_window,
            min_confidence=args.min_confidence,
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        # Print report
        print(f"\n{'='*60}")
        print("BACKTEST RESULTS")
        print(f"{'='*60}")
        report = result['report']
        for key, value in report.items():
            if isinstance(value, float):
                print(f"{key:25s}: {value:15.4f}")
            else:
                print(f"{key:25s}: {value}")
        
        # Save results
        output_file = output_dir / f"backtest_{tickers[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            # Convert non-serializable objects
            save_data = {
                'report': report,
                'portfolio_history': result['portfolio_history']
            }
            json.dump(save_data, f, indent=2, default=str)
        print(f"\nResults saved to: {output_file}")
        
        # Generate plot
        if args.plot:
            plot_file = output_dir / f"backtest_{tickers[0]}_plot.png"
            plot_performance_metrics(result['portfolio_history'], save_path=str(plot_file))
    else:
        # Multiple tickers
        results = run_multi_ticker_backtest(
            processed_file=args.processed_file,
            tickers=tickers,
            initial_cash=args.initial_cash,
            model_dir=args.model_dir,
            lookback_window=args.lookback_window,
            min_confidence=args.min_confidence,
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        # Print summary
        print(f"\n{'='*60}")
        print("BACKTEST SUMMARY")
        print(f"{'='*60}")
        print(f"{'Ticker':<10} {'Total Return':<15} {'Sharpe Ratio':<15} {'Max DD':<15} {'Win Rate':<15}")
        print("-" * 70)
        
        for ticker, result in results.items():
            if 'error' in result:
                print(f"{ticker:<10} ERROR: {result['error']}")
            else:
                report = result['report']
                print(f"{ticker:<10} {report.get('total_return', 0):>12.2f}%  "
                      f"{report.get('sharpe_ratio', 0):>12.4f}  "
                      f"{report.get('max_drawdown', 0):>12.2f}%  "
                      f"{report.get('win_rate', 0):>12.2f}%")
        
        # Save summary
        summary_file = output_dir / f"backtest_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary_data = {}
        for ticker, result in results.items():
            if 'error' not in result:
                summary_data[ticker] = result['report']
            else:
                summary_data[ticker] = {'error': result['error']}
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
