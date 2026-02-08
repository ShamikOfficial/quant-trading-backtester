# Script to train ML models on processed data

import argparse
from pathlib import Path
from src.strategies.ml_models import (
    train_model_on_processed_data,
    train_models_for_all_tickers,
    predict_from_processed_data,
    load_processed_data,
    XGBoostModel
)


def main():
    parser = argparse.ArgumentParser(description='Train and use ML models on processed data')
    
    parser.add_argument('--processed-file', type=str, required=True,
                       help='Path to processed CSV file')
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'train-all'],
                       default='train-all',
                       help='Mode: train (single ticker), train-all (all tickers), predict')
    parser.add_argument('--ticker', type=str, default=None,
                       help='Specific ticker (for train or predict mode)')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory to save/load models')
    parser.add_argument('--model-file', type=str, default=None,
                       help='Specific model file to load (for predict mode)')
    parser.add_argument('--lookback-window', type=int, default=10,
                       help='Lookback window for features')
    parser.add_argument('--n-estimators', type=int, default=100,
                       help='Number of XGBoost estimators')
    parser.add_argument('--max-depth', type=int, default=5,
                       help='Max tree depth')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                       help='Learning rate')
    parser.add_argument('--n-predictions', type=int, default=1,
                       help='Number of predictions to generate (for predict mode)')
    
    args = parser.parse_args()
    
    # Check if processed file exists (handle glob patterns)
    try:
        from src.strategies.ml_models import find_processed_file
        if '*' in args.processed_file or '?' in args.processed_file:
            args.processed_file = find_processed_file(args.processed_file)
            print(f"Using file: {args.processed_file}")
        elif not Path(args.processed_file).exists():
            print(f"Error: Processed file not found: {args.processed_file}")
            return
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    model_params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate
    }
    
    if args.mode == 'train':
        # Train single model
        if not args.ticker:
            print("Error: --ticker required for train mode")
            return
        
        print(f"Training model for {args.ticker}...")
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)
        
        model_path = Path(args.model_dir) / f"model_{args.ticker}.pkl"
        
        try:
            model, metrics = train_model_on_processed_data(
                processed_file=args.processed_file,
                ticker=args.ticker,
                lookback_window=args.lookback_window,
                model_params=model_params,
                save_model_path=str(model_path)
            )
            print(f"\nModel training completed for {args.ticker}")
            print(f"  Model saved to: {model_path}")
        except Exception as e:
            print(f"Error training model: {e}")
            import traceback
            traceback.print_exc()
    
    elif args.mode == 'train-all':
        # Train models for all tickers
        print("Training models for all tickers...")
        try:
            results = train_models_for_all_tickers(
                processed_file=args.processed_file,
                model_dir=args.model_dir,
                lookback_window=args.lookback_window,
                model_params=model_params
            )
            
            print(f"\n{'='*60}")
            print("Training Summary")
            print(f"{'='*60}")
            successful = 0
            failed = 0
            
            for ticker, metrics in results.items():
                if 'error' not in metrics:
                    print(f"{ticker:8s}: Test R² = {metrics.get('test_r2', 0):7.4f}, "
                          f"Test RMSE = {metrics.get('test_rmse', 0):.6f}")
                    successful += 1
                else:
                    print(f"{ticker:8s}: ERROR - {metrics['error']}")
                    failed += 1
            
            print(f"\nTotal: {successful} successful, {failed} failed")
        except Exception as e:
            print(f"Error training models: {e}")
            import traceback
            traceback.print_exc()
    
    elif args.mode == 'predict':
        # Generate predictions
        if not args.ticker:
            print("Error: --ticker required for predict mode")
            return
        
        # Determine model file
        if args.model_file:
            model_path = args.model_file
        else:
            model_path = Path(args.model_dir) / f"model_{args.ticker}.pkl"
        
        if not Path(model_path).exists():
            print(f"Error: Model file not found: {model_path}")
            print("Please train the model first using --mode train")
            return
        
        print(f"Loading model from: {model_path}")
        model = XGBoostModel()
        model.load_model(str(model_path))
        
        print(f"Generating predictions for {args.ticker}...")
        try:
            predictions = predict_from_processed_data(
                model=model,
                processed_file=args.processed_file,
                ticker=args.ticker,
                lookback_window=args.lookback_window,
                n_predictions=args.n_predictions
            )
            
            print("\nPredictions:")
            print("="*60)
            print(predictions.to_string(index=False))
        except Exception as e:
            print(f"Error generating predictions: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
