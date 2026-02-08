# Data collection runner with checkpoint/resume capability

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
from src.data_loader import DataLoader
from src.stock_selector import (
    get_tickers,
    load_selection,
    create_selection
)


class DataCollectionRunner:
    # Data collection runner with checkpoint/resume functionality
    
    def __init__(self, 
                 checkpoint_dir: str = "checkpoints",
                 raw_data_dir: str = "data/raw",
                 processed_data_dir: str = "data/processed",
                 api_base: Optional[str] = None,
                 api_key: Optional[str] = None):
        # Initialize runner with directories and API credentials
        self.checkpoint_dir = Path(checkpoint_dir)
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data loader
        self.loader = DataLoader(api_base=api_base, api_key=api_key)
        
        # Checkpoint files
        self.raw_checkpoint_file = self.checkpoint_dir / "raw_collection_checkpoint.json"
        self.processing_checkpoint_file = self.checkpoint_dir / "processing_checkpoint.json"
        
    def save_checkpoint(self, checkpoint_type: str, data: Dict) -> None:
        # Save checkpoint to file
        checkpoint_file = self.raw_checkpoint_file if checkpoint_type == 'raw' else self.processing_checkpoint_file
        
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        print(f"Checkpoint saved: {checkpoint_file}")
    
    def load_checkpoint(self, checkpoint_type: str) -> Optional[Dict]:
        # Load checkpoint from file
        checkpoint_file = self.raw_checkpoint_file if checkpoint_type == 'raw' else self.processing_checkpoint_file
        
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            print(f"Checkpoint loaded: {checkpoint_file}")
            print(f"  Timestamp: {checkpoint.get('timestamp', 'Unknown')}")
            return checkpoint.get('data', {})
        
        return None
    
    def clear_checkpoint(self, checkpoint_type: str) -> None:
        # Clear checkpoint file
        checkpoint_file = self.raw_checkpoint_file if checkpoint_type == 'raw' else self.processing_checkpoint_file
        
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print(f"Checkpoint cleared: {checkpoint_file}")
    
    def collect_raw_data(self,
                        tickers: List[str],
                        n_weekdays: int = 60,
                        bar_minutes: int = 1,
                        resume: bool = True,
                        output_filename: Optional[str] = None) -> pd.DataFrame:
        # Collect raw data for multiple tickers with checkpoint/resume capability
        print("=" * 80)
        print("RAW DATA COLLECTION PHASE")
        print("=" * 80)
        
        # Load checkpoint if resuming
        checkpoint = None
        completed_tickers = []
        all_raw_data = []
        
        if resume:
            checkpoint = self.load_checkpoint('raw')
            if checkpoint:
                completed_tickers = checkpoint.get('completed_tickers', [])
                print(f"Resuming: {len(completed_tickers)} tickers already completed")
                print(f"Completed tickers: {completed_tickers}")
        
        # Determine which tickers to process
        remaining_tickers = [t for t in tickers if t not in completed_tickers]
        
        if not remaining_tickers:
            print("All tickers already collected. Loading existing data...")
            # Load existing raw data
            if output_filename and os.path.exists(self.raw_data_dir / output_filename):
                return pd.read_csv(self.raw_data_dir / output_filename)
            return pd.DataFrame()
        
        print(f"\nProcessing {len(remaining_tickers)} ticker(s): {remaining_tickers}")
        print(f"Total tickers: {len(tickers)}, Completed: {len(completed_tickers)}, Remaining: {len(remaining_tickers)}")
        
        # Collect data for each ticker
        for i, ticker in enumerate(remaining_tickers, 1):
            print(f"\n[{i}/{len(remaining_tickers)}] Collecting data for {ticker}...")
            
            try:
                # Fetch data for this ticker
                ticker_data = self.loader.load_api(
                    tickers=[ticker],
                    n_weekdays=n_weekdays,
                    bar_minutes=bar_minutes
                )
                
                if ticker_data is not None and not ticker_data.empty:
                    all_raw_data.append(ticker_data)
                    completed_tickers.append(ticker)
                    
                    # Save checkpoint after each successful ticker
                    checkpoint_data = {
                        'completed_tickers': completed_tickers,
                        'n_weekdays': n_weekdays,
                        'bar_minutes': bar_minutes,
                        'total_tickers': len(tickers)
                    }
                    self.save_checkpoint('raw', checkpoint_data)
                    
                    print(f"{ticker}: {len(ticker_data)} rows collected")
                else:
                    print(f"{ticker}: No data returned")
                    
            except Exception as e:
                print(f"Error collecting {ticker}: {e}")
                print(f"  Checkpoint saved. Resume later to continue from {ticker}")
                # Save checkpoint even on error
                checkpoint_data = {
                    'completed_tickers': completed_tickers,
                    'n_weekdays': n_weekdays,
                    'bar_minutes': bar_minutes,
                    'total_tickers': len(tickers),
                    'last_error': str(e),
                    'failed_ticker': ticker
                }
                self.save_checkpoint('raw', checkpoint_data)
                continue
        
        # Combine all data
        if all_raw_data:
            combined_data = pd.concat(all_raw_data, ignore_index=True)
            
            # Save raw data
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"raw_data_{n_weekdays}wd_{bar_minutes}min_{timestamp}.csv"
            
            output_path = self.raw_data_dir / output_filename
            combined_data.to_csv(output_path, index=False)
            print(f"\nRaw data saved: {output_path}")
            print(f"  Total rows: {len(combined_data):,}")
            print(f"  Tickers: {combined_data['ticker'].nunique()}")
            
            # Update checkpoint with output filename
            checkpoint_data = {
                'completed_tickers': completed_tickers,
                'n_weekdays': n_weekdays,
                'bar_minutes': bar_minutes,
                'total_tickers': len(tickers),
                'output_file': output_filename,
                'total_rows': len(combined_data)
            }
            self.save_checkpoint('raw', checkpoint_data)
            
            return combined_data
        else:
            print("\nNo data collected")
            return pd.DataFrame()
    
    def process_data(self,
                    raw_data_file: Optional[str] = None,
                    resume: bool = True,
                    output_filename: Optional[str] = None) -> pd.DataFrame:
        # Process raw data (clean and add features) with checkpoint capability
        print("=" * 80)
        print("DATA PROCESSING PHASE")
        print("=" * 80)
        
        # Determine input file
        if raw_data_file is None:
            # Try to get from checkpoint
            checkpoint = self.load_checkpoint('raw')
            if checkpoint and 'output_file' in checkpoint:
                raw_data_file = checkpoint['output_file']
                print(f"Using raw data file from checkpoint: {raw_data_file}")
            else:
                # Find most recent raw data file
                raw_files = list(self.raw_data_dir.glob("raw_data_*.csv"))
                if raw_files:
                    raw_data_file = max(raw_files, key=os.path.getctime).name
                    print(f"Using most recent raw data file: {raw_data_file}")
                else:
                    raise FileNotFoundError("No raw data file found and no checkpoint available")
        
        raw_data_path = self.raw_data_dir / raw_data_file
        
        if not raw_data_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {raw_data_path}")
        
        print(f"\nLoading raw data from: {raw_data_path}")
        raw_data = pd.read_csv(raw_data_path)
        print(f"Loaded {len(raw_data):,} rows")
        print(f"Tickers: {raw_data['ticker'].unique()}")
        
        # Load processing checkpoint if resuming
        checkpoint = None
        processed_tickers = []
        
        if resume:
            checkpoint = self.load_checkpoint('processing')
            if checkpoint:
                processed_tickers = checkpoint.get('processed_tickers', [])
                print(f"\nResuming: {len(processed_tickers)} tickers already processed")
        
        # Determine which tickers to process
        all_tickers = sorted(raw_data['ticker'].unique())
        remaining_tickers = [t for t in all_tickers if t not in processed_tickers]
        
        if not remaining_tickers:
            print("All tickers already processed. Loading existing processed data...")
            # Try to load existing processed data
            if output_filename and os.path.exists(self.processed_data_dir / output_filename):
                return pd.read_csv(self.processed_data_dir / output_filename)
            # Generate output filename from checkpoint
            if checkpoint and 'output_file' in checkpoint:
                processed_file = self.processed_data_dir / checkpoint['output_file']
                if processed_file.exists():
                    return pd.read_csv(processed_file)
            return pd.DataFrame()
        
        print(f"\nProcessing {len(remaining_tickers)} ticker(s): {remaining_tickers}")
        
        # Process data for each ticker
        processed_frames = []
        
        for i, ticker in enumerate(remaining_tickers, 1):
            print(f"\n[{i}/{len(remaining_tickers)}] Processing {ticker}...")
            
            try:
                # Get ticker data
                ticker_raw = raw_data[raw_data['ticker'] == ticker].copy()
                
                # Load into loader
                self.loader.raw_data = ticker_raw
                
                # Clean data
                ticker_cleaned = self.loader.clean_data()
                
                if ticker_cleaned is not None and not ticker_cleaned.empty:
                    # Add technical features
                    ticker_processed = self.loader.add_technical_features()
                    processed_frames.append(ticker_processed)
                    processed_tickers.append(ticker)
                    
                    # Save checkpoint after each successful ticker
                    checkpoint_data = {
                        'processed_tickers': processed_tickers,
                        'raw_data_file': raw_data_file,
                        'total_tickers': len(all_tickers)
                    }
                    self.save_checkpoint('processing', checkpoint_data)
                    
                    print(f"{ticker}: {len(ticker_processed)} rows processed")
                else:
                    print(f"{ticker}: No data after cleaning")
                    
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                import traceback
                traceback.print_exc()
                # Save checkpoint even on error
                checkpoint_data = {
                    'processed_tickers': processed_tickers,
                    'raw_data_file': raw_data_file,
                    'total_tickers': len(all_tickers),
                    'last_error': str(e),
                    'failed_ticker': ticker
                }
                self.save_checkpoint('processing', checkpoint_data)
                continue
        
        # Combine all processed data
        if processed_frames:
            combined_processed = pd.concat(processed_frames, ignore_index=True)
            
            # Save processed data
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = Path(raw_data_file).stem
                output_filename = f"processed_{base_name}_{timestamp}.csv"
            
            output_path = self.processed_data_dir / output_filename
            combined_processed.to_csv(output_path, index=False)
            print(f"\nProcessed data saved: {output_path}")
            print(f"  Total rows: {len(combined_processed):,}")
            print(f"  Tickers: {combined_processed['ticker'].nunique()}")
            
            # Update checkpoint
            checkpoint_data = {
                'processed_tickers': processed_tickers,
                'raw_data_file': raw_data_file,
                'output_file': output_filename,
                'total_tickers': len(all_tickers),
                'total_rows': len(combined_processed)
            }
            self.save_checkpoint('processing', checkpoint_data)
            
            return combined_processed
        else:
            print("\nNo data processed")
            return pd.DataFrame()
    
    def get_status(self) -> Dict:
        """
        Get current status of data collection and processing.
        
        Returns:
            Dictionary with status information
        """
        status = {
            'raw_collection': {},
            'processing': {}
        }
        
        # Raw collection status
        raw_checkpoint = self.load_checkpoint('raw')
        if raw_checkpoint:
            status['raw_collection'] = {
                'completed_tickers': raw_checkpoint.get('completed_tickers', []),
                'total_tickers': raw_checkpoint.get('total_tickers', 0),
                'output_file': raw_checkpoint.get('output_file'),
                'last_error': raw_checkpoint.get('last_error'),
                'failed_ticker': raw_checkpoint.get('failed_ticker')
            }
        
        # Processing status
        proc_checkpoint = self.load_checkpoint('processing')
        if proc_checkpoint:
            status['processing'] = {
                'processed_tickers': proc_checkpoint.get('processed_tickers', []),
                'raw_data_file': proc_checkpoint.get('raw_data_file'),
                'output_file': proc_checkpoint.get('output_file'),
                'total_tickers': proc_checkpoint.get('total_tickers', 0),
                'last_error': proc_checkpoint.get('last_error'),
                'failed_ticker': proc_checkpoint.get('failed_ticker')
            }
        
        return status
    
    def print_status(self) -> None:
        """Print current status."""
        status = self.get_status()
        
        print("=" * 80)
        print("DATA COLLECTION STATUS")
        print("=" * 80)
        
        # Raw collection status
        if status['raw_collection']:
            raw = status['raw_collection']
            print("\n📥 Raw Data Collection:")
            print(f"  Completed tickers: {len(raw.get('completed_tickers', []))} / {raw.get('total_tickers', 0)}")
            if raw.get('completed_tickers'):
                print(f"  Tickers: {raw['completed_tickers']}")
            if raw.get('output_file'):
                print(f"  Output file: {raw['output_file']}")
            if raw.get('last_error'):
                print(f"  Last error: {raw['last_error']}")
                print(f"  Failed at ticker: {raw.get('failed_ticker', 'Unknown')}")
        else:
            print("\n📥 Raw Data Collection: Not started")
        
        # Processing status
        if status['processing']:
            proc = status['processing']
            print("\nData Processing:")
            print(f"  Processed tickers: {len(proc.get('processed_tickers', []))} / {proc.get('total_tickers', 0)}")
            if proc.get('processed_tickers'):
                print(f"  Tickers: {proc['processed_tickers']}")
            if proc.get('raw_data_file'):
                print(f"  Raw data file: {proc['raw_data_file']}")
            if proc.get('output_file'):
                print(f"  Output file: {proc['output_file']}")
            if proc.get('last_error'):
                print(f"  Last error: {proc['last_error']}")
                print(f"  Failed at ticker: {proc.get('failed_ticker', 'Unknown')}")
        else:
            print("\nData Processing: Not started")


def main():
    """
    Main function to run data collection and processing.
    
    Usage:
        # Collect raw data only
        python run_data_collection.py --collect-raw --tickers AAPL MSFT GOOGL
        
        # Process raw data only
        python run_data_collection.py --process --raw-file raw_data_*.csv
        
        # Full pipeline
        python run_data_collection.py --full --tickers AAPL MSFT GOOGL
        
        # Check status
        python run_data_collection.py --status
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Collection Runner with Checkpoints')
    parser.add_argument('--collect-raw', action='store_true', 
                       help='Collect raw data only')
    parser.add_argument('--process', action='store_true',
                       help='Process raw data only')
    parser.add_argument('--full', action='store_true',
                       help='Run full pipeline (collect + process)')
    parser.add_argument('--status', action='store_true',
                       help='Show current status')
    parser.add_argument('--tickers', nargs='+', 
                       help='List of ticker symbols')
    parser.add_argument('--selection-file', type=str,
                       help='Path to stock selection file (CSV or JSON) to get tickers from')
    parser.add_argument('--create-selection', action='store_true',
                       help='Create new stock selection with equal sector representation')
    parser.add_argument('--n-stocks', type=int, default=20,
                       help='Number of stocks to select (when using --create-selection, default: 20)')
    parser.add_argument('--selection-seed', type=int, default=42,
                       help='Random seed for stock selection (default: 42)')
    parser.add_argument('--raw-file', type=str,
                       help='Raw data file to process')
    parser.add_argument('--n-weekdays', type=int, default=60,
                       help='Number of weekdays to fetch (default: 60)')
    parser.add_argument('--bar-minutes', type=int, default=1,
                       help='Bar size in minutes (default: 1)')
    parser.add_argument('--api-base', type=str,
                       help='API base URL')
    parser.add_argument('--api-key', type=str,
                       help='API key')
    parser.add_argument('--no-resume', action='store_true',
                       help='Do not resume from checkpoint')
    parser.add_argument('--clear-checkpoints', action='store_true',
                       help='Clear all checkpoints')
    
    args = parser.parse_args()
    
    # Get API credentials from args or environment
    api_base = args.api_base or os.getenv('API_BASE')
    api_key = args.api_key or os.getenv('API_KEY')
    
    # Initialize runner
    runner = DataCollectionRunner(
        api_base=api_base,
        api_key=api_key
    )
    
    # Clear checkpoints if requested
    if args.clear_checkpoints:
        runner.clear_checkpoint('raw')
        runner.clear_checkpoint('processing')
        print("All checkpoints cleared")
        return
    
    # Show status
    if args.status:
        runner.print_status()
        return
    
    # Create stock selection if requested
    if args.create_selection:
        json_path = create_selection(
            n_stocks=args.n_stocks,
            random_seed=args.selection_seed
        )
        
        # If also collecting data, use the newly created selection
        if args.collect_raw or args.full:
            args.selection_file = json_path
    
    # Determine tickers to use
    tickers_to_use = None
    if args.selection_file:
        tickers_to_use = get_tickers(args.selection_file)
        print(f"Loaded {len(tickers_to_use)} tickers from selection file")
    elif args.tickers:
        tickers_to_use = args.tickers
    
    # Collect raw data
    if args.collect_raw or args.full:
        if not tickers_to_use:
            print("Error: --tickers or --selection-file required for raw data collection")
            return
        
        # Check for API credentials before starting
        api_base = args.api_base or os.getenv('API_BASE')
        api_key = args.api_key or os.getenv('API_KEY')
        
        if not api_base or not api_key:
            print("\n" + "="*60)
            print("ERROR: API credentials are required for raw data collection")
            print("="*60)
            print("\nPlease provide API credentials using one of these methods:")
            print("\n1. Command-line arguments:")
            print("   --api-base <URL> --api-key <KEY>")
            print("\n2. Environment variables:")
            print("   set API_BASE=<URL>")
            print("   set API_KEY=<KEY>")
            print("\nExample:")
            print("   python run_data_collection.py --collect-raw --selection-file data/selections/selected_stocks_20_*.json \\")
            print("     --api-base https://api.polygon.io --api-key YOUR_API_KEY")
            print("\nOr set environment variables:")
            print("   set API_BASE=https://api.polygon.io")
            print("   set API_KEY=YOUR_API_KEY")
            print("   python run_data_collection.py --collect-raw --selection-file data/selections/selected_stocks_20_*.json")
            print("="*60 + "\n")
            return
        
        print(f"\nStarting raw data collection for {len(tickers_to_use)} ticker(s)...")
        raw_data = runner.collect_raw_data(
            tickers=tickers_to_use,
            n_weekdays=args.n_weekdays,
            bar_minutes=args.bar_minutes,
            resume=not args.no_resume
        )
        print(f"\nRaw data collection completed: {len(raw_data):,} rows")
    
    # Process data
    if args.process or args.full:
        print(f"\nStarting data processing...")
        processed_data = runner.process_data(
            raw_data_file=args.raw_file,
            resume=not args.no_resume
        )
        print(f"\nData processing completed: {len(processed_data):,} rows")
    
    # Show final status
    runner.print_status()


if __name__ == "__main__":
    main()
