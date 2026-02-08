# Stock selector for S&P 500 with equal sector representation

import warnings
import sys
from io import StringIO, BytesIO

# Suppress all warnings at module level
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

import pandas as pd
import requests
import json
from datetime import datetime
from pathlib import Path
import numpy as np


def fetch_sp500_tickers():
    # Fetch S&P 500 tickers and sectors from Wikipedia
    wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    # Suppress stdout/stderr during HTML parsing
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    
    try:
        response = requests.get(wiki_url, headers=headers, timeout=10)
        html = response.content
        
        # Parse HTML - try different methods
        try:
            # Method 1: Use lxml with displayed_only=False
            df = pd.read_html(BytesIO(html), flavor='lxml', displayed_only=False)[0]
        except (ImportError, ValueError, IndexError):
            try:
                # Method 2: Use html5lib
                df = pd.read_html(BytesIO(html), flavor='html5lib', displayed_only=False)[0]
            except (ImportError, ValueError, IndexError):
                # Method 3: Try without displayed_only
                try:
                    df = pd.read_html(BytesIO(html), flavor='lxml')[0]
                except:
                    df = pd.read_html(BytesIO(html), flavor='html5lib')[0]
    finally:
        # Always restore stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    df = df.rename(columns={
        "Symbol": "ticker",
        "Security": "name",
        "GICS Sector": "sector"
    })
    
    return df[['ticker', 'name', 'sector']].dropna(subset=['ticker', 'sector'])


def select_stocks_equal_sectors(n_stocks, sp500_df=None, random_seed=42):
    """
    Select stocks with equal representation from each sector.
    
    Args:
        n_stocks: Total number of stocks to select
        sp500_df: Optional pre-fetched DataFrame
        random_seed: Random seed for reproducibility
    """
    if sp500_df is None:
        sp500_df = fetch_sp500_tickers()
    
    sectors = sorted(sp500_df['sector'].unique())
    n_sectors = len(sectors)
    stocks_per_sector = n_stocks // n_sectors
    remainder = n_stocks % n_sectors
    
    np.random.seed(random_seed)
    selected_stocks = []
    
    for i, sector in enumerate(sectors):
        sector_stocks = sp500_df[sp500_df['sector'] == sector]
        n_select = stocks_per_sector + (1 if i < remainder else 0)
        n_select = min(n_select, len(sector_stocks))
        
        if n_select > 0:
            selected = sector_stocks.sample(n=n_select, random_state=random_seed + i)
            selected_stocks.append(selected)
    
    result = pd.concat(selected_stocks, ignore_index=True)
    return result.sort_values(['sector', 'ticker']).reset_index(drop=True)


def save_selection_json(selected_stocks, output_dir="data/selections", filename=None):
    """Save selection to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        n_stocks = len(selected_stocks)
        filename = f"selected_stocks_{n_stocks}_{timestamp}.json"
    
    json_path = output_path / filename
    
    json_data = {
        'metadata': {
            'selection_date': datetime.now().strftime('%Y-%m-%d'),
            'total_stocks': len(selected_stocks),
            'n_sectors': selected_stocks['sector'].nunique()
        },
        'stocks': selected_stocks[['ticker', 'name', 'sector']].to_dict('records'),
        'sector_distribution': selected_stocks['sector'].value_counts().to_dict()
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    return str(json_path)


def create_selection(n_stocks=20, random_seed=42, output_dir="data/selections"):
    """
    Main function: Create stock selection and save to JSON.
    
    Args:
        n_stocks: Number of stocks to select (default: 20)
        random_seed: Random seed (default: 42)
        output_dir: Output directory (default: data/selections)
    
    Returns:
        Path to saved JSON file
    """
    # Suppress output during fetching
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    
    try:
        sp500_df = fetch_sp500_tickers()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    selected_stocks = select_stocks_equal_sectors(n_stocks, sp500_df, random_seed)
    json_path = save_selection_json(selected_stocks, output_dir)
    
    # Verify file was created
    json_file = Path(json_path)
    if not json_file.exists():
        raise FileNotFoundError(f"JSON file was not created: {json_path}")
    
    print(f"Selected {len(selected_stocks)} stocks from {selected_stocks['sector'].nunique()} sectors")
    print(f"Saved to: {json_path}")
    
    return json_path


def find_selection_file(pattern_or_path, search_dir="data/selections"):
    """
    Find selection file by pattern or return exact path.
    Supports glob patterns like "selected_stocks_20_*.json"
    
    Args:
        pattern_or_path: File path or pattern (e.g., "selected_stocks_20_*.json")
        search_dir: Directory to search in
        
    Returns:
        Full path to the file
    """
    from glob import glob
    
    path = Path(pattern_or_path)
    
    # If it's an absolute path and exists, return it
    if path.is_absolute() and path.exists():
        return str(path)
    
    # Extract just the filename pattern (remove directory if present)
    if '/' in pattern_or_path or '\\' in pattern_or_path:
        # Pattern includes directory, extract filename
        filename_pattern = Path(pattern_or_path).name
        # Use the directory from pattern if absolute, otherwise use search_dir
        if path.is_absolute():
            search_dir = str(path.parent)
    else:
        filename_pattern = pattern_or_path
    
    # Replace .csv with .json if user specified csv
    if filename_pattern.endswith('.csv'):
        filename_pattern = filename_pattern.replace('.csv', '.json')
    
    # If it contains wildcards, search for matching files
    if '*' in filename_pattern or '?' in filename_pattern:
        search_path = Path(search_dir)
        if not search_path.exists():
            raise FileNotFoundError(f"Search directory does not exist: {search_dir}")
        
        # Try with .json extension if not specified
        if not filename_pattern.endswith('.json') and not filename_pattern.endswith('.csv'):
            filename_pattern = filename_pattern + '.json'
        
        # Search in the directory
        full_pattern = str(search_path / filename_pattern)
        matches = glob(full_pattern)
        
        if matches:
            # Return most recent file
            return max(matches, key=lambda p: Path(p).stat().st_mtime)
        else:
            raise FileNotFoundError(f"No files found matching pattern: {filename_pattern} in {search_dir}")
    
    # Try relative to search_dir
    full_path = Path(search_dir) / pattern_or_path
    if full_path.exists():
        return str(full_path)
    
    # Try as absolute path
    if path.exists():
        return str(path)
    
    # Try adding .json extension
    if not pattern_or_path.endswith('.json') and not pattern_or_path.endswith('.csv'):
        json_path = Path(search_dir) / (pattern_or_path + '.json')
        if json_path.exists():
            return str(json_path)
    
    # Try replacing .csv with .json
    if pattern_or_path.endswith('.csv'):
        json_path = Path(search_dir) / pattern_or_path.replace('.csv', '.json')
        if json_path.exists():
            return str(json_path)
    
    raise FileNotFoundError(f"Selection file not found: {pattern_or_path}")


def load_selection(json_path):
    """Load stock selection from JSON file."""
    # Resolve pattern if needed
    resolved_path = find_selection_file(json_path)
    
    with open(resolved_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data['stocks'])


def get_tickers(json_path):
    """Get list of tickers from JSON file."""
    df = load_selection(json_path)
    return df['ticker'].tolist()


if __name__ == "__main__":
    # Simple usage: just run the file
    # Modify n_stocks as needed
    try:
        create_selection(n_stocks=20, random_seed=42)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
