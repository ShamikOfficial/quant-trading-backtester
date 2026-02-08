# Data loading and preprocessing module

import pandas as pd
import numpy as np
import requests
import time
from typing import Optional, List, Dict
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo


class DataLoader:
    # Handles data loading, cleaning, and streaming
    
    def __init__(self, data_source: Optional[str] = None, 
                 api_base: Optional[str] = None,
                 api_key: Optional[str] = None):
        # Initialize DataLoader with data source and API credentials
        self.data_source = data_source
        self.api_base = api_base
        self.api_key = api_key
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        
        # Exchange timezone (US/Eastern for US markets)
        self.tz_et = ZoneInfo("America/New_York")
        
        # Regular Trading Hours (RTH)
        self.rth_start = (9, 30)
        self.rth_end = (16, 0)
        
    def load_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            **kwargs: Additional arguments to pass to pd.read_csv()
            
        Returns:
            DataFrame containing the loaded data
        """
        self.raw_data = pd.read_csv(file_path, **kwargs)
        return self.raw_data
    
    def _is_weekday(self, d: date) -> bool:
        # Check if date is a weekday
        return d.weekday() < 5
    
    def _previous_weekdays(self, n_weekdays: int) -> List[str]:
        # Get list of previous n weekdays
        weekdays = []
        current_date = datetime.now().date() - timedelta(days=1)
        while len(weekdays) < n_weekdays:
            if self._is_weekday(current_date):
                weekdays.append(current_date.isoformat())
            current_date -= timedelta(days=1)
        return weekdays
    
    def _fetch_minute_bars(self, ticker: str, n_minutes: int, 
                          start: str, end: str, 
                          max_retries: int = 3) -> List[Dict]:
        # Fetch n-minute aggregate bars from API
        if not self.api_base or not self.api_key:
            raise ValueError("API base URL and API key must be provided")
        
        url = f"{self.api_base}/v2/aggs/ticker/{ticker}/range/{n_minutes}/minute/{start}/{end}"
        params = {
            "apiKey": self.api_key,
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000
        }
        
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, params=params, timeout=30)
                
                if resp.status_code == 200:
                    payload = resp.json()
                    return payload.get("results", []) or []
                elif resp.status_code == 429:
                    wait_time = 12.5 * (2 ** attempt)
                    print(f"Rate limited. Waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    print(f"API error {resp.status_code}: {resp.text[:200]}")
                    return []
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        
        return []
    
    def _rows_to_df(self, rows: List[Dict], ticker: str) -> pd.DataFrame:
        """Convert API response rows to DataFrame."""
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows).rename(columns={
            "t": "t_ms",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "vw": "vwap",
            "n": "num_trades",
        })
        df["ticker"] = ticker
        
        keep = ["ticker", "t_ms", "open", "high", "low", "close", "volume", "vwap", "num_trades"]
        for col in keep:
            if col not in df.columns:
                df[col] = np.nan
        
        return df[keep]
    
    def load_api(self, tickers: List[str], 
                 n_weekdays: int = 60, 
                 bar_minutes: int = 1) -> pd.DataFrame:
        """
        Load minute-bar data for multiple tickers from API.
        
        Args:
            tickers: List of ticker symbols
            n_weekdays: Number of weekdays to fetch
            bar_minutes: Bar size in minutes (default: 1)
            
        Returns:
            DataFrame with raw data
        """
        if not tickers:
            self.raw_data = pd.DataFrame()
            return self.raw_data
        
        if not self.api_base or not self.api_key:
            raise ValueError("API base URL and API key must be set in __init__")
        
        dates = self._previous_weekdays(n_weekdays)
        start_date = dates[-1]
        end_date = dates[0]
        
        frames = []
        for ticker in tickers:
            rows = self._fetch_minute_bars(ticker, bar_minutes, start_date, end_date)
            if rows:
                df = self._rows_to_df(rows, ticker)
                frames.append(df)
            else:
                pass  # No data available
            time.sleep(0.5)  # Rate limiting
        
        self.raw_data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return self.raw_data
    
    def clean_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Clean and preprocess the loaded data.
        
        Cleaning steps:
        1) Convert timestamps to US/Eastern timezone
        2) Filter to Regular Trading Hours (09:30-16:00 ET)
        3) Sort by time and remove duplicates
        4) Reindex to continuous 1-min grid per (ticker, day) - only for actual data range
        5) Impute missing values only for edge cases (start/end of raw data):
           - Price columns: impute at start/end edges, then forward-fill/back-fill for small gaps
           - Volume/num_trades: fill edge cases with 0, then fill remaining gaps with 0
        
        Args:
            df: Optional DataFrame to clean. If None, uses self.raw_data
            
        Returns:
            Cleaned DataFrame
        """
        if df is None:
            if self.raw_data is None:
                raise ValueError("No data to clean. Call load_csv or load_api first.")
            df = self.raw_data
        
        if df.empty:
            self.processed_data = df.copy()
            return self.processed_data
        
        out = df.copy()
        
        # 1) Convert timestamps to datetime, then to US/Eastern
        if "t_ms" in out.columns:
            out["datetime_utc"] = pd.to_datetime(out["t_ms"], unit="ms", utc=True)
        elif "datetime_utc" in out.columns:
            out["datetime_utc"] = pd.to_datetime(out["datetime_utc"], utc=True)
        elif "datetime" in out.columns:
            out["datetime_utc"] = pd.to_datetime(out["datetime"], utc=True)
        else:
            raise ValueError("Data must contain 't_ms', 'datetime_utc', or 'datetime' column")
        
        out["datetime_et"] = out["datetime_utc"].dt.tz_convert(self.tz_et)
        out["date_et"] = out["datetime_et"].dt.date
        
        # 2) Filter Regular Trading Hours (RTH)
        rth_start_time = datetime(2000, 1, 1, self.rth_start[0], self.rth_start[1]).time()
        rth_end_time = datetime(2000, 1, 1, self.rth_end[0], self.rth_end[1]).time()
        time_col = out["datetime_et"].dt.time
        out = out[(time_col >= rth_start_time) & (time_col < rth_end_time)].copy()
        
        if out.empty:
            self.processed_data = out
            return out
        
        # 3) Sort and drop duplicates
        out = out.sort_values(["ticker", "datetime_et"]).drop_duplicates(
            ["ticker", "datetime_et"], keep="last"
        )
        
        price_cols = ["open", "high", "low", "close", "vwap"]
        vol_cols = ["volume", "num_trades"]
        
        # 4) Reindex to continuous 1-min grid per (ticker, day) - only for actual data range
        cleaned_frames = []
        for (ticker, day), group in out.groupby(["ticker", "date_et"], sort=False):
            group = group.sort_values("datetime_et").copy()
            
            # Get actual data range (not full trading day)
            actual_start = group["datetime_et"].min().floor("min")
            actual_end = group["datetime_et"].max().floor("min")
            
            # Build continuous 1-min grid only for the actual data range
            minute_grid = pd.date_range(start=actual_start, end=actual_end, freq="1min", inclusive="both")
            
            # Reindex to grid (this will create gaps for missing minutes, but won't extend beyond data)
            group = group.set_index(group["datetime_et"].dt.floor("min")).reindex(minute_grid)
            
            group["ticker"] = ticker
            group["datetime_et"] = group.index
            group["datetime_utc"] = group["datetime_et"].dt.tz_convert("UTC")
            group["date_et"] = day
            
            # Ensure numeric types
            for col in price_cols + vol_cols:
                if col in group.columns:
                    group[col] = pd.to_numeric(group[col], errors="coerce")
            
            # 5) Missing value imputation - only for edge cases (start/end of raw data)
            # Only impute at the very first and very last timestamps if they're missing
            for col in price_cols:
                if col in group.columns:
                    # Find first and last valid values
                    valid_mask = group[col].notna()
                    if valid_mask.any():
                        first_valid_idx = group.index[valid_mask].min()
                        last_valid_idx = group.index[valid_mask].max()
                        
                        # Only impute at the very start if missing (edge case at start of raw data)
                        if group.index[0] < first_valid_idx:
                            group.loc[group.index[0]:first_valid_idx, col] = group.loc[first_valid_idx, col]
                        
                        # Only impute at the very end if missing (edge case at end of raw data)
                        if group.index[-1] > last_valid_idx:
                            group.loc[last_valid_idx:group.index[-1], col] = group.loc[last_valid_idx, col]
                        
                        # For small gaps in the middle (1-2 minutes), forward fill then back fill
                        # This handles minor data collection gaps but doesn't extend beyond actual data
                        group[col] = group[col].ffill().bfill()
                    else:
                        # If no valid data at all, skip this column
                        pass
            
            # Volume/trades: fill edge cases with 0, then fill small gaps
            for col in vol_cols:
                if col in group.columns:
                    # Find first and last valid values
                    valid_mask = group[col].notna()
                    if valid_mask.any():
                        first_valid_idx = group.index[valid_mask].min()
                        last_valid_idx = group.index[valid_mask].max()
                        
                        # Only fill edge cases with 0 if completely missing at start/end
                        if group.index[0] < first_valid_idx:
                            group.loc[group.index[0]:first_valid_idx, col] = 0.0
                        if group.index[-1] > last_valid_idx:
                            group.loc[last_valid_idx:group.index[-1], col] = 0.0
                    
                    # Fill remaining NaN with 0 (for small gaps in middle, but won't extend beyond data range)
                    group[col] = group[col].fillna(0.0)
            
            cleaned_frames.append(group.reset_index(drop=True))
        
        cleaned = pd.concat(cleaned_frames, ignore_index=True)
        cleaned = cleaned.sort_values(["ticker", "datetime_et"]).reset_index(drop=True)
        
        # Keep standard columns
        keep_cols = [
            "ticker", "datetime_et", "datetime_utc", "date_et",
            "open", "high", "low", "close", "volume", "vwap", "num_trades"
        ]
        for col in keep_cols:
            if col not in cleaned.columns:
                cleaned[col] = np.nan
        
        cleaned = cleaned[[c for c in keep_cols if c in cleaned.columns]]
        
        self.processed_data = cleaned
        return cleaned
    
    def normalize_data(self, df: Optional[pd.DataFrame] = None, 
                      columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Normalize data columns using min-max scaling.
        
        Args:
            df: Optional DataFrame to normalize. If None, uses self.processed_data
            columns: List of column names to normalize. If None, normalizes all numeric columns
            
        Returns:
            Normalized DataFrame
        """
        if df is None:
            df = self.processed_data.copy() if self.processed_data is not None else pd.DataFrame()
        
        if df.empty:
            return df
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        normalized_df = df.copy()
        for col in columns:
            if col in normalized_df.columns:
                col_min = normalized_df[col].min()
                col_max = normalized_df[col].max()
                if col_max > col_min:
                    normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
        
        return normalized_df
    
    def stream_data(self, chunk_size: int = 100):
        """
        Stream data in chunks for real-time processing.
        
        Args:
            chunk_size: Number of rows per chunk
            
        Yields:
            DataFrame chunks
        """
        df = self.processed_data if self.processed_data is not None else self.raw_data
        if df is None or df.empty:
            return
        
        for i in range(0, len(df), chunk_size):
            yield df.iloc[i:i + chunk_size]
    
    def get_latest_data(self, n_rows: int = 1) -> pd.DataFrame:
        """
        Get the most recent n rows of data.
        
        Args:
            n_rows: Number of recent rows to retrieve
            
        Returns:
            DataFrame with the latest n rows
        """
        df = self.processed_data if self.processed_data is not None else self.raw_data
        if df is None or df.empty:
            return pd.DataFrame()
        
        return df.tail(n_rows)
    
    def add_technical_features(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Add technical indicator features to the dataset.
        
        Args:
            df: Optional DataFrame. If None, uses self.processed_data
            
        Returns:
            DataFrame with added technical features
        """
        from src.strategies.technical_indicators import (
            calculate_sma, calculate_ema, calculate_rsi
        )
        
        if df is None:
            df = self.processed_data.copy() if self.processed_data is not None else pd.DataFrame()
        
        if df.empty or 'close' not in df.columns:
            return df
        
        result_df = df.copy()
        
        # Add moving averages
        result_df['sma_20'] = calculate_sma(result_df['close'], 20)
        result_df['sma_50'] = calculate_sma(result_df['close'], 50)
        result_df['ema_12'] = calculate_ema(result_df['close'], 12)
        result_df['ema_26'] = calculate_ema(result_df['close'], 26)
        
        # Add RSI
        result_df['rsi_14'] = calculate_rsi(result_df['close'], 14)
        
        # Add returns
        result_df['returns'] = result_df['close'].pct_change()
        result_df['log_returns'] = np.log(result_df['close'] / result_df['close'].shift(1))
        
        return result_df
