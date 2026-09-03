# Technical analysis indicators implementation

import pandas as pd
import numpy as np
from typing import Optional, Union
from src.strategies.base_strategy import BaseStrategy


def calculate_sma(closing_price_data: pd.Series, window: int) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).
    
    Args:
        data: Price series (typically closing prices)
        window: Number of periods for the moving average
        
    Returns:
        Series containing SMA values
    """
    return closing_price_data.rolling(window=window).mean()


def calculate_ema(closing_price_data: pd.Series, window: int, alpha: Optional[float] = None) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).

    Args:
        closing_price_data: Price series
        window: Number of periods for the moving average
        alpha: Smoothing factor (optional; derived from window if not provided)

    Returns:
        Series containing EMA values
    """
    if alpha is None:
        alpha = 2.0 / (window + 1.0)
    return closing_price_data.ewm(alpha=alpha, adjust=False).mean()


def calculate_rsi(closing_price_data: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    RSI ranges from 0 to 100. Values above 70 indicate overbought,
    values below 30 indicate oversold.
    
    Args:
        closing_price_data: Price series (typically closing prices)
        window: Number of periods for RSI calculation (default: 14)
        
    Returns:
        Series containing RSI values (0-100)
    """
    delta = closing_price_data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(closing_price_data: pd.Series, fast_window: int = 12, 
                   slow_window: int = 26, signal_window: int = 9) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        closing_price_data: Price series
        fast_window: Fast EMA period (default: 12)
        slow_window: Slow EMA period (default: 26)
        signal_window: Signal line EMA period (default: 9)
        
    Returns:
        DataFrame with columns: 'MACD', 'Signal', 'Histogram'
    """
    ema_fast = calculate_ema(closing_price_data, fast_window)
    ema_slow = calculate_ema(closing_price_data, slow_window)
    
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal_window)
    histogram = macd_line - signal_line
    
    result = pd.DataFrame({
        'MACD': macd_line,
        'Signal': signal_line,
        'Histogram': histogram
    })
    
    return result


def calculate_bollinger_bands(closing_price_data: pd.Series, window: int = 20, 
                              num_std: float = 2.0) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.
    
    Args:
        closing_price_data: Price series
        window: Number of periods for moving average (default: 20)
        num_std: Number of standard deviations for bands (default: 2.0)
        
    Returns:
        DataFrame with columns: 'Upper', 'Middle', 'Lower'
    """
    middle = calculate_sma(closing_price_data, window)
    std = closing_price_data.rolling(window=window).std()
    
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    
    result = pd.DataFrame({
        'Upper': upper,
        'Middle': middle,
        'Lower': lower
    })
    
    return result


def calculate_stochastic_oscillator(high: pd.Series, low: pd.Series, 
                                    close: pd.Series, k_window: int = 14, 
                                    d_window: int = 3) -> pd.DataFrame:
    """
    Calculate Stochastic Oscillator (%K and %D).
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        k_window: Period for %K calculation (default: 14)
        d_window: Period for %D smoothing (default: 3)
        
    Returns:
        DataFrame with columns: '%K', '%D'
    """
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_window).mean()
    
    result = pd.DataFrame({
        '%K': k_percent,
        '%D': d_percent
    })
    
    return result


class TechnicalStrategy(BaseStrategy):
    
    def __init__(self, name: str = "TechnicalStrategy", 
                 parameters: Optional[dict] = None):

        super().__init__(name, parameters)
        self.sma_window = self.parameters.get('sma_window', 20)
        self.rsi_window = self.parameters.get('rsi_window', 14)
        self.rsi_overbought = self.parameters.get('rsi_overbought', 70)
        self.rsi_oversold = self.parameters.get('rsi_oversold', 30)
    
    def generate_signal(self, data: pd.DataFrame, 
                       current_position: dict) -> dict:
        """
        Generate trading signals using technical indicators.
        
        Args:
            data: Current market data DataFrame
            current_position: Current portfolio position
            
        Returns:
            Signal dictionary
        """
        if 'close' not in data.columns:
            return {'action': 'HOLD', 'confidence': 0.0}
        
        close_prices = data['close']
        
        # Calculate indicators
        sma = calculate_sma(close_prices, self.sma_window)
        rsi = calculate_rsi(close_prices, self.rsi_window)
        
        # Get latest values
        current_price = close_prices.iloc[-1]
        current_sma = sma.iloc[-1]
        current_rsi = rsi.iloc[-1]
        
        # Check for NaN values
        if pd.isna(current_sma) or pd.isna(current_rsi):
            return {'action': 'HOLD', 'confidence': 0.0}
        
        # Signal generation logic
        signal = {'action': 'HOLD', 'confidence': 0.0}
        
        # RSI-based signals
        if current_rsi < self.rsi_oversold:
            signal = {
                'action': 'BUY',
                'confidence': min((self.rsi_oversold - current_rsi) / self.rsi_oversold, 1.0),
                'reason': 'RSI oversold'
            }
        elif current_rsi > self.rsi_overbought:
            signal = {
                'action': 'SELL',
                'confidence': min((current_rsi - self.rsi_overbought) / (100 - self.rsi_overbought), 1.0),
                'reason': 'RSI overbought'
            }
        
        # SMA crossover confirmation
        if len(sma) >= 2:
            prev_sma = sma.iloc[-2]
            if not pd.isna(prev_sma):
                if current_price > current_sma and current_price < prev_sma:
                    # Price crossed above SMA
                    if signal['action'] == 'BUY':
                        signal['confidence'] = min(signal['confidence'] + 0.2, 1.0)
                elif current_price < current_sma and current_price > prev_sma:
                    # Price crossed below SMA
                    if signal['action'] == 'SELL':
                        signal['confidence'] = min(signal['confidence'] + 0.2, 1.0)
        
        self.signals_history.append(signal)
        return signal
    
    def create_order(self, signal: dict, portfolio_state: dict) -> dict:
        """
        Create order from technical indicator signal.
        
        Args:
            signal: Signal dictionary
            portfolio_state: Current portfolio state
            
        Returns:
            Order dictionary
        """
        from datetime import datetime
        
        if signal['action'] == 'HOLD':
            return None
        
        # Get current price from portfolio state or signal
        current_price = portfolio_state.get('current_price', 0.0)
        symbol = portfolio_state.get('symbol', 'UNKNOWN')
        cash = portfolio_state.get('cash', 0.0)
        current_position = portfolio_state.get('positions', {}).get(symbol, {})
        current_quantity = current_position.get('quantity', 0)
        
        # Calculate order quantity based on confidence and available cash
        confidence = signal.get('confidence', 0.5)
        
        if signal['action'] == 'BUY':
            # Calculate how many shares we can buy
            if current_price > 0:
                # Use a percentage of cash based on confidence
                cash_to_use = cash * confidence * 0.1  # Use 10% of cash per signal, scaled by confidence
                quantity = int(cash_to_use / current_price)
            else:
                quantity = 0
            
            if quantity > 0:
                order = {
                    'symbol': symbol,
                    'action': 'BUY',
                    'quantity': quantity,
                    'price': current_price,
                    'timestamp': datetime.now()
                }
                self.orders_history.append(order)
                return order
        
        elif signal['action'] == 'SELL':
            # Sell based on current position
            if current_quantity > 0:
                # Sell a percentage based on confidence
                quantity_to_sell = int(current_quantity * confidence)
                if quantity_to_sell > 0:
                    order = {
                        'symbol': symbol,
                        'action': 'SELL',
                        'quantity': quantity_to_sell,
                        'price': current_price,
                        'timestamp': datetime.now()
                    }
                    self.orders_history.append(order)
                    return order
        
        return None
