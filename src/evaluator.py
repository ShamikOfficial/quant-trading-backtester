# Performance evaluation and metrics calculation

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from src.simulator import MockTrader, Trade


def calculate_sharpe_ratio(returns: pd.Series, 
                           risk_free_rate: float = 0.0,
                           periods_per_year: int = 252) -> float:
    # Calculate annualized Sharpe ratio
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    mean_return = returns.mean() * periods_per_year
    std_return = returns.std() * np.sqrt(periods_per_year)
    
    if std_return == 0:
        return 0.0
    
    return (mean_return - risk_free_rate) / std_return


def calculate_sortino_ratio(returns: pd.Series,
                            risk_free_rate: float = 0.0,
                            periods_per_year: int = 252) -> float:
    # Calculate annualized Sortino ratio
    if len(returns) == 0:
        return 0.0
    
    # Calculate downside deviation (only negative returns)
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0:
        return float('inf') if returns.mean() > 0 else 0.0
    
    downside_std = downside_returns.std() * np.sqrt(periods_per_year)
    
    if downside_std == 0:
        return 0.0
    
    mean_return = returns.mean() * periods_per_year
    return (mean_return - risk_free_rate) / downside_std


def calculate_max_drawdown(portfolio_values: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    # Calculate maximum drawdown
    if len(portfolio_values) == 0:
        return (0.0, None, None)
    
    # Calculate running maximum (peak)
    running_max = portfolio_values.expanding().max()
    
    # Calculate drawdown
    drawdown = (portfolio_values - running_max) / running_max
    
    # Find maximum drawdown
    max_dd_idx = drawdown.idxmin()
    max_dd = drawdown.min()
    
    # Find peak before drawdown
    if max_dd_idx is not None:
        peak_slice = running_max[:max_dd_idx]
        if len(peak_slice) > 0:
            peak_idx = peak_slice.idxmax()
        else:
            # If max_dd_idx is the first element, peak is also the first element
            peak_idx = portfolio_values.index[0] if len(portfolio_values) > 0 else None
    else:
        peak_idx = None
    
    return (abs(max_dd), peak_idx, max_dd_idx)


def calculate_drawdowns(portfolio_values: pd.Series) -> pd.Series:
    # Calculate drawdown series
    if len(portfolio_values) == 0:
        return pd.Series(dtype=float)
    
    running_max = portfolio_values.expanding().max()
    drawdown = (portfolio_values - running_max) / running_max
    return drawdown


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Calculate Mean Absolute Error
    return np.mean(np.abs(y_true - y_pred))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Calculate Root Mean Squared Error
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_total_return(initial_value: float, final_value: float) -> float:
    # Calculate total return percentage
    if initial_value == 0:
        return 0.0
    return ((final_value - initial_value) / initial_value) * 100


def calculate_annualized_return(returns: pd.Series, 
                               periods_per_year: int = 252) -> float:
    # Calculate annualized return
    if len(returns) == 0:
        return 0.0
    
    total_return = (1 + returns).prod() - 1
    num_periods = len(returns)
    
    if num_periods == 0:
        return 0.0
    
    annualized = (1 + total_return) ** (periods_per_year / num_periods) - 1
    return annualized * 100


def calculate_volatility(returns: pd.Series, 
                        periods_per_year: int = 252) -> float:
    # Calculate annualized volatility
    if len(returns) == 0:
        return 0.0
    
    return returns.std() * np.sqrt(periods_per_year) * 100


def calculate_win_rate(trades: List[Trade]) -> float:
    # Calculate win rate percentage
    if len(trades) == 0:
        return 0.0
    
    # Group trades by symbol to calculate P&L per position
    positions = {}
    for trade in trades:
        symbol = trade.symbol
        if symbol not in positions:
            positions[symbol] = {'buys': [], 'sells': []}
        
        if trade.action == 'BUY':
            positions[symbol]['buys'].append(trade)
        else:
            positions[symbol]['sells'].append(trade)
    
    # Calculate P&L for each closed position
    profitable_trades = 0
    total_closed = 0
    
    for symbol, pos_trades in positions.items():
        buys = pos_trades['buys']
        sells = pos_trades['sells']
        
        # Match buys and sells (FIFO)
        buy_idx = 0
        for sell in sells:
            if buy_idx >= len(buys):
                break
            
            buy = buys[buy_idx]
            pnl = (sell.price - buy.price) * min(sell.quantity, buy.quantity)
            
            if pnl > 0:
                profitable_trades += 1
            total_closed += 1
            
            buy_idx += 1
    
    if total_closed == 0:
        return 0.0
    
    return (profitable_trades / total_closed) * 100


def calculate_profit_factor(trades: List[Trade]) -> float:
    """
    Calculate profit factor (gross profit / gross loss).
    
    Args:
        trades: List of Trade objects
        
    Returns:
        Profit factor ratio
    """
    if len(trades) == 0:
        return 0.0
    
    # Group trades by symbol
    positions = {}
    for trade in trades:
        symbol = trade.symbol
        if symbol not in positions:
            positions[symbol] = {'buys': [], 'sells': []}
        
        if trade.action == 'BUY':
            positions[symbol]['buys'].append(trade)
        else:
            positions[symbol]['sells'].append(trade)
    
    gross_profit = 0.0
    gross_loss = 0.0
    
    for symbol, pos_trades in positions.items():
        buys = pos_trades['buys']
        sells = pos_trades['sells']
        
        buy_idx = 0
        for sell in sells:
            if buy_idx >= len(buys):
                break
            
            buy = buys[buy_idx]
            pnl = (sell.price - buy.price) * min(sell.quantity, buy.quantity)
            
            if pnl > 0:
                gross_profit += pnl
            else:
                gross_loss += abs(pnl)
            
            buy_idx += 1
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def generate_performance_report(trader: MockTrader,
                               portfolio_history: List[Dict],
                               benchmark_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
    """
    Generate comprehensive performance report.
    
    Args:
        trader: MockTrader instance
        portfolio_history: List of portfolio snapshots
        benchmark_returns: Optional benchmark returns for comparison
        
    Returns:
        Dictionary containing all performance metrics:
        - total_return
        - annualized_return
        - volatility
        - sharpe_ratio
        - sortino_ratio
        - max_drawdown
        - win_rate
        - profit_factor
        - total_trades
        - etc.
    """
    if len(portfolio_history) == 0:
        return {}
    
    # Convert portfolio history to DataFrame
    df = pd.DataFrame(portfolio_history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Calculate returns
    portfolio_values = df['total_value']
    returns = portfolio_values.pct_change().dropna()
    
    # Basic metrics
    initial_value = trader.initial_cash
    final_value = portfolio_values.iloc[-1]
    total_return = calculate_total_return(initial_value, final_value)
    
    # Risk metrics
    if len(returns) > 0:
        annualized_return = calculate_annualized_return(returns)
        volatility = calculate_volatility(returns)
        sharpe_ratio = calculate_sharpe_ratio(returns)
        sortino_ratio = calculate_sortino_ratio(returns)
    else:
        annualized_return = 0.0
        volatility = 0.0
        sharpe_ratio = 0.0
        sortino_ratio = 0.0
    
    # Drawdown
    max_dd, peak_date, trough_date = calculate_max_drawdown(portfolio_values)
    
    # Trade metrics
    trades = trader.get_trade_history()
    win_rate = calculate_win_rate(trades)
    profit_factor = calculate_profit_factor(trades)
    
    report = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_dd * 100,  # Convert to percentage
        'max_drawdown_peak_date': peak_date,
        'max_drawdown_trough_date': trough_date,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_trades': len(trades),
        'initial_cash': initial_value,
        'final_value': final_value,
        'cash_balance': trader.get_cash_balance(),
        'num_positions': len(trader.get_positions()),
        'periods': len(portfolio_history)
    }
    
    # Add benchmark comparison if provided
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        benchmark_total_return = calculate_total_return(
            benchmark_returns.iloc[0] if isinstance(benchmark_returns, pd.Series) else benchmark_returns[0],
            benchmark_returns.iloc[-1] if isinstance(benchmark_returns, pd.Series) else benchmark_returns[-1]
        )
        report['benchmark_return'] = benchmark_total_return
        report['excess_return'] = total_return - benchmark_total_return
    
    return report


def plot_performance_metrics(portfolio_history: List[Dict],
                            benchmark_data: Optional[pd.DataFrame] = None,
                            save_path: Optional[str] = None) -> None:
    # Generate performance visualization plots
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    if len(portfolio_history) == 0:
        print("No portfolio history to plot")
        return
    
    df = pd.DataFrame(portfolio_history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Portfolio Value
    ax1 = axes[0]
    ax1.plot(df['timestamp'], df['total_value'], label='Portfolio Value', linewidth=2)
    if benchmark_data is not None:
        ax1.plot(benchmark_data['timestamp'], benchmark_data['value'], 
                label='Benchmark', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title('Portfolio Value Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: Returns
    ax2 = axes[1]
    returns = df['total_value'].pct_change().dropna()
    ax2.plot(df['timestamp'].iloc[1:], returns * 100, label='Daily Returns', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Return (%)')
    ax2.set_title('Daily Returns')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 3: Drawdown
    ax3 = axes[2]
    drawdowns = calculate_drawdowns(df['total_value'])
    ax3.fill_between(df['timestamp'], drawdowns * 100, 0, alpha=0.3, color='red')
    ax3.plot(df['timestamp'], drawdowns * 100, color='red', linewidth=1)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Drawdown (%)')
    ax3.set_title('Drawdown Over Time')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
