# Mock trading simulator for state management

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass


@dataclass
class Trade:
    """
    Data class representing a single trade execution.
    """
    timestamp: datetime
    symbol: str
    action: str  # 'BUY' or 'SELL'
    quantity: int
    price: float
    commission: float = 0.0
    total_cost: float = 0.0


@dataclass
class Position:
    # Represents a position in a single asset
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float = 0.0
    
    @property
    def market_value(self) -> float:
        # Calculate current market value
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        # Calculate unrealized profit/loss
        return (self.current_price - self.avg_cost) * self.quantity


class MockTrader:
    # Mock trading environment simulator
    
    def __init__(self, initial_cash: float = 10000000.0, 
                 commission_rate: float = 0.001):
        # Initialize simulator with starting cash and commission rate
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission_rate = commission_rate
        
        # Portfolio state
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.trade_history: List[Trade] = []
        self.portfolio_history: List[Dict[str, Any]] = []
        
    def get_portfolio_value(self, current_prices: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate total portfolio value (cash + positions).
        
        Args:
            current_prices: Dictionary of current prices for each symbol
            
        Returns:
            Total portfolio value
        """
        if current_prices:
            self.update_prices(current_prices)
        
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value
    
    def get_cash_balance(self) -> float:
        """
        Get current cash balance.
        
        Returns:
            Current cash amount
        """
        return self.cash
    
    def get_positions(self) -> Dict[str, Position]:
        """
        Get current positions.
        
        Returns:
            Dictionary of symbol -> Position
        """
        return self.positions.copy()
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a specific symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Position object or None if no position exists
        """
        return self.positions.get(symbol)
    
    def execute_buy_order(self, symbol: str, quantity: int, 
                         price: float) -> Dict[str, Any]:
        """
        Execute a buy order.
        
        Args:
            symbol: Stock symbol to buy
            quantity: Number of shares to buy
            price: Price per share
            
        Returns:
            Execution result dictionary
        """
        if quantity <= 0:
            return {'status': 'REJECTED', 'reason': 'Invalid quantity'}
        
        commission = self.calculate_commission(quantity, price)
        total_cost = (quantity * price) + commission
        
        if total_cost > self.cash:
            # Partial fill if possible
            max_affordable = int((self.cash - commission) / price)
            if max_affordable <= 0:
                return {'status': 'REJECTED', 'reason': 'Insufficient cash'}
            quantity = max_affordable
            total_cost = (quantity * price) + commission
        
        # Execute trade
        self.cash -= total_cost
        
        # Update or create position
        if symbol in self.positions:
            pos = self.positions[symbol]
            total_shares = pos.quantity + quantity
            total_cost_basis = (pos.quantity * pos.avg_cost) + (quantity * price)
            pos.quantity = total_shares
            pos.avg_cost = total_cost_basis / total_shares
            pos.current_price = price
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_cost=price,
                current_price=price
            )
        
        # Record trade
        trade = Trade(
            timestamp=datetime.now(),
            symbol=symbol,
            action='BUY',
            quantity=quantity,
            price=price,
            commission=commission,
            total_cost=total_cost
        )
        self.trade_history.append(trade)
        
        return {
            'status': 'FILLED',
            'filled_quantity': quantity,
            'avg_price': price,
            'commission': commission,
            'total_cost': total_cost,
            'timestamp': trade.timestamp
        }
    
    def execute_sell_order(self, symbol: str, quantity: int, 
                          price: float) -> Dict[str, Any]:
        """
        Execute a sell order.
        
        Args:
            symbol: Stock symbol to sell
            quantity: Number of shares to sell
            price: Price per share
            
        Returns:
            Execution result dictionary
        """
        if quantity <= 0:
            return {'status': 'REJECTED', 'reason': 'Invalid quantity'}
        
        if symbol not in self.positions:
            return {'status': 'REJECTED', 'reason': 'No position to sell'}
        
        pos = self.positions[symbol]
        if pos.quantity < quantity:
            # Partial fill
            quantity = pos.quantity
        
        commission = self.calculate_commission(quantity, price)
        proceeds = (quantity * price) - commission
        
        # Execute trade
        self.cash += proceeds
        
        # Update position
        pos.quantity -= quantity
        pos.current_price = price
        
        # Remove position if fully sold
        if pos.quantity == 0:
            del self.positions[symbol]
        
        # Record trade
        trade = Trade(
            timestamp=datetime.now(),
            symbol=symbol,
            action='SELL',
            quantity=quantity,
            price=price,
            commission=commission,
            total_cost=proceeds
        )
        self.trade_history.append(trade)
        
        return {
            'status': 'FILLED',
            'filled_quantity': quantity,
            'avg_price': price,
            'commission': commission,
            'proceeds': proceeds,
            'timestamp': trade.timestamp
        }
    
    def execute_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an order (wrapper for buy/sell).
        
        Args:
            order: Order dictionary with 'symbol', 'action', 'quantity', 'price'
            
        Returns:
            Execution result dictionary
        """
        symbol = order.get('symbol')
        action = order.get('action')
        quantity = order.get('quantity', 0)
        price = order.get('price', 0.0)
        
        if action == 'BUY':
            return self.execute_buy_order(symbol, quantity, price)
        elif action == 'SELL':
            return self.execute_sell_order(symbol, quantity, price)
        else:
            return {'status': 'REJECTED', 'reason': f'Invalid action: {action}'}
    
    def update_prices(self, current_prices: Dict[str, float]) -> None:
        """
        Update current prices for all positions.
        
        Args:
            current_prices: Dictionary of symbol -> current price
        """
        for symbol, price in current_prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price
    
    def calculate_commission(self, quantity: int, price: float) -> float:
        """
        Calculate commission for a trade.
        
        Args:
            quantity: Number of shares
            price: Price per share
            
        Returns:
            Commission amount
        """
        return quantity * price * self.commission_rate
    
    def get_trade_history(self) -> List[Trade]:
        """
        Get complete trade history.
        
        Returns:
            List of Trade objects
        """
        return self.trade_history.copy()
    
    def get_portfolio_summary(self, current_prices: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Get comprehensive portfolio summary.
        
        Args:
            current_prices: Dictionary of current prices
            
        Returns:
            Dictionary containing:
            - total_value: Total portfolio value
            - cash: Cash balance
            - positions_value: Total value of positions
            - positions: List of position details
            - total_pnl: Total profit/loss
            - return_pct: Percentage return
        """
        if current_prices:
            self.update_prices(current_prices)
        
        positions_value = sum(pos.market_value for pos in self.positions.values())
        total_value = self.cash + positions_value
        total_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        return_pct = ((total_value - self.initial_cash) / self.initial_cash) * 100
        
        position_details = []
        for symbol, pos in self.positions.items():
            position_details.append({
                'symbol': symbol,
                'quantity': pos.quantity,
                'avg_cost': pos.avg_cost,
                'current_price': pos.current_price,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl
            })
        
        return {
            'total_value': total_value,
            'cash': self.cash,
            'positions_value': positions_value,
            'positions': position_details,
            'total_pnl': total_pnl,
            'return_pct': return_pct,
            'num_positions': len(self.positions),
            'num_trades': len(self.trade_history)
        }
    
    def reset(self) -> None:
        """
        Reset the simulator to initial state.
        """
        self.cash = self.initial_cash
        self.positions = {}
        self.trade_history = []
        self.portfolio_history = []
    
    def record_portfolio_snapshot(self, timestamp: datetime, 
                                 current_prices: Optional[Dict[str, float]] = None) -> None:
        """
        Record a snapshot of portfolio state at a given timestamp.
        
        Args:
            timestamp: Timestamp for the snapshot
            current_prices: Current prices for valuation
        """
        summary = self.get_portfolio_summary(current_prices)
        snapshot = {
            'timestamp': timestamp,
            **summary
        }
        self.portfolio_history.append(snapshot)
