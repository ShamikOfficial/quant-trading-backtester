# Base strategy abstract class

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from datetime import datetime
import pandas as pd


class BaseStrategy(ABC):
    # Abstract base class for all trading strategies
    
    def __init__(self, name: str, parameters: Optional[Dict[str, Any]] = None):
        # Initialize base strategy
        self.name = name
        self.parameters = parameters or {}
        self.signals_history = []
        self.orders_history = []
        
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, current_position: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on current market data.
        
        This is the core method that each strategy must implement.
        
        Args:
            data: Current market data (DataFrame)
            current_position: Current portfolio position information
            
        Returns:
            Dictionary containing signal information:
            {
                'action': 'BUY', 'SELL', or 'HOLD',
                'confidence': float (0-1),
                'quantity': int (optional),
                'price': float (optional),
                'reason': str (optional)
            }
        """
        pass
    
    @abstractmethod
    def create_order(self, signal: Dict[str, Any], portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a signal into an executable order.
        
        Args:
            signal: Signal dictionary from generate_signal()
            portfolio_state: Current portfolio state (cash, holdings, etc.)
            
        Returns:
            Order dictionary:
            {
                'symbol': str,
                'action': 'BUY' or 'SELL',
                'quantity': int,
                'price': float,
                'timestamp': datetime
            }
        """
        pass
    
    def validate_order(self, order: Dict[str, Any], portfolio_state: Dict[str, Any]) -> bool:
        """
        Validate if an order can be executed given current portfolio constraints.
        
        Args:
            order: Order dictionary to validate
            portfolio_state: Current portfolio state
            
        Returns:
            True if order is valid, False otherwise
        """
        if not order:
            return False
        
        action = order.get('action')
        quantity = order.get('quantity', 0)
        price = order.get('price', 0.0)
        symbol = order.get('symbol')
        
        # Basic validation
        if action not in ['BUY', 'SELL']:
            return False
        
        if quantity <= 0:
            return False
        
        if price <= 0:
            return False
        
        if not symbol:
            return False
        
        # Check portfolio constraints
        cash = portfolio_state.get('cash', 0.0)
        positions = portfolio_state.get('positions', {})
        current_position = positions.get(symbol, {})
        current_quantity = current_position.get('quantity', 0)
        
        if action == 'BUY':
            # Check if we have enough cash
            total_cost = quantity * price
            if total_cost > cash:
                return False
        
        elif action == 'SELL':
            # Check if we have enough shares
            if current_quantity < quantity:
                return False
        
        return True
    
    def execute_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an order (typically called by the simulator).
        
        Note: This is a placeholder method. Actual execution should be done
        through MockTrader.execute_order() in the simulator.
        
        Args:
            order: Order dictionary to execute
            
        Returns:
            Execution result dictionary:
            {
                'status': 'FILLED', 'PARTIAL', or 'REJECTED',
                'filled_quantity': int,
                'avg_price': float,
                'timestamp': datetime
            }
        """
        # Placeholder method - orders executed by MockTrader
        return {
            'status': 'PENDING',
            'filled_quantity': 0,
            'avg_price': order.get('price', 0.0),
            'timestamp': datetime.now(),
            'message': 'Order should be executed through MockTrader.execute_order()'
        }
    
    def update_parameters(self, new_parameters: Dict[str, Any]) -> None:
        """
        Update strategy parameters dynamically.
        
        Args:
            new_parameters: Dictionary of new parameter values
        """
        if new_parameters:
            self.parameters.update(new_parameters)
    
    def get_strategy_state(self) -> Dict[str, Any]:
        """
        Get current strategy state and statistics.
        
        Returns:
            Dictionary containing strategy state information
        """
        return {
            'name': self.name,
            'parameters': self.parameters.copy(),
            'total_signals': len(self.signals_history),
            'total_orders': len(self.orders_history),
            'recent_signals': self.signals_history[-10:] if len(self.signals_history) > 0 else [],
            'recent_orders': self.orders_history[-10:] if len(self.orders_history) > 0 else []
        }
