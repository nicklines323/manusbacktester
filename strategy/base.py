# src/strategy/base.py

import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    """
    Base class for all trading strategies.
    
    This class defines the interface that all strategies must implement.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the BaseStrategy.
        
        Args:
            params (Dict[str, Any], optional): Strategy parameters
        """
        self.name = "base"
        self.params = params or {}
    
    @abstractmethod
    def populate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the data.
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: Data with indicators
        """
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals.
        
        Args:
            data (pd.DataFrame): Price data with indicators
            
        Returns:
            pd.DataFrame: Data with signals
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, data: pd.DataFrame, balance: float) -> float:
        """
        Calculate position size based on available balance and risk settings.
        
        Args:
            data (pd.DataFrame): Price data with indicators
            balance (float): Available balance
            
        Returns:
            float: Position size
        """
        pass
    
    def backtest(self, data: pd.DataFrame, initial_balance: float = 10000, commission: float = 0.001) -> Dict[str, Any]:
        """
        Run a backtest of the strategy.
        
        Args:
            data (pd.DataFrame): Price data
            initial_balance (float, optional): Initial balance
            commission (float, optional): Commission rate
            
        Returns:
            Dict[str, Any]: Backtest results
        """
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Add indicators and signals if not already present
        if 'buy_signal' not in df.columns or 'sell_signal' not in df.columns:
            df = self.generate_signals(df)
        
        # Initialize backtest variables
        balance = initial_balance
        position = 0
        entry_price = 0
        trades = []
        
        # Run backtest
        for i in range(1, len(df)):
            date = df.index[i]
            close = df['close'].iloc[i]
            
            # Check for buy signal
            if df['buy_signal'].iloc[i] and position <= 0:
                # Close short position if exists
                if position < 0:
                    profit = (entry_price - close) * abs(position)
                    balance += profit
                    balance -= commission * close * abs(position)
                    
                    trades.append({
                        'date': date,
                        'type': 'close_short',
                        'price': close,
                        'position': abs(position),
                        'profit': profit,
                        'balance': balance
                    })
                
                # Calculate position size
                position_size = self.calculate_position_size(df.iloc[:i], balance)
                position = position_size / close
                entry_price = close
                balance -= commission * close * position
                
                trades.append({
                    'date': date,
                    'type': 'buy',
                    'price': close,
                    'position': position,
                    'balance': balance
                })
            
            # Check for sell signal
            elif df['sell_signal'].iloc[i] and position >= 0:
                # Close long position if exists
                if position > 0:
                    profit = (close - entry_price) * position
                    balance += profit
                    balance -= commission * close * position
                    
                    trades.append({
                        'date': date,
                        'type': 'close_long',
                        'price': close,
                        'position': position,
                        'profit': profit,
                        'balance': balance
                    })
                
                # Calculate position size for short
                position_size = self.calculate_position_size(df.iloc[:i], balance)
                position = -position_size / close
                entry_price = close
                balance -= commission * close * abs(position)
                
                trades.append({
                    'date': date,
                    'type': 'sell',
                    'price': close,
                    'position': abs(position),
                    'balance': balance
                })
        
        # Close final position
        if position != 0:
            date = df.index[-1]
            close = df['close'].iloc[-1]
            
            if position > 0:
                profit = (close - entry_price) * position
                balance += profit
                balance -= commission * close * position
                
                trades.append({
                    'date': date,
                    'type': 'close_long',
                    'price': close,
                    'position': position,
                    'profit': profit,
                    'balance': balance
                })
            else:
                profit = (entry_price - close) * abs(position)
                balance += profit
                balance -= commission * close * abs(position)
                
                trades.append({
                    'date': date,
                    'type': 'close_short',
                    'price': close,
                    'position': abs(position),
                    'profit': profit,
                    'balance': balance
                })
        
        # Calculate performance metrics
        total_trades = len([t for t in trades if t['type'] in ['buy', 'sell']])
        winning_trades = len([t for t in trades if t['type'] in ['close_long', 'close_short'] and t.get('profit', 0) > 0])
        losing_trades = len([t for t in trades if t['type'] in ['close_long', 'close_short'] and t.get('profit', 0) <= 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Return results
        return {
            'initial_balance': initial_balance,
            'final_balance': balance,
            'return': (balance / initial_balance - 1) * 100,
            'trades': trades,
            'metrics': {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate
            }
        }
