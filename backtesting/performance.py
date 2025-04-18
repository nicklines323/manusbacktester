# src/backtesting/performance.py

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Analyzes trading performance.
    
    This class:
    - Calculates performance metrics
    - Analyzes trade statistics
    - Computes risk metrics
    """
    
    def __init__(self):
        """
        Initialize the PerformanceAnalyzer.
        """
        logger.info("Initialized PerformanceAnalyzer")
    
    def calculate_metrics(self, 
                         backtest_results: Dict[str, Any],
                         data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate performance metrics.
        
        Args:
            backtest_results (Dict[str, Any]): Backtest results
            data (pd.DataFrame): Price data
            
        Returns:
            Dict[str, Any]: Performance metrics
        """
        metrics = {}
        
        try:
            # Extract data
            initial_balance = backtest_results.get('initial_balance', 0)
            final_balance = backtest_results.get('final_balance', 0)
            trades = backtest_results.get('trades', [])
            
            # Return metrics
            metrics['total_return'] = (final_balance / initial_balance - 1) * 100
            
            # Calculate daily returns
            daily_returns = self._calculate_daily_returns(trades, data)
            
            # Risk metrics
            metrics['volatility'] = daily_returns.std() * np.sqrt(252) * 100
            metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(daily_returns)
            metrics['sortino_ratio'] = self._calculate_sortino_ratio(daily_returns)
            metrics['max_drawdown'] = self._calculate_max_drawdown(trades) * 100
            metrics['calmar_ratio'] = metrics['total_return'] / metrics['max_drawdown'] if metrics['max_drawdown'] > 0 else 0
            metrics['value_at_risk'] = self._calculate_value_at_risk(daily_returns) * 100
            
            # Trade statistics
            metrics['total_trades'] = len([t for t in trades if t['type'] in ['buy', 'sell']])
            metrics['winning_trades'] = len([t for t in trades if t['type'] in ['close_long', 'close_short'] and t.get('profit', 0) > 0])
            metrics['losing_trades'] = len([t for t in trades if t['type'] in ['close_long', 'close_short'] and t.get('profit', 0) <= 0])
            
            metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades'] if metrics['total_trades'] > 0 else 0
            
            # Calculate profit factor
            gross_profit = sum([t.get('profit', 0) for t in trades if t['type'] in ['close_long', 'close_short'] and t.get('profit', 0) > 0])
            gross_loss = sum([abs(t.get('profit', 0)) for t in trades if t['type'] in ['close_long', 'close_short'] and t.get('profit', 0) < 0])
            
            metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Calculate expectancy
            avg_win = gross_profit / metrics['winning_trades'] if metrics['winning_trades'] > 0 else 0
            avg_loss = gross_loss / metrics['losing_trades'] if metrics['losing_trades'] > 0 else 0
            
            metrics['expectancy'] = (metrics['win_rate'] * avg_win) - ((1 - metrics['win_rate']) * avg_loss)
            
            # Calculate average trade metrics
            metrics['avg_trade_return'] = metrics['total_return'] / metrics['total_trades'] if metrics['total_trades'] > 0 else 0
            metrics['avg_trade_duration'] = self._calculate_avg_trade_duration(trades)
            
            # Calculate period returns
            metrics['daily_returns'] = self._calculate_period_returns(daily_returns, 'D')
            metrics['monthly_returns'] = self._calculate_period_returns(daily_returns, 'M')
            metrics['yearly_returns'] = self._calculate_period_returns(daily_returns, 'Y')
            
            logger.info(f"Calculated performance metrics: {len(metrics)} metrics")
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
        
        return metrics
    
    def _calculate_daily_returns(self, 
                               trades: List[Dict[str, Any]],
                               data: pd.DataFrame) -> pd.Series:
        """
        Calculate daily returns.
        
        Args:
            trades (List[Dict[str, Any]]): List of trades
            data (pd.DataFrame): Price data
            
        Returns:
            pd.Series: Daily returns
        """
        try:
            # Create DataFrame with balance history
            balance_history = []
            
            for trade in trades:
                balance_history.append({
                    'date': trade['date'],
                    'balance': trade['balance']
                })
            
            if not balance_history:
                return pd.Series()
            
            balance_df = pd.DataFrame(balance_history)
            balance_df['date'] = pd.to_datetime(balance_df['date'])
            balance_df.set_index('date', inplace=True)
            
            # Resample to daily frequency
            daily_balance = balance_df.resample('D').last()
            
            # Forward fill missing values
            daily_balance = daily_balance.fillna(method='ffill')
            
            # Calculate daily returns
            daily_returns = daily_balance['balance'].pct_change().dropna()
            
            return daily_returns
            
        except Exception as e:
            logger.error(f"Error calculating daily returns: {e}")
            return pd.Series()
    
    def _calculate_sharpe_ratio(self, daily_returns: pd.Series) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            daily_returns (pd.Series): Daily returns
            
        Returns:
            float: Sharpe ratio
        """
        try:
            if daily_returns.empty:
                return 0
            
            # Calculate annualized return and volatility
            annualized_return = daily_returns.mean() * 252
            annualized_volatility = daily_returns.std() * np.sqrt(252)
            
            # Calculate Sharpe ratio
            risk_free_rate = 0.02 / 252  # Assuming 2% annual risk-free rate
            sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
            
            return sharpe_ratio
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0
    
    def _calculate_sortino_ratio(self, daily_returns: pd.Series) -> float:
        """
        Calculate Sortino ratio.
        
        Args:
            daily_returns (pd.Series): Daily returns
            
        Returns:
            float: Sortino ratio
        """
        try:
            if daily_returns.empty:
                return 0
            
            # Calculate annualized return
            annualized_return = daily_returns.mean() * 252
            
            # Calculate downside deviation
            negative_returns = daily_returns[daily_returns < 0]
            downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
            
            # Calculate Sortino ratio
            risk_free_rate = 0.02 / 252  # Assuming 2% annual risk-free rate
            sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            return sortino_ratio
            
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return 0
    
    def _calculate_max_drawdown(self, trades: List[Dict[str, Any]]) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            trades (List[Dict[str, Any]]): List of trades
            
        Returns:
            float: Maximum drawdown
        """
        try:
            if not trades:
                return 0
            
            # Extract balance history
            balance_history = [trade['balance'] for trade in trades]
            
            # Calculate running maximum
            running_max = np.maximum.accumulate(balance_history)
            
            # Calculate drawdown
            drawdown = (running_max - balance_history) / running_max
            
            # Get maximum drawdown
            max_drawdown = np.max(drawdown)
            
            return max_drawdown
            
        except Exception as e:
            logger.error(f"Error calculating maximum drawdown: {e}")
            return 0
    
    def _calculate_value_at_risk(self, daily_returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            daily_returns (pd.Series): Daily returns
            confidence (float): Confidence level
            
        Returns:
            float: Value at Risk
        """
        try:
            if daily_returns.empty:
                return 0
            
            # Calculate VaR
            var = np.percentile(daily_returns, 100 * (1 - confidence))
            
            return abs(var)
            
        except Exception as e:
            logger.error(f"Error calculating Value at Risk: {e}")
            return 0
    
    def _calculate_avg_trade_duration(self, trades: List[Dict[str, Any]]) -> float:
        """
        Calculate average trade duration.
        
        Args:
            trades (List[Dict[str, Any]]): List of trades
            
        Returns:
            float: Average trade duration in days
        """
        try:
            # Filter trades
            entry_trades = [t for t in trades if t['type'] in ['buy', 'sell']]
            exit_trades = [t for t in trades if t['type'] in ['close_long', 'close_short']]
            
            if len(entry_trades) != len(exit_trades):
                # Last trade might not have an exit
                min_len = min(len(entry_trades), len(exit_trades))
                entry_trades = entry_trades[:min_len]
                exit_trades = exit_trades[:min_len]
            
            if not entry_trades or not exit_trades:
                return 0
            
            # Calculate durations
            durations = []
            
            for entry, exit in zip(entry_trades, exit_trades):
                entry_date = pd.to_datetime(entry['date'])
                exit_date = pd.to_datetime(exit['date'])
                
                duration = (exit_date - entry_date).total_seconds() / (24 * 60 * 60)  # Convert to days
                durations.append(duration)
            
            # Calculate average duration
            avg_duration = np.mean(durations)
            
            return avg_duration
            
        except Exception as e:
            logger.error(f"Error calculating average trade duration: {e}")
            return 0
    
    def _calculate_period_returns(self, daily_returns: pd.Series, period: str) -> Dict[str, float]:
        """
        Calculate period returns.
        
        Args:
            daily_returns (pd.Series): Daily returns
            period (str): Period ('D' for daily, 'M' for monthly, 'Y' for yearly)
            
        Returns:
            Dict[str, float]: Period returns
        """
        try:
            if daily_returns.empty:
                return {}
            
            # Resample returns
            if period == 'D':
                period_returns = daily_returns
            elif period == 'M':
                period_returns = (1 + daily_returns).resample('M').prod() - 1
            elif period == 'Y':
                period_returns = (1 + daily_returns).resample('Y').prod() - 1
            else:
                return {}
            
            # Convert to dictionary
            return period_returns.to_dict()
            
        except Exception as e:
            logger.error(f"Error calculating period returns: {e}")
            return {}
