# src/backtesting/visualization.py

import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BacktestVisualizer:
    """
    Creates visualizations for backtest results.
    
    This class:
    - Generates equity curve charts
    - Creates drawdown visualizations
    - Produces trade analysis charts
    - Visualizes strategy performance
    """
    
    def __init__(self):
        """
        Initialize the BacktestVisualizer.
        """
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Configure figure size and DPI
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['figure.dpi'] = 100
        
        logger.info("Initialized BacktestVisualizer")
    
    def create_equity_curve(self, 
                           backtest_results: Dict[str, Any],
                           output_path: Optional[str] = None) -> None:
        """
        Create equity curve chart.
        
        Args:
            backtest_results (Dict[str, Any]): Backtest results
            output_path (str, optional): Output path for the chart
        """
        try:
            # Extract trades
            trades = backtest_results.get('trades', [])
            
            if not trades:
                logger.warning("No trades to visualize")
                return
            
            # Create DataFrame with balance history
            balance_history = []
            
            for trade in trades:
                balance_history.append({
                    'date': trade['date'],
                    'balance': trade['balance']
                })
            
            balance_df = pd.DataFrame(balance_history)
            balance_df['date'] = pd.to_datetime(balance_df['date'])
            balance_df.set_index('date', inplace=True)
            
            # Create figure
            fig, ax = plt.subplots()
            
            # Plot equity curve
            ax.plot(balance_df.index, balance_df['balance'], linewidth=2)
            
            # Add initial balance
            initial_balance = backtest_results.get('initial_balance', 0)
            ax.axhline(y=initial_balance, color='r', linestyle='--', alpha=0.5)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
            # Add labels and title
            ax.set_xlabel('Date')
            ax.set_ylabel('Balance')
            ax.set_title('Equity Curve')
            
            # Add grid
            ax.grid(True)
            
            # Add legend
            ax.legend(['Equity', 'Initial Balance'])
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or show
            if output_path:
                plt.savefig(output_path)
                logger.info(f"Saved equity curve chart to {output_path}")
            else:
                plt.show()
            
            # Close figure
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error creating equity curve chart: {e}")
    
    def create_drawdown_chart(self, 
                             backtest_results: Dict[str, Any],
                             output_path: Optional[str] = None) -> None:
        """
        Create drawdown chart.
        
        Args:
            backtest_results (Dict[str, Any]): Backtest results
            output_path (str, optional): Output path for the chart
        """
        try:
            # Extract trades
            trades = backtest_results.get('trades', [])
            
            if not trades:
                logger.warning("No trades to visualize")
                return
            
            # Create DataFrame with balance history
            balance_history = []
            
            for trade in trades:
                balance_history.append({
                    'date': trade['date'],
                    'balance': trade['balance']
                })
            
            balance_df = pd.DataFrame(balance_history)
            balance_df['date'] = pd.to_datetime(balance_df['date'])
            balance_df.set_index('date', inplace=True)
            
            # Calculate running maximum
            balance_df['running_max'] = balance_df['balance'].cummax()
            
            # Calculate drawdown
            balance_df['drawdown'] = (balance_df['running_max'] - balance_df['balance']) / balance_df['running_max'] * 100
            
            # Create figure
            fig, ax = plt.subplots()
            
            # Plot drawdown
            ax.fill_between(balance_df.index, 0, balance_df['drawdown'], color='red', alpha=0.3)
            ax.plot(balance_df.index, balance_df['drawdown'], color='red', linewidth=1)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
            # Add labels and title
            ax.set_xlabel('Date')
            ax.set_ylabel('Drawdown (%)')
            ax.set_title('Drawdown')
            
            # Invert y-axis
            ax.invert_yaxis()
            
            # Add grid
            ax.grid(True)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or show
            if output_path:
                plt.savefig(output_path)
                logger.info(f"Saved drawdown chart to {output_path}")
            else:
                plt.show()
            
            # Close figure
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error creating drawdown chart: {e}")
    
    def create_monthly_returns_heatmap(self, 
                                     backtest_results: Dict[str, Any],
                                     output_path: Optional[str] = None) -> None:
        """
        Create monthly returns heatmap.
        
        Args:
            backtest_results (Dict[str, Any]): Backtest results
            output_path (str, optional): Output path for the chart
        """
        try:
            # Extract trades
            trades = backtest_results.get('trades', [])
            
            if not trades:
                logger.warning("No trades to visualize")
                return
            
            # Create DataFrame with balance history
            balance_history = []
            
            for trade in trades:
                balance_history.append({
                    'date': trade['date'],
                    'balance': trade['balance']
                })
            
            balance_df = pd.DataFrame(balance_history)
            balance_df['date'] = pd.to_datetime(balance_df['date'])
            balance_df.set_index('date', inplace=True)
            
            # Resample to daily frequency
            daily_balance = balance_df.resample('D').last()
            
            # Forward fill missing values
            daily_balance = daily_balance.fillna(method='ffill')
            
            # Calculate daily returns
            daily_balance['return'] = daily_balance['balance'].pct_change()
            
            # Calculate monthly returns
            monthly_returns = daily_balance['return'].resample('M').apply(lambda x: (1 + x).prod() - 1)
            
            # Create pivot table
            monthly_returns.index = pd.MultiIndex.from_arrays([
                monthly_returns.index.year,
                monthly_returns.index.month
            ])
            
            monthly_returns = monthly_returns.reset_index()
            pivot = monthly_returns.pivot('level_0', 'level_1', 0)
            
            # Rename columns to month names
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            pivot.columns = [month_names[i-1] for i in pivot.columns]
            
            # Create figure
            fig, ax = plt.subplots()
            
            # Create heatmap
            sns.heatmap(
                pivot * 100,
                annot=True,
                fmt='.2f',
                cmap='RdYlGn',
                center=0,
                linewidths=1,
                ax=ax,
                cbar_kws={'label': 'Return (%)'}
            )
            
            # Add labels and title
            ax.set_title('Monthly Returns (%)')
            ax.set_ylabel('Year')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or show
            if output_path:
                plt.savefig(output_path)
                logger.info(f"Saved monthly returns heatmap to {output_path}")
            else:
                plt.show()
            
            # Close figure
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error creating monthly returns heatmap: {e}")
    
    def create_trade_analysis(self, 
                             backtest_results: Dict[str, Any],
                             output_path: Optional[str] = None) -> None:
        """
        Create trade analysis charts.
        
        Args:
            backtest_results (Dict[str, Any]): Backtest results
            output_path (str, optional): Output path for the chart
        """
        try:
            # Extract trades
            trades = backtest_results.get('trades', [])
            
            if not trades:
                logger.warning("No trades to visualize")
                return
            
            # Filter trades
            close_trades = [t for t in trades if t['type'] in ['close_long', 'close_short']]
            
            if not close_trades:
                logger.warning("No closed trades to visualize")
                return
            
            # Extract profits
            profits = [t.get('profit', 0) for t in close_trades]
            
            # Create figure with subplots
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            
            # Plot profit distribution
            axs[0, 0].hist(profits, bins=20, alpha=0.7, color='blue')
            axs[0, 0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
            axs[0, 0].set_xlabel('Profit')
            axs[0, 0].set_ylabel('Frequency')
            axs[0, 0].set_title('Profit Distribution')
            axs[0, 0].grid(True)
            
            # Plot cumulative profit
            cumulative_profit = np.cumsum(profits)
            axs[0, 1].plot(cumulative_profit, linewidth=2)
            axs[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axs[0, 1].set_xlabel('Trade Number')
            axs[0, 1].set_ylabel('Cumulative Profit')
            axs[0, 1].set_title('Cumulative Profit')
            axs[0, 1].grid(True)
            
            # Plot profit by trade type
            long_profits = [t.get('profit', 0) for t in close_trades if t['type'] == 'close_long']
            short_profits = [t.get('profit', 0) for t in close_trades if t['type'] == 'close_short']
            
            axs[1, 0].boxplot([long_profits, short_profits], labels=['Long', 'Short'])
            axs[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axs[1, 0].set_xlabel('Trade Type')
            axs[1, 0].set_ylabel('Profit')
            axs[1, 0].set_title('Profit by Trade Type')
            axs[1, 0].grid(True)
            
            # Plot win/loss ratio
            win_count = len([p for p in profits if p > 0])
            loss_count = len([p for p in profits if p <= 0])
            
            axs[1, 1].bar(['Win', 'Loss'], [win_count, loss_count], alpha=0.7, color=['green', 'red'])
            axs[1, 1].set_xlabel('Outcome')
            axs[1, 1].set_ylabel('Count')
            axs[1, 1].set_title('Win/Loss Count')
            axs[1, 1].grid(True)
            
            # Add win rate as text
            win_rate = win_count / (win_count + loss_count) if (win_count + loss_count) > 0 else 0
            axs[1, 1].text(0.5, 0.9, f'Win Rate: {win_rate:.2%}', horizontalalignment='center', transform=axs[1, 1].transAxes)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or show
            if output_path:
                plt.savefig(output_path)
                logger.info(f"Saved trade analysis chart to {output_path}")
            else:
                plt.show()
            
            # Close figure
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error creating trade analysis chart: {e}")
    
    def create_strategy_chart(self, 
                             data: pd.DataFrame,
                             strategy: Any,
                             backtest_results: Dict[str, Any],
                             output_path: Optional[str] = None) -> None:
        """
        Create strategy chart with price, indicators, and signals.
        
        Args:
            data (pd.DataFrame): Price data
            strategy (Any): Strategy instance
            backtest_results (Dict[str, Any]): Backtest results
            output_path (str, optional): Output path for the chart
        """
        try:
            # Add indicators and signals if not already present
            if 'average' not in data.columns:
                data = strategy.populate_indicators(data)
            
            if 'buy_signal' not in data.columns:
                data = strategy.generate_signals(data)
            
            # Extract trades
            trades = backtest_results.get('trades', [])
            
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Plot price
            ax.plot(data.index, data['close'], label='Price', linewidth=1)
            
            # Plot average
            ax.plot(data.index, data['average'], label='Average', linewidth=1)
            
            # Plot envelope bands
            for envelope in strategy.params['envelopes']:
                band_high = f'band_high_{envelope * 100:.0f}'
                band_low = f'band_low_{envelope * 100:.0f}'
                
                if band_high in data.columns and band_low in data.columns:
                    ax.plot(data.index, data[band_high], label=f'Upper Band ({envelope * 100:.0f}%)', linestyle='--', linewidth=1)
                    ax.plot(data.index, data[band_low], label=f'Lower Band ({envelope * 100:.0f}%)', linestyle='--', linewidth=1)
            
            # Plot buy signals
            buy_signals = data[data['buy_signal']]
            ax.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='Buy Signal')
            
            # Plot sell signals
            sell_signals = data[data['sell_signal']]
            ax.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100, label='Sell Signal')
            
            # Plot trades
            for trade in trades:
                if trade['type'] == 'buy':
                    ax.scatter(trade['date'], trade['price'], marker='o', color='green', s=50)
                elif trade['type'] == 'sell':
                    ax.scatter(trade['date'], trade['price'], marker='o', color='red', s=50)
                elif trade['type'] == 'close_long':
                    ax.scatter(trade['date'], trade['price'], marker='x', color='blue', s=50)
                elif trade['type'] == 'close_short':
                    ax.scatter(trade['date'], trade['price'], marker='x', color='purple', s=50)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
            # Add labels and title
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.set_title('Strategy Chart')
            
            # Add grid
            ax.grid(True)
            
            # Add legend
            ax.legend()
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or show
            if output_path:
                plt.savefig(output_path)
                logger.info(f"Saved strategy chart to {output_path}")
            else:
                plt.show()
            
            # Close figure
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error creating strategy chart: {e}")
    
    def create_parameter_comparison(self, 
                                  optimization_results: Dict[str, Any],
                                  metric: str,
                                  output_path: Optional[str] = None) -> None:
        """
        Create parameter comparison chart.
        
        Args:
            optimization_results (Dict[str, Any]): Optimization results
            metric (str): Metric to compare
            output_path (str, optional): Output path for the chart
        """
        try:
            # Extract results
            all_results = optimization_results.get('all_results', [])
            
            if not all_results:
                logger.warning("No optimization results to visualize")
                return
            
            # Create DataFrame
            results_df = pd.DataFrame(all_results)
            
            # Get parameter columns
            param_columns = [col for col in results_df.columns if col.startswith('params.')]
            
            if not param_columns:
                # Extract parameters from dictionary
                params_df = pd.DataFrame([r['params'] for r in all_results])
                
                # Merge with results
                results_df = pd.concat([results_df.drop('params', axis=1), params_df], axis=1)
                
                # Get parameter columns
                param_columns = [col for col in results_df.columns if col not in ['value', 'trades', 'final_balance']]
            
            # Create figure with subplots
            n_params = len(param_columns)
            fig, axs = plt.subplots(n_params, 1, figsize=(12, 4 * n_params))
            
            # Handle single parameter case
            if n_params == 1:
                axs = [axs]
            
            # Plot parameter comparison
            for i, param in enumerate(param_columns):
                # Group by parameter
                grouped = results_df.groupby(param)['value'].mean().reset_index()
                
                # Sort by parameter value
                grouped = grouped.sort_values(param)
                
                # Plot
                axs[i].bar(grouped[param].astype(str), grouped['value'], alpha=0.7)
                
                # Add labels and title
                axs[i].set_xlabel(param)
                axs[i].set_ylabel(metric)
                axs[i].set_title(f'{metric} by {param}')
                
                # Add grid
                axs[i].grid(True)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or show
            if output_path:
                plt.savefig(output_path)
                logger.info(f"Saved parameter comparison chart to {output_path}")
            else:
                plt.show()
            
            # Close figure
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error creating parameter comparison chart: {e}")
