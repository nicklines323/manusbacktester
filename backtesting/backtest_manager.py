# src/backtesting/backtest_manager.py

import logging
import os
import json
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..data.data_manager import DataManager
from ..strategy.strategy_manager import StrategyManager
from .performance import PerformanceAnalyzer
from .visualization import BacktestVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BacktestManager:
    """
    Manages backtesting operations.
    
    This class:
    - Runs backtests for trading strategies
    - Analyzes performance
    - Visualizes results
    - Optimizes strategy parameters
    """
    
    def __init__(self, strategy_manager=None, data_manager=None):
        """
        Initialize the BacktestManager.
        
        Args:
            strategy_manager: Strategy manager instance
            data_manager: Data manager instance
        """
        self.strategy_manager = strategy_manager or StrategyManager()
        self.data_manager = data_manager or DataManager()
        self.performance_analyzer = PerformanceAnalyzer()
        self.visualizer = BacktestVisualizer()
        
        logger.info("Initialized BacktestManager")
    
    def run_backtest(self,
                    strategy_name: str,
                    symbol: str,
                    timeframe: str = '1h',
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None,
                    strategy_params: Optional[Dict[str, Any]] = None,
                    initial_balance: float = 10000,
                    commission: float = 0.001,
                    output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a backtest for a strategy.
        
        Args:
            strategy_name (str): Name of the strategy
            symbol (str): Trading symbol
            timeframe (str): Timeframe
            start_date (datetime, optional): Start date
            end_date (datetime, optional): End date
            strategy_params (Dict[str, Any], optional): Strategy parameters
            initial_balance (float): Initial balance
            commission (float): Commission rate
            output_dir (str, optional): Output directory for results
            
        Returns:
            Dict[str, Any]: Backtest results
        """
        logger.info(f"Running backtest for {strategy_name} on {symbol} {timeframe}")
        
        try:
            # Get strategy
            strategy = self.strategy_manager.get_strategy(
                strategy_name=strategy_name,
                params=strategy_params
            )
            
            # Get data
            data = self.data_manager.get_data(
                symbol=symbol,
                timeframe=timeframe,
                since=start_date,
                add_indicators=False,
                add_envelopes=False
            )
            
            # Filter data by date range
            if start_date:
                data = data[data.index >= pd.Timestamp(start_date)]
            if end_date:
                data = data[data.index <= pd.Timestamp(end_date)]
            
            if data.empty:
                logger.warning(f"No data available for {symbol} {timeframe} in the specified date range")
                return {}
            
            # Run backtest
            backtest_results = strategy.backtest(
                data=data,
                initial_balance=initial_balance,
                commission=commission
            )
            
            # Calculate performance metrics
            performance_metrics = self.performance_analyzer.calculate_metrics(
                backtest_results=backtest_results,
                data=data
            )
            
            # Add metrics to results
            backtest_results['metrics'] = performance_metrics
            
            # Create visualizations if output directory is provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
                # Create visualizations
                self.visualizer.create_equity_curve(
                    backtest_results=backtest_results,
                    output_path=os.path.join(output_dir, 'equity_curve.png')
                )
                
                self.visualizer.create_drawdown_chart(
                    backtest_results=backtest_results,
                    output_path=os.path.join(output_dir, 'drawdown.png')
                )
                
                self.visualizer.create_monthly_returns_heatmap(
                    backtest_results=backtest_results,
                    output_path=os.path.join(output_dir, 'monthly_returns.png')
                )
                
                self.visualizer.create_trade_analysis(
                    backtest_results=backtest_results,
                    output_path=os.path.join(output_dir, 'trade_analysis.png')
                )
                
                self.visualizer.create_strategy_chart(
                    data=data,
                    strategy=strategy,
                    backtest_results=backtest_results,
                    output_path=os.path.join(output_dir, 'strategy_chart.png')
                )
                
                # Save results to JSON
                with open(os.path.join(output_dir, 'backtest_results.json'), 'w') as f:
                    json.dump(
                        {k: v for k, v in backtest_results.items() if k != 'trades'},
                        f,
                        indent=2,
                        default=str
                    )
                
                # Save trades to CSV
                trades_df = pd.DataFrame(backtest_results['trades'])
                trades_df.to_csv(os.path.join(output_dir, 'trades.csv'), index=False)
            
            logger.info(f"Backtest completed: {len(backtest_results['trades'])} trades, final balance: ${backtest_results['final_balance']:.2f}")
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {}
    
    def optimize_parameters(self,
                           strategy_name: str,
                           symbol: str,
                           timeframe: str = '1h',
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None,
                           param_grid: Dict[str, List[Any]] = None,
                           initial_balance: float = 10000,
                           commission: float = 0.001,
                           metric: str = 'total_return',
                           output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Optimize strategy parameters.
        
        Args:
            strategy_name (str): Name of the strategy
            symbol (str): Trading symbol
            timeframe (str): Timeframe
            start_date (datetime, optional): Start date
            end_date (datetime, optional): End date
            param_grid (Dict[str, List[Any]]): Parameter grid
            initial_balance (float): Initial balance
            commission (float): Commission rate
            metric (str): Metric to optimize
            output_dir (str, optional): Output directory for results
            
        Returns:
            Dict[str, Any]: Optimization results
        """
        logger.info(f"Optimizing parameters for {strategy_name} on {symbol} {timeframe}")
        
        if not param_grid:
            logger.warning("No parameter grid provided")
            return {}
        
        try:
            # Get data
            data = self.data_manager.get_data(
                symbol=symbol,
                timeframe=timeframe,
                since=start_date,
                add_indicators=False,
                add_envelopes=False
            )
            
            # Filter data by date range
            if start_date:
                data = data[data.index >= pd.Timestamp(start_date)]
            if end_date:
                data = data[data.index <= pd.Timestamp(end_date)]
            
            if data.empty:
                logger.warning(f"No data available for {symbol} {timeframe} in the specified date range")
                return {}
            
            # Generate parameter combinations
            param_keys = list(param_grid.keys())
            param_values = list(param_grid.values())
            
            # Initialize results
            optimization_results = {
                'best_params': {},
                'best_value': -float('inf') if metric != 'max_drawdown' else float('inf'),
                'all_results': []
            }
            
            # Helper function to generate parameter combinations
            def generate_combinations(keys, values, current_idx=0, current_params={}):
                if current_idx == len(keys):
                    # Run backtest with current parameters
                    strategy = self.strategy_manager.get_strategy(
                        strategy_name=strategy_name,
                        params=current_params
                    )
                    
                    backtest_results = strategy.backtest(
                        data=data,
                        initial_balance=initial_balance,
                        commission=commission
                    )
                    
                    performance_metrics = self.performance_analyzer.calculate_metrics(
                        backtest_results=backtest_results,
                        data=data
                    )
                    
                    # Get metric value
                    metric_value = performance_metrics.get(metric)
                    
                    if metric_value is None:
                        logger.warning(f"Metric {metric} not found in performance metrics")
                        return
                    
                    # Check if this is the best result
                    is_better = False
                    if metric == 'max_drawdown':
                        # For drawdown, lower is better
                        is_better = metric_value < optimization_results['best_value']
                    else:
                        # For other metrics, higher is better
                        is_better = metric_value > optimization_results['best_value']
                    
                    if is_better:
                        optimization_results['best_params'] = current_params.copy()
                        optimization_results['best_value'] = metric_value
                    
                    # Add to all results
                    optimization_results['all_results'].append({
                        'params': current_params.copy(),
                        'value': metric_value,
                        'trades': len(backtest_results['trades']),
                        'final_balance': backtest_results['final_balance']
                    })
                    
                    return
                
                # Recursively generate combinations
                key = keys[current_idx]
                for value in values[current_idx]:
                    new_params = current_params.copy()
                    new_params[key] = value
                    generate_combinations(keys, values, current_idx + 1, new_params)
            
            # Generate and test all combinations
            generate_combinations(param_keys, param_values)
            
            # Sort results
            optimization_results['all_results'].sort(
                key=lambda x: x['value'],
                reverse=metric != 'max_drawdown'
            )
            
            # Save results if output directory is provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
                # Save optimization results to JSON
                with open(os.path.join(output_dir, 'optimization_results.json'), 'w') as f:
                    json.dump(optimization_results, f, indent=2, default=str)
                
                # Create parameter comparison chart
                self.visualizer.create_parameter_comparison(
                    optimization_results=optimization_results,
                    metric=metric,
                    output_path=os.path.join(output_dir, 'parameter_comparison.png')
                )
            
            logger.info(f"Parameter optimization completed: {len(optimization_results['all_results'])} combinations tested")
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error optimizing parameters: {e}")
            return {}
    
    def generate_report(self,
                       backtest_results: Dict[str, Any],
                       output_path: str) -> bool:
        """
        Generate a backtest report.
        
        Args:
            backtest_results (Dict[str, Any]): Backtest results
            output_path (str): Output path for the report
            
        Returns:
            bool: Whether the report was generated successfully
        """
        logger.info(f"Generating backtest report at {output_path}")
        
        try:
            # Create report directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Generate HTML report
            with open(output_path, 'w') as f:
                f.write('<html>\n')
                f.write('<head>\n')
                f.write('<title>Backtest Report</title>\n')
                f.write('<style>\n')
                f.write('body { font-family: Arial, sans-serif; margin: 20px; }\n')
                f.write('h1, h2 { color: #333; }\n')
                f.write('table { border-collapse: collapse; width: 100%; }\n')
                f.write('th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n')
                f.write('th { background-color: #f2f2f2; }\n')
                f.write('tr:nth-child(even) { background-color: #f9f9f9; }\n')
                f.write('.metric { font-weight: bold; }\n')
                f.write('.positive { color: green; }\n')
                f.write('.negative { color: red; }\n')
                f.write('</style>\n')
                f.write('</head>\n')
                f.write('<body>\n')
                
                # Header
                f.write('<h1>Backtest Report</h1>\n')
                f.write(f'<p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>\n')
                
                # Summary
                f.write('<h2>Summary</h2>\n')
                f.write('<table>\n')
                f.write('<tr><th>Metric</th><th>Value</th></tr>\n')
                f.write(f'<tr><td>Initial Balance</td><td>${backtest_results["initial_balance"]:.2f}</td></tr>\n')
                f.write(f'<tr><td>Final Balance</td><td>${backtest_results["final_balance"]:.2f}</td></tr>\n')
                
                total_return = backtest_results["metrics"].get("total_return", 0)
                return_class = "positive" if total_return >= 0 else "negative"
                f.write(f'<tr><td>Total Return</td><td class="{return_class}">{total_return:.2f}%</td></tr>\n')
                
                f.write(f'<tr><td>Sharpe Ratio</td><td>{backtest_results["metrics"].get("sharpe_ratio", 0):.2f}</td></tr>\n')
                f.write(f'<tr><td>Max Drawdown</td><td class="negative">{backtest_results["metrics"].get("max_drawdown", 0):.2f}%</td></tr>\n')
                f.write(f'<tr><td>Total Trades</td><td>{backtest_results["metrics"].get("total_trades", 0)}</td></tr>\n')
                f.write(f'<tr><td>Win Rate</td><td>{backtest_results["metrics"].get("win_rate", 0) * 100:.2f}%</td></tr>\n')
                f.write(f'<tr><td>Profit Factor</td><td>{backtest_results["metrics"].get("profit_factor", 0):.2f}</td></tr>\n')
                f.write('</table>\n')
                
                # Performance Metrics
                f.write('<h2>Performance Metrics</h2>\n')
                f.write('<table>\n')
                f.write('<tr><th>Metric</th><th>Value</th></tr>\n')
                
                for key, value in backtest_results["metrics"].items():
                    if key not in ['total_return', 'sharpe_ratio', 'max_drawdown', 'total_trades', 'win_rate', 'profit_factor']:
                        if isinstance(value, (int, float)):
                            f.write(f'<tr><td>{key.replace("_", " ").title()}</td><td>{value:.4f}</td></tr>\n')
                        else:
                            f.write(f'<tr><td>{key.replace("_", " ").title()}</td><td>{value}</td></tr>\n')
                
                f.write('</table>\n')
                
                # Trades
                f.write('<h2>Trades</h2>\n')
                f.write('<table>\n')
                f.write('<tr><th>Date</th><th>Type</th><th>Price</th><th>Position</th><th>Profit</th><th>Balance</th></tr>\n')
                
                for trade in backtest_results["trades"]:
                    trade_type = trade.get('type', '')
                    profit = trade.get('profit', 0)
                    profit_class = "positive" if profit > 0 else "negative" if profit < 0 else ""
                    
                    f.write('<tr>\n')
                    f.write(f'<td>{trade.get("date").strftime("%Y-%m-%d %H:%M:%S") if isinstance(trade.get("date"), datetime) else trade.get("date")}</td>\n')
                    f.write(f'<td>{trade_type}</td>\n')
                    f.write(f'<td>${trade.get("price", 0):.2f}</td>\n')
                    f.write(f'<td>{trade.get("position", 0):.6f}</td>\n')
                    
                    if 'profit' in trade:
                        f.write(f'<td class="{profit_class}">${profit:.2f}</td>\n')
                    else:
                        f.write('<td>-</td>\n')
                    
                    f.write(f'<td>${trade.get("balance", 0):.2f}</td>\n')
                    f.write('</tr>\n')
                
                f.write('</table>\n')
                
                # Images
                f.write('<h2>Charts</h2>\n')
                
                # Check if image files exist
                image_files = [
                    ('Equity Curve', 'equity_curve.png'),
                    ('Drawdown', 'drawdown.png'),
                    ('Monthly Returns', 'monthly_returns.png'),
                    ('Trade Analysis', 'trade_analysis.png'),
                    ('Strategy Chart', 'strategy_chart.png')
                ]
                
                for title, filename in image_files:
                    image_path = os.path.join(os.path.dirname(output_path), filename)
                    if os.path.exists(image_path):
                        f.write(f'<h3>{title}</h3>\n')
                        f.write(f'<img src="{filename}" alt="{title}" style="max-width: 100%;">\n')
                
                f.write('</body>\n')
                f.write('</html>\n')
            
            logger.info(f"Backtest report generated at {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating backtest report: {e}")
            return False


# Example usage
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from data.data_manager import DataManager
    from strategy.strategy_manager import StrategyManager
    
    # Create managers
    data_manager = DataManager(exchange_id='binance', use_testnet=True)
    strategy_manager = StrategyManager()
    
    # Create backtest manager
    backtest_manager = BacktestManager(
        strategy_manager=strategy_manager,
        data_manager=data_manager
    )
    
    # Run backtest
    results = backtest_manager.run_backtest(
        strategy_name='envelope',
        symbol='BTC/USDT',
        timeframe='1h',
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now(),
        strategy_params={
            'average_type': 'sma',
            'average_period': 20,
            'envelopes': [0.1],
            'stop_loss_pct': 0.03,
            'mode': 'both'
        },
        initial_balance=10000,
        commission=0.001,
        output_dir='backtest_results'
    )
    
    # Generate report
    backtest_manager.generate_report(
        backtest_results=results,
        output_path='backtest_results/report.html'
    )
