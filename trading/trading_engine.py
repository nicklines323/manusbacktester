# src/trading/trading_engine.py

import logging
import time
import threading
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from ..data.data_manager import DataManager
from ..strategy.strategy_manager import StrategyManager
from .exchange import ExchangeConnector
from .order_manager import OrderManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingEngine:
    """
    Trading engine for executing live trades.
    
    This class:
    - Monitors market data in real-time
    - Generates trading signals using strategies
    - Executes trades based on signals
    - Manages positions and risk
    """
    
    def __init__(self, 
                strategy_manager=None, 
                data_manager=None,
                exchange_connector=None,
                order_manager=None):
        """
        Initialize the TradingEngine.
        
        Args:
            strategy_manager: Strategy manager instance
            data_manager: Data manager instance
            exchange_connector: Exchange connector instance
            order_manager: Order manager instance
        """
        self.strategy_manager = strategy_manager or StrategyManager()
        self.data_manager = data_manager or DataManager()
        self.exchange_connector = exchange_connector or ExchangeConnector()
        self.order_manager = order_manager or OrderManager(self.exchange_connector)
        
        self.running = False
        self.trading_thread = None
        self.stop_event = threading.Event()
        
        self.symbol = None
        self.timeframe = None
        self.strategy = None
        self.strategy_params = None
        
        self.data = None
        self.position = None
        self.trades = []
        self.balance = 0
        
        logger.info("Initialized TradingEngine")
    
    def start_trading(self, 
                     strategy_name: str,
                     symbol: str,
                     timeframe: str = '1h',
                     strategy_params: Optional[Dict[str, Any]] = None) -> bool:
        """
        Start trading.
        
        Args:
            strategy_name (str): Name of the strategy
            symbol (str): Trading symbol
            timeframe (str): Timeframe
            strategy_params (Dict[str, Any], optional): Strategy parameters
            
        Returns:
            bool: Whether trading was started successfully
        """
        if self.running:
            logger.warning("Trading is already running")
            return False
        
        try:
            # Set parameters
            self.symbol = symbol
            self.timeframe = timeframe
            self.strategy_params = strategy_params
            
            # Get strategy
            self.strategy = self.strategy_manager.get_strategy(
                strategy_name=strategy_name,
                params=strategy_params
            )
            
            # Get initial data
            self.data = self.data_manager.get_data(
                symbol=symbol,
                timeframe=timeframe,
                limit=100,
                add_indicators=False,
                add_envelopes=False
            )
            
            if self.data.empty:
                logger.error(f"No data available for {symbol} {timeframe}")
                return False
            
            # Add indicators and signals
            self.data = self.strategy.generate_signals(self.data)
            
            # Get account balance
            account_info = self.exchange_connector.get_account_info()
            self.balance = account_info.get('balance', 0)
            
            # Get current position
            self.position = self.exchange_connector.get_position(symbol)
            
            # Reset trades
            self.trades = []
            
            # Set running flag
            self.running = True
            
            # Reset stop event
            self.stop_event.clear()
            
            # Start trading thread
            self.trading_thread = threading.Thread(target=self._trading_loop)
            self.trading_thread.daemon = True
            self.trading_thread.start()
            
            logger.info(f"Started trading {strategy_name} on {symbol} {timeframe}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting trading: {e}")
            self.running = False
            return False
    
    def stop_trading(self) -> bool:
        """
        Stop trading.
        
        Returns:
            bool: Whether trading was stopped successfully
        """
        if not self.running:
            logger.warning("Trading is not running")
            return False
        
        try:
            # Set stop event
            self.stop_event.set()
            
            # Wait for trading thread to stop
            if self.trading_thread and self.trading_thread.is_alive():
                self.trading_thread.join(timeout=10)
            
            # Set running flag
            self.running = False
            
            logger.info("Stopped trading")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping trading: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get trading status.
        
        Returns:
            Dict[str, Any]: Trading status
        """
        if not self.running:
            return {}
        
        try:
            # Get current price
            ticker = self.data_manager.data_fetcher.get_ticker(self.symbol)
            current_price = ticker.get('last', 0)
            
            # Get position
            position = self.position or {}
            
            # Calculate unrealized P/L
            unrealized_pl = 0
            
            if position and position.get('size', 0) > 0:
                entry_price = position.get('entry_price', 0)
                size = position.get('size', 0)
                
                if position.get('type') == 'long':
                    unrealized_pl = (current_price - entry_price) * size
                else:
                    unrealized_pl = (entry_price - current_price) * size
                
                # Update position with unrealized P/L
                position['unrealized_pl'] = unrealized_pl
            
            # Create status
            status = {
                'running': self.running,
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'current_price': current_price,
                'balance': self.balance,
                'position': position,
                'trades': self.trades,
                'data': self.data
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting trading status: {e}")
            return {}
    
    def _trading_loop(self) -> None:
        """
        Trading loop.
        """
        logger.info("Starting trading loop")
        
        try:
            while not self.stop_event.is_set():
                # Get latest data
                latest_data = self.data_manager.get_latest_data(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    add_indicators=False,
                    add_envelopes=False
                )
                
                if latest_data.empty:
                    logger.warning(f"No data available for {self.symbol} {self.timeframe}")
                    time.sleep(5)
                    continue
                
                # Update data
                self.data = latest_data
                
                # Add indicators and signals
                self.data = self.strategy.generate_signals(self.data)
                
                # Check for signals
                self._check_signals()
                
                # Update position
                self.position = self.exchange_connector.get_position(self.symbol)
                
                # Update account balance
                account_info = self.exchange_connector.get_account_info()
                self.balance = account_info.get('balance', 0)
                
                # Sleep
                time.sleep(5)
                
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            self.running = False
    
    def _check_signals(self) -> None:
        """
        Check for trading signals.
        """
        try:
            # Get latest row
            latest_row = self.data.iloc[-1]
            
            # Get current position
            current_position = self.position or {}
            position_size = current_position.get('size', 0)
            position_type = current_position.get('type', '')
            
            # Check for buy signal
            if latest_row['buy_signal'] and (position_size == 0 or position_type == 'short'):
                # Close short position if exists
                if position_size > 0 and position_type == 'short':
                    self._close_position()
                
                # Calculate position size
                position_size = self.strategy.calculate_position_size(self.data, self.balance)
                
                # Open long position
                order = self.order_manager.create_market_order(
                    symbol=self.symbol,
                    side='buy',
                    amount=position_size / latest_row['close']
                )
                
                if order:
                    # Record trade
                    trade = {
                        'date': datetime.now(),
                        'type': 'buy',
                        'price': latest_row['close'],
                        'position': position_size / latest_row['close'],
                        'balance': self.balance
                    }
                    
                    self.trades.append(trade)
                    
                    logger.info(f"Opened long position: {position_size / latest_row['close']} {self.symbol} at {latest_row['close']}")
            
            # Check for sell signal
            elif latest_row['sell_signal'] and (position_size == 0 or position_type == 'long'):
                # Close long position if exists
                if position_size > 0 and position_type == 'long':
                    self._close_position()
                
                # Calculate position size
                position_size = self.strategy.calculate_position_size(self.data, self.balance)
                
                # Open short position
                order = self.order_manager.create_market_order(
                    symbol=self.symbol,
                    side='sell',
                    amount=position_size / latest_row['close']
                )
                
                if order:
                    # Record trade
                    trade = {
                        'date': datetime.now(),
                        'type': 'sell',
                        'price': latest_row['close'],
                        'position': position_size / latest_row['close'],
                        'balance': self.balance
                    }
                    
                    self.trades.append(trade)
                    
                    logger.info(f"Opened short position: {position_size / latest_row['close']} {self.symbol} at {latest_row['close']}")
            
            # Check for stop loss
            elif position_size > 0:
                entry_price = current_position.get('entry_price', 0)
                stop_loss_pct = self.strategy_params.get('stop_loss_pct', 0.03)
                
                if position_type == 'long' and latest_row['close'] < entry_price * (1 - stop_loss_pct):
                    # Close long position
                    self._close_position()
                    
                    logger.info(f"Closed long position due to stop loss: {position_size} {self.symbol} at {latest_row['close']}")
                
                elif position_type == 'short' and latest_row['close'] > entry_price * (1 + stop_loss_pct):
                    # Close short position
                    self._close_position()
                    
                    logger.info(f"Closed short position due to stop loss: {position_size} {self.symbol} at {latest_row['close']}")
            
        except Exception as e:
            logger.error(f"Error checking signals: {e}")
    
    def _close_position(self) -> None:
        """
        Close current position.
        """
        try:
            # Get current position
            current_position = self.position or {}
            position_size = current_position.get('size', 0)
            position_type = current_position.get('type', '')
            
            if position_size == 0:
                logger.warning("No position to close")
                return
            
            # Close position
            if position_type == 'long':
                order = self.order_manager.create_market_order(
                    symbol=self.symbol,
                    side='sell',
                    amount=position_size
                )
                
                if order:
                    # Calculate profit
                    entry_price = current_position.get('entry_price', 0)
                    exit_price = self.data.iloc[-1]['close']
                    profit = (exit_price - entry_price) * position_size
                    
                    # Record trade
                    trade = {
                        'date': datetime.now(),
                        'type': 'close_long',
                        'price': exit_price,
                        'position': position_size,
                        'profit': profit,
                        'balance': self.balance + profit
                    }
                    
                    self.trades.append(trade)
                    
                    logger.info(f"Closed long position: {position_size} {self.symbol} at {exit_price}, profit: {profit}")
            
            elif position_type == 'short':
                order = self.order_manager.create_market_order(
                    symbol=self.symbol,
                    side='buy',
                    amount=position_size
                )
                
                if order:
                    # Calculate profit
                    entry_price = current_position.get('entry_price', 0)
                    exit_price = self.data.iloc[-1]['close']
                    profit = (entry_price - exit_price) * position_size
                    
                    # Record trade
                    trade = {
                        'date': datetime.now(),
                        'type': 'close_short',
                        'price': exit_price,
                        'position': position_size,
                        'profit': profit,
                        'balance': self.balance + profit
                    }
                    
                    self.trades.append(trade)
                    
                    logger.info(f"Closed short position: {position_size} {self.symbol} at {exit_price}, profit: {profit}")
            
            # Reset position
            self.position = None
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
