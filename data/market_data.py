# src/data/market_data.py

import logging
import time
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import ccxt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MarketDataFetcher:
    """
    Fetches market data from cryptocurrency exchanges.
    
    This class:
    - Connects to exchanges using CCXT
    - Retrieves historical and real-time price data
    - Handles rate limiting and error recovery
    """
    
    def __init__(self, exchange_id: str = 'binance', use_testnet: bool = False):
        """
        Initialize the MarketDataFetcher.
        
        Args:
            exchange_id (str): Exchange ID (e.g., 'binance', 'bybit')
            use_testnet (bool): Whether to use testnet/sandbox
        """
        self.exchange_id = exchange_id
        self.use_testnet = use_testnet
        
        # Initialize exchange
        self._init_exchange()
        
        logger.info(f"Initialized MarketDataFetcher for {exchange_id}")
    
    def _init_exchange(self) -> None:
        """
        Initialize the exchange connection.
        """
        try:
            # Create exchange instance
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'
                }
            })
            
            # Use testnet/sandbox if specified
            if self.use_testnet and self.exchange.has['test']:
                self.exchange.set_sandbox_mode(True)
                logger.info(f"Using {self.exchange_id} testnet/sandbox")
            
            # Load markets
            self.exchange.load_markets()
            
            logger.info(f"Connected to {self.exchange_id}")
            
        except Exception as e:
            logger.error(f"Error initializing exchange: {e}")
            raise
    
    def get_historical_data(self, 
                           symbol: str, 
                           timeframe: str = '1h', 
                           limit: int = 1000,
                           since: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get historical OHLCV data.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
            timeframe (str): Timeframe (e.g., '1m', '1h', '1d')
            limit (int): Number of candles to retrieve
            since (datetime, optional): Start time
            
        Returns:
            pd.DataFrame: Historical data
        """
        try:
            # Convert since to timestamp if provided
            since_timestamp = int(since.timestamp() * 1000) if since else None
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                since=since_timestamp
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Retrieved {len(df)} {timeframe} candles for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get current ticker data.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
            
        Returns:
            Dict[str, Any]: Ticker data
        """
        try:
            # Fetch ticker
            ticker = self.exchange.fetch_ticker(symbol)
            
            logger.info(f"Retrieved ticker for {symbol}")
            return ticker
            
        except Exception as e:
            logger.error(f"Error fetching ticker: {e}")
            return {}
    
    def get_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """
        Get order book data.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
            limit (int): Number of orders to retrieve
            
        Returns:
            Dict[str, Any]: Order book data
        """
        try:
            # Fetch order book
            order_book = self.exchange.fetch_order_book(symbol, limit)
            
            logger.info(f"Retrieved order book for {symbol}")
            return order_book
            
        except Exception as e:
            logger.error(f"Error fetching order book: {e}")
            return {}
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available trading symbols.
        
        Returns:
            List[str]: Available symbols
        """
        try:
            # Get markets
            markets = self.exchange.markets
            
            # Extract symbols
            symbols = [market for market in markets.keys()]
            
            logger.info(f"Retrieved {len(symbols)} available symbols")
            return symbols
            
        except Exception as e:
            logger.error(f"Error fetching available symbols: {e}")
            return []
    
    def get_exchange_info(self) -> Dict[str, Any]:
        """
        Get exchange information.
        
        Returns:
            Dict[str, Any]: Exchange information
        """
        try:
            # Get exchange info
            info = {
                'id': self.exchange.id,
                'name': self.exchange.name,
                'countries': self.exchange.countries,
                'urls': self.exchange.urls,
                'version': self.exchange.version,
                'has': self.exchange.has,
                'timeframes': self.exchange.timeframes,
                'timeout': self.exchange.timeout,
                'rateLimit': self.exchange.rateLimit,
                'userAgent': self.exchange.userAgent,
                'symbols': len(self.exchange.symbols),
                'currencies': len(self.exchange.currencies),
            }
            
            logger.info(f"Retrieved exchange info for {self.exchange_id}")
            return info
            
        except Exception as e:
            logger.error(f"Error fetching exchange info: {e}")
            return {}

