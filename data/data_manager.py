# src/data/data_manager.py

import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from .market_data import MarketDataFetcher
from .preprocessor import DataPreprocessor
from .storage import DataStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataManager:
    """
    Manages data operations for the crypto algotrading platform.
    
    This class:
    - Integrates data fetching, preprocessing, and storage
    - Provides a unified interface for data operations
    """
    
    def __init__(self, exchange_id: str = 'binance', use_testnet: bool = False):
        """
        Initialize the DataManager.
        
        Args:
            exchange_id (str): Exchange ID (e.g., 'binance', 'bybit')
            use_testnet (bool): Whether to use testnet/sandbox
        """
        self.exchange_id = exchange_id
        self.use_testnet = use_testnet
        
        # Initialize components
        self.data_fetcher = MarketDataFetcher(exchange_id, use_testnet)
        self.preprocessor = DataPreprocessor()
        self.storage = DataStorage()
        
        logger.info(f"Initialized DataManager for {exchange_id}")
    
    def get_data(self, 
                symbol: str, 
                timeframe: str = '1h', 
                limit: int = 1000,
                since: Optional[datetime] = None,
                add_indicators: bool = True,
                add_envelopes: bool = True,
                use_cache: bool = True) -> pd.DataFrame:
        """
        Get data for a symbol.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
            timeframe (str): Timeframe (e.g., '1m', '1h', '1d')
            limit (int): Number of candles to retrieve
            since (datetime, optional): Start time
            add_indicators (bool): Whether to add technical indicators
            add_envelopes (bool): Whether to add envelope bands
            use_cache (bool): Whether to use cached data
            
        Returns:
            pd.DataFrame: Data
        """
        # Check cache first if enabled
        if use_cache:
            cached_data = self.storage.get_data(symbol, timeframe)
            if not cached_data.empty:
                logger.info(f"Using cached data for {symbol} {timeframe}")
                return cached_data
        
        # Fetch data if not in cache or cache disabled
        data = self.data_fetcher.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
            since=since
        )
        
        if data.empty:
            logger.warning(f"No data available for {symbol} {timeframe}")
            return pd.DataFrame()
        
        # Add indicators if requested
        if add_indicators:
            data = self.preprocessor.add_indicators(data)
        
        # Add envelope bands if requested
        if add_envelopes:
            data = self.preprocessor.add_envelopes(data)
        
        # Cache data if enabled
        if use_cache:
            self.storage.save_data(data, symbol, timeframe)
        
        return data
    
    def get_latest_data(self, 
                       symbol: str, 
                       timeframe: str = '1h',
                       add_indicators: bool = True,
                       add_envelopes: bool = True) -> pd.DataFrame:
        """
        Get latest data for a symbol.
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT')
            timeframe (str): Timeframe (e.g., '1m', '1h', '1d')
            add_indicators (bool): Whether to add technical indicators
            add_envelopes (bool): Whether to add envelope bands
            
        Returns:
            pd.DataFrame: Latest data
        """
        # Get ticker
        ticker = self.data_fetcher.get_ticker(symbol)
        
        if not ticker:
            logger.warning(f"No ticker available for {symbol}")
            return pd.DataFrame()
        
        # Get historical data
        data = self.get_data(
            symbol=symbol,
            timeframe=timeframe,
            limit=100,
            add_indicators=add_indicators,
            add_envelopes=add_envelopes,
            use_cache=True
        )
        
        if data.empty:
            logger.warning(f"No historical data available for {symbol} {timeframe}")
            return pd.DataFrame()
        
        # Update last row with latest ticker data
        last_row = data.iloc[-1].copy()
        last_row['close'] = ticker['last']
        last_row['high'] = max(last_row['high'], ticker['last'])
        last_row['low'] = min(last_row['low'], ticker['last'])
        
        # Create new DataFrame with updated last row
        updated_data = data.copy()
        updated_data.iloc[-1] = last_row
        
        # Recalculate indicators and envelopes
        if add_indicators:
            updated_data = self.preprocessor.add_indicators(updated_data)
        
        if add_envelopes:
            updated_data = self.preprocessor.add_envelopes(updated_data)
        
        return updated_data
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available trading symbols.
        
        Returns:
            List[str]: Available symbols
        """
        return self.data_fetcher.get_available_symbols()
    
    def get_exchange_info(self) -> Dict[str, Any]:
        """
        Get exchange information.
        
        Returns:
            Dict[str, Any]: Exchange information
        """
        return self.data_fetcher.get_exchange_info()
