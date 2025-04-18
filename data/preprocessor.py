# src/data/preprocessor.py

import logging
from typing import Dict, List, Optional, Union, Any

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocesses market data.
    
    This class:
    - Cleans and formats data
    - Adds technical indicators
    - Creates envelope bands
    """
    
    def __init__(self):
        """
        Initialize the DataPreprocessor.
        """
        logger.info("Initialized DataPreprocessor")
    
    def add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the data.
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: Data with indicators
        """
        try:
            # Make a copy to avoid modifying the original data
            df = data.copy()
            
            # Add Simple Moving Average (SMA)
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['sma_200'] = df['close'].rolling(window=200).mean()
            
            # Add Exponential Moving Average (EMA)
            df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
            
            # Add Relative Strength Index (RSI)
            df['rsi_14'] = self._calculate_rsi(df['close'], window=14)
            
            # Add Moving Average Convergence Divergence (MACD)
            df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(df['close'])
            
            # Add Bollinger Bands
            df['bb_middle'], df['bb_upper'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
            
            # Add Average True Range (ATR)
            df['atr_14'] = self._calculate_atr(df, window=14)
            
            logger.info(f"Added indicators to data: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error adding indicators: {e}")
            return data
    
    def add_envelopes(self, data: pd.DataFrame, average_type: str = 'sma', period: int = 20, envelopes: List[float] = [0.1]) -> pd.DataFrame:
        """
        Add envelope bands to the data.
        
        Args:
            data (pd.DataFrame): Price data
            average_type (str): Type of moving average ('sma', 'ema', 'wma', 'dcm')
            period (int): Period for the moving average
            envelopes (List[float]): List of envelope percentages
            
        Returns:
            pd.DataFrame: Data with envelope bands
        """
        try:
            # Make a copy to avoid modifying the original data
            df = data.copy()
            
            # Calculate moving average based on selected type
            if average_type == 'sma':
                # Simple Moving Average
                df['average'] = df['close'].rolling(window=period).mean()
            elif average_type == 'ema':
                # Exponential Moving Average
                df['average'] = df['close'].ewm(span=period, adjust=False).mean()
            elif average_type == 'wma':
                # Weighted Moving Average
                weights = np.arange(1, period + 1)
                df['average'] = df['close'].rolling(window=period).apply(
                    lambda x: np.sum(weights * x) / np.sum(weights), raw=True
                )
            elif average_type == 'dcm':
                # Donchian Channel Middle
                df['highest_high'] = df['high'].rolling(window=period).max()
                df['lowest_low'] = df['low'].rolling(window=period).min()
                df['average'] = (df['highest_high'] + df['lowest_low']) / 2
            else:
                logger.warning(f"Unknown average type: {average_type}, using SMA")
                df['average'] = df['close'].rolling(window=period).mean()
            
            # Calculate envelope bands
            for envelope in envelopes:
                band_high = f'band_high_{envelope * 100:.0f}'
                band_low = f'band_low_{envelope * 100:.0f}'
                
                df[band_high] = df['average'] * (1 + envelope)
                df[band_low] = df['average'] * (1 - envelope)
            
            logger.info(f"Added envelope bands to data: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error adding envelope bands: {e}")
            return data
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices (pd.Series): Price series
            window (int): Window size
            
        Returns:
            pd.Series: RSI values
        """
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Calculate RS
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple:
        """
        Calculate Moving Average Convergence Divergence (MACD).
        
        Args:
            prices (pd.Series): Price series
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
            signal_period (int): Signal EMA period
            
        Returns:
            tuple: (MACD, Signal, Histogram)
        """
        # Calculate EMAs
        fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
        slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD
        macd = fast_ema - slow_ema
        
        # Calculate Signal
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        
        # Calculate Histogram
        histogram = macd - signal
        
        return macd, signal, histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> tuple:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices (pd.Series): Price series
            window (int): Window size
            num_std (float): Number of standard deviations
            
        Returns:
            tuple: (Middle Band, Upper Band, Lower Band)
        """
        # Calculate middle band (SMA)
        middle_band = prices.rolling(window=window).mean()
        
        # Calculate standard deviation
        std = prices.rolling(window=window).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        
        return middle_band, upper_band, lower_band
    
    def _calculate_atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            data (pd.DataFrame): Price data
            window (int): Window size
            
        Returns:
            pd.Series: ATR values
        """
        # Calculate True Range
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = tr.rolling(window=window).mean()
        
        return atr
