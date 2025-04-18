# src/strategy/envelope.py

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from .base import BaseStrategy

class EnvelopeStrategy(BaseStrategy):
    """
    Envelope Trading Strategy.
    
    This strategy:
    - Uses different types of moving averages (SMA, EMA, WMA, Donchian)
    - Creates upper and lower bands (envelopes) around these averages
    - Generates buy signals when price crosses below the lower band
    - Generates sell signals when price crosses above the upper band
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the EnvelopeStrategy.
        
        Args:
            params (Dict[str, Any], optional): Strategy parameters
        """
        super().__init__(params)
        self.name = "envelope"
        
        # Default parameters
        self.default_params = {
            'average_type': 'sma',  # 'sma', 'ema', 'wma', 'dcm'
            'average_period': 20,
            'envelopes': [0.1],  # Percentage deviation from average
            'stop_loss_pct': 0.03,
            'price_jump_pct': 0.01,
            'position_size_percentage': 100,
            'mode': 'both'  # 'long', 'short', or 'both'
        }
        
        # Merge with provided parameters
        self.params = {**self.default_params, **(params or {})}
    
    def populate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the data.
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: Data with indicators
        """
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Calculate moving average based on selected type
        average_type = self.params['average_type']
        period = self.params['average_period']
        
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
            raise ValueError(f"Unknown average type: {average_type}")
        
        # Calculate envelope bands
        for envelope in self.params['envelopes']:
            band_high = f'band_high_{envelope * 100:.0f}'
            band_low = f'band_low_{envelope * 100:.0f}'
            
            df[band_high] = df['average'] * (1 + envelope)
            df[band_low] = df['average'] * (1 - envelope)
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals.
        
        Args:
            data (pd.DataFrame): Price data with indicators
            
        Returns:
            pd.DataFrame: Data with signals
        """
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Add indicators if not already present
        if 'average' not in df.columns:
            df = self.populate_indicators(df)
        
        # Initialize signal columns
        df['buy_signal'] = False
        df['sell_signal'] = False
        
        # Get trading mode
        mode = self.params['mode']
        
        # Get first envelope (we use the first one for signals)
        envelope = self.params['envelopes'][0]
        band_high = f'band_high_{envelope * 100:.0f}'
        band_low = f'band_low_{envelope * 100:.0f}'
        
        # Calculate price jump threshold
        price_jump_pct = self.params['price_jump_pct']
        
        # Generate signals
        for i in range(1, len(df)):
            # Check for buy signals (price crosses below lower band)
            if (mode in ['long', 'both'] and 
                df['close'].iloc[i-1] >= df[band_low].iloc[i-1] and 
                df['close'].iloc[i] < df[band_low].iloc[i]):
                
                # Check for significant price jump
                price_change = abs(df['close'].iloc[i] / df['close'].iloc[i-1] - 1)
                if price_change >= price_jump_pct:
                    df.loc[df.index[i], 'buy_signal'] = True
            
            # Check for sell signals (price crosses above upper band)
            if (mode in ['short', 'both'] and 
                df['close'].iloc[i-1] <= df[band_high].iloc[i-1] and 
                df['close'].iloc[i] > df[band_high].iloc[i]):
                
                # Check for significant price jump
                price_change = abs(df['close'].iloc[i] / df['close'].iloc[i-1] - 1)
                if price_change >= price_jump_pct:
                    df.loc[df.index[i], 'sell_signal'] = True
        
        return df
    
    def calculate_position_size(self, data: pd.DataFrame, balance: float) -> float:
        """
        Calculate position size based on available balance and risk settings.
        
        Args:
            data (pd.DataFrame): Price data with indicators
            balance (float): Available balance
            
        Returns:
            float: Position size
        """
        # Get position size percentage
        position_size_pct = self.params['position_size_percentage'] / 100
        
        # Calculate position size
        position_size = balance * position_size_pct
        
        return position_size
