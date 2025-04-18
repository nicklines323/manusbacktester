# src/strategy/strategy_manager.py

import logging
from typing import Dict, List, Optional, Union, Any

from .base import BaseStrategy
from .envelope import EnvelopeStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StrategyManager:
    """
    Manages trading strategies.
    
    This class:
    - Creates strategy instances
    - Provides a unified interface for strategy operations
    """
    
    def __init__(self):
        """
        Initialize the StrategyManager.
        """
        logger.info("Initialized StrategyManager")
    
    def get_strategy(self, strategy_name: str, params: Optional[Dict[str, Any]] = None) -> BaseStrategy:
        """
        Get a strategy instance.
        
        Args:
            strategy_name (str): Name of the strategy
            params (Dict[str, Any], optional): Strategy parameters
            
        Returns:
            BaseStrategy: Strategy instance
        """
        try:
            # Create strategy instance based on name
            if strategy_name == 'envelope':
                strategy = EnvelopeStrategy(params)
            else:
                raise ValueError(f"Unknown strategy: {strategy_name}")
            
            logger.info(f"Created {strategy_name} strategy")
            return strategy
            
        except Exception as e:
            logger.error(f"Error creating strategy: {e}")
            raise
    
    def get_available_strategies(self) -> List[str]:
        """
        Get list of available strategies.
        
        Returns:
            List[str]: Available strategies
        """
        return ['envelope']
    
    def generate_signals(self, 
                        strategy_name: str, 
                        data: Any,
                        params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Generate signals for a strategy.
        
        Args:
            strategy_name (str): Name of the strategy
            data (Any): Data to generate signals for
            params (Dict[str, Any], optional): Strategy parameters
            
        Returns:
            Any: Data with signals
        """
        try:
            # Get strategy
            strategy = self.get_strategy(strategy_name, params)
            
            # Generate signals
            signals = strategy.generate_signals(data)
            
            logger.info(f"Generated signals for {strategy_name}")
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            raise
