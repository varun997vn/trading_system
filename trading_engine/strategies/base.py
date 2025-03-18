"""
Base trading strategy implementation.
"""
import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from trading_engine.utils.logging import get_logger

logger = get_logger(__name__)


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    """
    
    def __init__(self, symbols, **kwargs):
        """
        Initialize the base strategy.
        
        Args:
            symbols (list): List of symbols to trade
            **kwargs: Additional strategy parameters
        """
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.positions = {}  # Current positions {symbol: quantity}
        self.parameters = kwargs  # Strategy-specific parameters
        
        logger.info(f"Initialized {self.__class__.__name__} for {len(self.symbols)} symbols")
    
    @abstractmethod
    def generate_signals(self, data):
        """
        Generate trading signals for the given data.
        
        Args:
            data (pandas.DataFrame): Market data with MultiIndex (symbol, timestamp)
            
        Returns:
            pandas.DataFrame: DataFrame with signals for each symbol and timestamp
        """
        pass
    
    def set_position(self, symbol, quantity):
        """
        Set the current position for a symbol.
        
        Args:
            symbol (str): Symbol
            quantity (float): Position quantity
        """
        self.positions[symbol] = quantity
        logger.debug(f"Set position for {symbol}: {quantity}")
    
    def get_position(self, symbol):
        """
        Get the current position for a symbol.
        
        Args:
            symbol (str): Symbol
            
        Returns:
            float: Position quantity (0 if no position)
        """
        return self.positions.get(symbol, 0)
    
    def calculate_returns(self, data, window=252):
        """
        Calculate returns statistics for the given data.
        
        Args:
            data (pandas.DataFrame): Market data with MultiIndex (symbol, timestamp)
            window (int, optional): Rolling window size for volatility calculation. Defaults to 252.
            
        Returns:
            pandas.DataFrame: DataFrame with return statistics
        """
        # Calculate daily returns
        returns = data.xs('close', level=1, axis=1).pct_change()
        
        # Calculate rolling volatility (annualized)
        volatility = returns.rolling(window=window).std() * np.sqrt(window)
        
        # Calculate rolling Sharpe ratio (assuming 0% risk-free rate)
        sharpe = (returns.rolling(window=window).mean() * window) / volatility
        
        # Combine results
        result = pd.DataFrame(index=returns.index)
        result['daily_return'] = returns
        result['volatility'] = volatility
        result['sharpe_ratio'] = sharpe
        
        return result
    
    def update_parameters(self, **kwargs):
        """
        Update strategy parameters.
        
        Args:
            **kwargs: Parameters to update
        """
        self.parameters.update(kwargs)
        logger.info(f"Updated {self.__class__.__name__} parameters: {kwargs}")


class SignalType:
    """
    Enumeration of signal types.
    """
    BUY = 1
    SELL = -1
    HOLD = 0
    
    @staticmethod
    def to_string(signal):
        """
        Convert a signal value to a string representation.
        
        Args:
            signal (int): Signal value
            
        Returns:
            str: String representation
        """
        if signal == SignalType.BUY:
            return "BUY"
        elif signal == SignalType.SELL:
            return "SELL"
        else:
            return "HOLD"
    
    @staticmethod
    def from_string(signal_str):
        """
        Convert a string representation to a signal value.
        
        Args:
            signal_str (str): String representation
            
        Returns:
            int: Signal value
        """
        signal_str = signal_str.upper()
        if signal_str == "BUY":
            return SignalType.BUY
        elif signal_str == "SELL":
            return SignalType.SELL
        else:
            return SignalType.HOLD