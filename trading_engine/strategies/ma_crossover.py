"""
Moving Average Crossover trading strategy implementation.
"""
import pandas as pd
import numpy as np

from trading_engine.strategies.base import BaseStrategy, SignalType
from trading_engine.utils.logging import get_logger

logger = get_logger(__name__)


class MACrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover trading strategy.
    
    This strategy generates buy signals when the fast moving average crosses above
    the slow moving average, and sell signals when it crosses below.
    """
    
    def __init__(self, symbols, fast_period=20, slow_period=50, signal_period=9, **kwargs):
        """
        Initialize the Moving Average Crossover strategy.
        
        Args:
            symbols (list): List of symbols to trade
            fast_period (int, optional): Fast moving average period. Defaults to 20.
            slow_period (int, optional): Slow moving average period. Defaults to 50.
            signal_period (int, optional): Signal line period for MACD. Defaults to 9.
            **kwargs: Additional strategy parameters
        """
        super().__init__(symbols, **kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
        logger.info(f"Initialized MA Crossover strategy with fast_period={fast_period}, "
                    f"slow_period={slow_period}, signal_period={signal_period}")
    
    def generate_signals(self, data):
        """
        Generate trading signals based on moving average crossovers.
        
        Args:
            data (pandas.DataFrame): Market data with MultiIndex (symbol, timestamp)
            
        Returns:
            pandas.DataFrame: DataFrame with signals for each symbol and timestamp
        """
        # Extract close prices
        if 'close' in data.columns:
            close_prices = data['close']
        else:
            close_prices = data.xs('close', level=1, axis=1)
        
        # Calculate fast and slow moving averages
        fast_ma = close_prices.rolling(window=self.fast_period).mean()
        slow_ma = close_prices.rolling(window=self.slow_period).mean()
        
        # Calculate MACD line (difference between fast and slow MAs)
        macd = fast_ma - slow_ma
        
        # Calculate signal line (moving average of MACD)
        signal_line = macd.rolling(window=self.signal_period).mean()
        
        # Calculate MACD histogram (difference between MACD and signal line)
        histogram = macd - signal_line
        
        # Initialize signals DataFrame with HOLD
        signals = pd.DataFrame(index=close_prices.index, columns=close_prices.columns, data=SignalType.HOLD)
        
        # Calculate crossovers (fast MA crosses above slow MA)
        crossovers = ((fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1)))
        
        # Calculate crossunders (fast MA crosses below slow MA)
        crossunders = ((fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1)))
        
        # Buy signal on crossover
        signals[crossovers] = SignalType.BUY
        
        # Sell signal on crossunder
        signals[crossunders] = SignalType.SELL
        
        # No signal for the first slow_period days (insufficient data)
        signals.iloc[:self.slow_period] = SignalType.HOLD
        
        # Convert signal values to readable strings for logging
        signal_counts = {
            'BUY': (signals == SignalType.BUY).sum().sum(),
            'SELL': (signals == SignalType.SELL).sum().sum(),
            'HOLD': (signals == SignalType.HOLD).sum().sum(),
        }
        
        logger.info(f"Generated {len(signals)} signals: {signal_counts}")
        
        # Add moving averages and MACD values to the signals DataFrame
        result = pd.DataFrame(index=signals.index)
        
        for symbol in signals.columns:
            result[f'{symbol}_signal'] = signals[symbol]
            result[f'{symbol}_fast_ma'] = fast_ma[symbol]
            result[f'{symbol}_slow_ma'] = slow_ma[symbol]
            result[f'{symbol}_macd'] = macd[symbol]
            result[f'{symbol}_signal_line'] = signal_line[symbol]
            result[f'{symbol}_histogram'] = histogram[symbol]
        
        return result
    
    def calculate_position_sizes(self, signals, portfolio_value, max_position_size=0.2):
        """
        Calculate position sizes based on signals and portfolio value.
        
        Args:
            signals (pandas.DataFrame): Signal DataFrame
            portfolio_value (float): Current portfolio value
            max_position_size (float, optional): Maximum position size as a fraction of portfolio. Defaults to 0.2.
            
        Returns:
            dict: Dictionary mapping symbols to position sizes (in dollars)
        """
        position_sizes = {}
        
        # Count buy signals
        buy_signals = sum(1 for col in signals.columns if col.endswith('_signal') and signals[col].iloc[-1] == SignalType.BUY)
        
        if buy_signals > 0:
            # Equal position sizing
            position_value = min(portfolio_value / buy_signals, portfolio_value * max_position_size)
            
            for symbol in self.symbols:
                signal_col = f'{symbol}_signal'
                histogram_col = f'{symbol}_histogram'
                
                if signal_col in signals.columns:
                    if signals[signal_col].iloc[-1] == SignalType.BUY:
                        # Optionally scale by MACD histogram strength
                        if histogram_col in signals.columns:
                            histogram_value = signals[histogram_col].iloc[-1]
                            # Scale up to 150% for strong histogram values
                            scale_factor = min(1.0 + (histogram_value / max(abs(signals[histogram_col].dropna()))), 1.5)
                            position_sizes[symbol] = position_value * scale_factor
                        else:
                            position_sizes[symbol] = position_value
                    elif signals[signal_col].iloc[-1] == SignalType.SELL:
                        position_sizes[symbol] = 0
        
        return position_sizes