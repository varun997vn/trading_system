"""
Momentum trading strategy implementation.
"""
import pandas as pd
import numpy as np

from trading_engine.strategies.base import BaseStrategy, SignalType
from trading_engine.utils.logging import get_logger

logger = get_logger(__name__)


class MomentumStrategy(BaseStrategy):
    """
    Momentum trading strategy based on price movement.
    
    This strategy generates buy signals when the asset's return over the lookback period
    exceeds the threshold, and sell signals when it falls below the negative threshold.
    """
    
    def __init__(self, symbols, lookback_period=20, threshold=0.05, **kwargs):
        """
        Initialize the momentum strategy.
        
        Args:
            symbols (list): List of symbols to trade
            lookback_period (int, optional): Lookback period in days. Defaults to 20.
            threshold (float, optional): Return threshold for signal generation. Defaults to 0.05 (5%).
            **kwargs: Additional strategy parameters
        """
        super().__init__(symbols, **kwargs)
        self.lookback_period = lookback_period
        self.threshold = threshold
        
        logger.info(f"Initialized Momentum strategy with lookback_period={lookback_period}, threshold={threshold}")
    
    def generate_signals(self, data):
        """
        Generate trading signals based on price momentum.
        
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
        
        # Calculate momentum (rolling returns)
        momentum = close_prices.pct_change(self.lookback_period)
        
        # Generate signals based on momentum and threshold
        signals = pd.DataFrame(index=momentum.index, columns=momentum.columns, data=SignalType.HOLD)
        
        # Buy signal when momentum > threshold
        signals[momentum > self.threshold] = SignalType.BUY
        
        # Sell signal when momentum < -threshold
        signals[momentum < -self.threshold] = SignalType.SELL
        
        # No signal for the first lookback_period days
        signals.iloc[:self.lookback_period] = SignalType.HOLD
        
        # Convert signal values to readable strings for logging
        signal_counts = {
            'BUY': (signals == SignalType.BUY).sum().sum(),
            'SELL': (signals == SignalType.SELL).sum().sum(),
            'HOLD': (signals == SignalType.HOLD).sum().sum(),
        }
        
        logger.info(f"Generated {len(signals)} signals: {signal_counts}")
        
        # Add momentum values to the signals DataFrame
        result = pd.DataFrame(index=signals.index)
        
        for symbol in signals.columns:
            result[f'{symbol}_signal'] = signals[symbol]
            result[f'{symbol}_momentum'] = momentum[symbol]
        
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
                signal = signals.get(f'{symbol}_signal', None)
                
                if signal is not None and signal.iloc[-1] == SignalType.BUY:
                    position_sizes[symbol] = position_value
                elif signal is not None and signal.iloc[-1] == SignalType.SELL:
                    position_sizes[symbol] = 0
        
        return position_sizes