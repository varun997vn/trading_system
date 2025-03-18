"""
Mean reversion trading strategy implementation.
"""
import pandas as pd
import numpy as np

from trading_engine.strategies.base import BaseStrategy, SignalType
from trading_engine.utils.logging import get_logger

logger = get_logger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion trading strategy based on z-score deviations from a moving average.
    
    This strategy generates buy signals when the asset's price falls significantly below
    its moving average (low z-score) and sell signals when it rises significantly above
    its moving average (high z-score).
    """
    
    def __init__(self, symbols, lookback_period=30, z_score_threshold=2.0, **kwargs):
        """
        Initialize the mean reversion strategy.
        
        Args:
            symbols (list): List of symbols to trade
            lookback_period (int, optional): Lookback period in days for moving average. Defaults to 30.
            z_score_threshold (float, optional): Z-score threshold for signal generation. Defaults to 2.0.
            **kwargs: Additional strategy parameters
        """
        super().__init__(symbols, **kwargs)
        self.lookback_period = lookback_period
        self.z_score_threshold = z_score_threshold
        
        logger.info(f"Initialized Mean Reversion strategy with lookback_period={lookback_period}, "
                    f"z_score_threshold={z_score_threshold}")
    
    def generate_signals(self, data):
        """
        Generate trading signals based on mean reversion.
        
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
        
        # Calculate moving average
        moving_avg = close_prices.rolling(window=self.lookback_period).mean()
        
        # Calculate std dev of price
        rolling_std = close_prices.rolling(window=self.lookback_period).std()
        
        # Calculate z-score (distance from mean in terms of standard deviations)
        z_score = (close_prices - moving_avg) / rolling_std
        
        # Initialize signals DataFrame with HOLD
        signals = pd.DataFrame(index=z_score.index, columns=z_score.columns, data=SignalType.HOLD)
        
        # Buy signal when z-score < -threshold (price significantly below mean)
        signals[z_score < -self.z_score_threshold] = SignalType.BUY
        
        # Sell signal when z-score > threshold (price significantly above mean)
        signals[z_score > self.z_score_threshold] = SignalType.SELL
        
        # No signal for the first lookback_period days (insufficient data)
        signals.iloc[:self.lookback_period] = SignalType.HOLD
        
        # Convert signal values to readable strings for logging
        signal_counts = {
            'BUY': (signals == SignalType.BUY).sum().sum(),
            'SELL': (signals == SignalType.SELL).sum().sum(),
            'HOLD': (signals == SignalType.HOLD).sum().sum(),
        }
        
        logger.info(f"Generated {len(signals)} signals: {signal_counts}")
        
        # Add z-score values to the signals DataFrame
        result = pd.DataFrame(index=signals.index)
        
        for symbol in signals.columns:
            result[f'{symbol}_signal'] = signals[symbol]
            result[f'{symbol}_z_score'] = z_score[symbol]
            result[f'{symbol}_mean'] = moving_avg[symbol]
        
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
                z_score_col = f'{symbol}_z_score'
                
                if signal_col in signals.columns and signals[signal_col].iloc[-1] == SignalType.BUY:
                    # Scale position size by z-score magnitude (optional)
                    z_score = abs(signals[z_score_col].iloc[-1]) if z_score_col in signals.columns else self.z_score_threshold
                    scale_factor = min(z_score / self.z_score_threshold, 1.5)  # Cap at 150%
                    
                    position_sizes[symbol] = position_value * scale_factor
                elif signal_col in signals.columns and signals[signal_col].iloc[-1] == SignalType.SELL:
                    position_sizes[symbol] = 0
        
        return position_sizes