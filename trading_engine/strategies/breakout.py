"""
Breakout trading strategy implementation.
"""
import pandas as pd
import numpy as np

from trading_engine.strategies.base import BaseStrategy, SignalType
from trading_engine.utils.logging import get_logger

logger = get_logger(__name__)


class BreakoutStrategy(BaseStrategy):
    """
    Breakout trading strategy based on price movements beyond support and resistance levels.
    
    This strategy generates buy signals when the price breaks above a resistance level
    (defined as the highest high over the lookback period) and sell signals when it breaks
    below a support level (defined as the lowest low over the lookback period).
    """
    
    def __init__(self, symbols, lookback_period=20, breakout_threshold=0.02, atr_periods=14, **kwargs):
        """
        Initialize the breakout strategy.
        
        Args:
            symbols (list): List of symbols to trade
            lookback_period (int, optional): Lookback period for identifying support/resistance. Defaults to 20.
            breakout_threshold (float, optional): Minimum percentage move for a valid breakout. Defaults to 0.02 (2%).
            atr_periods (int, optional): Periods for ATR calculation. Defaults to 14.
            **kwargs: Additional strategy parameters
        """
        super().__init__(symbols, **kwargs)
        self.lookback_period = lookback_period
        self.breakout_threshold = breakout_threshold
        self.atr_periods = atr_periods
        
        logger.info(f"Initialized Breakout strategy with lookback_period={lookback_period}, "
                    f"breakout_threshold={breakout_threshold}")
    
    def _calculate_atr(self, high, low, close, period=14):
        """
        Calculate the Average True Range (ATR).
        
        Args:
            high (pandas.Series): High prices
            low (pandas.Series): Low prices
            close (pandas.Series): Close prices
            period (int, optional): ATR period. Defaults to 14.
            
        Returns:
            pandas.Series: ATR values
        """
        tr1 = high - low  # Current high - current low
        tr2 = abs(high - close.shift(1))  # Current high - previous close
        tr3 = abs(low - close.shift(1))  # Current low - previous close
        
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def generate_signals(self, data):
        """
        Generate trading signals based on breakouts.
        
        Args:
            data (pandas.DataFrame): Market data with MultiIndex (symbol, timestamp)
            
        Returns:
            pandas.DataFrame: DataFrame with signals for each symbol and timestamp
        """
        # Get the price data
        if isinstance(data.columns, pd.MultiIndex):
            high_prices = data.xs('high', level=1, axis=1)
            low_prices = data.xs('low', level=1, axis=1)
            close_prices = data.xs('close', level=1, axis=1)
        else:
            high_prices = data['high']
            low_prices = data['low']
            close_prices = data['close']
        
        # Calculate rolling max and min over the lookback period
        resistance_levels = high_prices.rolling(window=self.lookback_period).max().shift(1)
        support_levels = low_prices.rolling(window=self.lookback_period).min().shift(1)
        
        # Calculate ATR for each symbol
        atr_values = pd.DataFrame(index=close_prices.index)
        for symbol in close_prices.columns:
            atr_values[symbol] = self._calculate_atr(
                high_prices[symbol], 
                low_prices[symbol], 
                close_prices[symbol], 
                self.atr_periods
            )
        
        # Initialize signals DataFrame with HOLD
        signals = pd.DataFrame(index=close_prices.index, columns=close_prices.columns, data=SignalType.HOLD)
        
        # Identify breakouts
        for symbol in close_prices.columns:
            # Calculate price breakout as a percentage
            resistance_breakout = (close_prices[symbol] - resistance_levels[symbol]) / resistance_levels[symbol]
            support_breakout = (support_levels[symbol] - close_prices[symbol]) / support_levels[symbol]
            
            # Buy signal when close breaks above resistance by threshold percentage
            signals.loc[resistance_breakout > self.breakout_threshold, symbol] = SignalType.BUY
            
            # Sell signal when close breaks below support by threshold percentage
            signals.loc[support_breakout > self.breakout_threshold, symbol] = SignalType.SELL
        
        # No signal for the first lookback_period+1 days (insufficient data)
        signals.iloc[:self.lookback_period+1] = SignalType.HOLD
        
        # Convert signal values to readable strings for logging
        signal_counts = {
            'BUY': (signals == SignalType.BUY).sum().sum(),
            'SELL': (signals == SignalType.SELL).sum().sum(),
            'HOLD': (signals == SignalType.HOLD).sum().sum(),
        }
        
        logger.info(f"Generated {len(signals)} signals: {signal_counts}")
        
        # Add support/resistance levels and breakout values to the signals DataFrame
        result = pd.DataFrame(index=signals.index)
        
        for symbol in signals.columns:
            result[f'{symbol}_signal'] = signals[symbol]
            result[f'{symbol}_resistance'] = resistance_levels[symbol]
            result[f'{symbol}_support'] = support_levels[symbol]
            result[f'{symbol}_atr'] = atr_values[symbol]
        
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
                atr_col = f'{symbol}_atr'
                
                if signal_col in signals.columns:
                    if signals[signal_col].iloc[-1] == SignalType.BUY:
                        # Optionally scale position size inversely by ATR (volatility)
                        if atr_col in signals.columns and signals[atr_col].iloc[-1] > 0:
                            # Calculate relative ATR (as percentage of price)
                            price_col = f'{symbol}_resistance'  # Use resistance level as reference price
                            if price_col in signals.columns and signals[price_col].iloc[-1] > 0:
                                relative_atr = signals[atr_col].iloc[-1] / signals[price_col].iloc[-1]
                                # Adjust position size inversely to volatility (less for higher volatility)
                                avg_relative_atr = 0.015  # Assumed average (1.5% daily range)
                                scale_factor = avg_relative_atr / max(relative_atr, 0.005)  # Cap scaling
                                position_sizes[symbol] = position_value * min(scale_factor, 1.5)  # Cap at 150%
                            else:
                                position_sizes[symbol] = position_value
                        else:
                            position_sizes[symbol] = position_value
                    elif signals[signal_col].iloc[-1] == SignalType.SELL:
                        position_sizes[symbol] = 0
        
        return position_sizes