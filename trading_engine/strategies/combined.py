"""
Combined trading strategy implementation that aggregates signals from multiple strategies.
"""
import pandas as pd
import numpy as np

from trading_engine.strategies.base import BaseStrategy, SignalType
from trading_engine.utils.logging import get_logger

logger = get_logger(__name__)


class CombinedStrategy(BaseStrategy):
    """
    Combined trading strategy that aggregates signals from multiple strategies.
    
    This strategy combines signals from multiple underlying strategies and generates
    a consolidated signal based on the specified aggregation method.
    """
    
    def __init__(self, symbols, strategies, aggregation_method='majority', weights=None, **kwargs):
        """
        Initialize the combined strategy.
        
        Args:
            symbols (list): List of symbols to trade
            strategies (list): List of strategy instances to combine
            aggregation_method (str, optional): Method to aggregate signals ('majority', 'unanimous', 'weighted'). 
                                               Defaults to 'majority'.
            weights (dict, optional): Dictionary of strategy weights for weighted aggregation. Defaults to None.
            **kwargs: Additional strategy parameters
        """
        super().__init__(symbols, **kwargs)
        self.strategies = strategies
        self.aggregation_method = aggregation_method
        
        # Validate strategies
        for strategy in self.strategies:
            if not isinstance(strategy, BaseStrategy):
                raise TypeError(f"Expected BaseStrategy instance, got {type(strategy)}")
            
            # Ensure all strategies operate on the same symbols
            if set(strategy.symbols) != set(self.symbols):
                raise ValueError(f"Strategy symbols {strategy.symbols} don't match {self.symbols}")
        
        # Set up weights for weighted aggregation
        if aggregation_method == 'weighted':
            if weights is None:
                # Equal weights if none provided
                self.weights = {i: 1.0 / len(strategies) for i in range(len(strategies))}
            else:
                # Validate weights
                if len(weights) != len(strategies):
                    raise ValueError(f"Expected {len(strategies)} weights, got {len(weights)}")
                
                # Normalize weights to sum to 1
                total = sum(weights.values())
                self.weights = {k: v / total for k, v in weights.items()}
        else:
            self.weights = None
        
        logger.info(f"Initialized Combined strategy with {len(strategies)} strategies, "
                    f"aggregation_method={aggregation_method}")
    
    def generate_signals(self, data):
        """
        Generate trading signals by combining multiple strategies.
        
        Args:
            data (pandas.DataFrame): Market data with MultiIndex (symbol, timestamp)
            
        Returns:
            pandas.DataFrame: DataFrame with signals for each symbol and timestamp
        """
        # Generate signals from each strategy
        strategy_signals = []
        
        for i, strategy in enumerate(self.strategies):
            signals = strategy.generate_signals(data)
            strategy_signals.append(signals)
            logger.debug(f"Generated signals from strategy {i}: {strategy.__class__.__name__}")
        
        # Initialize result DataFrame
        result = pd.DataFrame(index=data.index.get_level_values(0).unique())
        
        # Combine signals for each symbol
        for symbol in self.symbols:
            signal_arrays = []
            
            # Extract signal columns for this symbol from each strategy
            for signals in strategy_signals:
                signal_col = f'{symbol}_signal'
                if signal_col in signals.columns:
                    signal_arrays.append(signals[signal_col])
            
            if not signal_arrays:
                continue
            
            # Combine signals based on aggregation method
            combined_signals = self._aggregate_signals(signal_arrays, symbol)
            result[f'{symbol}_signal'] = combined_signals
            
            # Include additional metrics and data from individual strategies
            for i, signals in enumerate(strategy_signals):
                strategy_name = self.strategies[i].__class__.__name__
                
                # Copy relevant metrics from each strategy
                for col in signals.columns:
                    if col.startswith(f'{symbol}_') and not col.endswith('_signal'):
                        result[f'{symbol}_{strategy_name}_{col.split("_", 1)[1]}'] = signals[col]
        
        # Convert signal values to readable strings for logging
        signal_columns = [col for col in result.columns if col.endswith('_signal')]
        signal_counts = {
            'BUY': sum((result[col] == SignalType.BUY).sum() for col in signal_columns),
            'SELL': sum((result[col] == SignalType.SELL).sum() for col in signal_columns),
            'HOLD': sum((result[col] == SignalType.HOLD).sum() for col in signal_columns),
        }
        
        logger.info(f"Generated {len(result)} combined signals: {signal_counts}")
        
        return result
    
    def _aggregate_signals(self, signal_arrays, symbol):
        """
        Aggregate signals from multiple strategies.
        
        Args:
            signal_arrays (list): List of signal arrays from different strategies
            symbol (str): Symbol for logging
            
        Returns:
            pandas.Series: Aggregated signals
        """
        # Stack signals into a DataFrame
        signals_df = pd.concat(signal_arrays, axis=1)
        
        # Create result Series with the same index
        result = pd.Series(index=signals_df.index, data=SignalType.HOLD)
        
        if self.aggregation_method == 'majority':
            # Count votes for each signal type
            buy_votes = (signals_df == SignalType.BUY).sum(axis=1)
            sell_votes = (signals_df == SignalType.SELL).sum(axis=1)
            hold_votes = (signals_df == SignalType.HOLD).sum(axis=1)
            
            # Find the majority vote
            max_votes = pd.concat([buy_votes, sell_votes, hold_votes], axis=1).max(axis=1)
            
            # Assign signal based on majority vote (with preference for action in case of ties)
            result[buy_votes == max_votes] = SignalType.BUY
            result[sell_votes == max_votes] = SignalType.SELL
            
        elif self.aggregation_method == 'unanimous':
            # Buy signal only if all strategies agree
            result[(signals_df == SignalType.BUY).all(axis=1)] = SignalType.BUY
            
            # Sell signal only if all strategies agree
            result[(signals_df == SignalType.SELL).all(axis=1)] = SignalType.SELL
            
        elif self.aggregation_method == 'weighted':
            # Convert signals to numeric values
            numeric_signals = signals_df.copy()
            
            # Apply weights
            weighted_signals = pd.DataFrame(index=signals_df.index)
            
            for i in range(len(signal_arrays)):
                weighted_signals[i] = numeric_signals.iloc[:, i] * self.weights[i]
            
            # Sum weighted signals
            signal_sum = weighted_signals.sum(axis=1)
            
            # Determine signal based on weighted sum
            result[signal_sum > 0.5] = SignalType.BUY
            result[signal_sum < -0.5] = SignalType.SELL
            
        else:
            logger.warning(f"Unknown aggregation method: {self.aggregation_method}. Using 'majority'.")
            
            # Default to majority voting
            buy_votes = (signals_df == SignalType.BUY).sum(axis=1)
            sell_votes = (signals_df == SignalType.SELL).sum(axis=1)
            hold_votes = (signals_df == SignalType.HOLD).sum(axis=1)
            
            max_votes = pd.concat([buy_votes, sell_votes, hold_votes], axis=1).max(axis=1)
            
            result[buy_votes == max_votes] = SignalType.BUY
            result[sell_votes == max_votes] = SignalType.SELL
        
        logger.debug(f"Aggregated signals for {symbol} using {self.aggregation_method} method")
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
                
                if signal_col in signals.columns:
                    if signals[signal_col].iloc[-1] == SignalType.BUY:
                        position_sizes[symbol] = position_value
                    elif signals[signal_col].iloc[-1] == SignalType.SELL:
                        position_sizes[symbol] = 0
        
        return position_sizes