"""
Trading strategies module.

This module contains various trading strategy implementations that can be used
with the trading engine for backtesting and live trading.

Available strategies:
- BaseStrategy: Abstract base class for all strategies
- MomentumStrategy: Strategy based on price momentum
- MeanReversionStrategy: Strategy based on mean reversion principles
- MACrossoverStrategy: Strategy based on moving average crossovers
- BreakoutStrategy: Strategy based on support/resistance breakouts
- CombinedStrategy: Strategy that combines signals from multiple strategies
"""

from trading_engine.strategies.base import BaseStrategy, SignalType
from trading_engine.strategies.momentum import MomentumStrategy
from trading_engine.strategies.mean_reversion import MeanReversionStrategy
from trading_engine.strategies.ma_crossover import MACrossoverStrategy
from trading_engine.strategies.breakout import BreakoutStrategy
from trading_engine.strategies.combined import CombinedStrategy

__all__ = [
    'BaseStrategy',
    'SignalType',
    'MomentumStrategy',
    'MeanReversionStrategy',
    'MACrossoverStrategy',
    'BreakoutStrategy',
    'CombinedStrategy',
]