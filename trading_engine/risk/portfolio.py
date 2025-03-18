"""
Portfolio risk management module.
"""
import numpy as np
import pandas as pd

from trading_engine.utils.logging import get_logger

logger = get_logger(__name__)


class PortfolioRiskManager:
    """
    Risk manager for portfolio position sizing and risk control.
    """
    
    def __init__(self, max_position_size=0.1, max_portfolio_risk=0.02,
                 stop_loss_pct=0.05, take_profit_pct=0.1):
        """
        Initialize the portfolio risk manager.
        
        Args:
            max_position_size (float, optional): Maximum position size as a fraction of portfolio. Defaults to 0.1 (10%).
            max_portfolio_risk (float, optional): Maximum portfolio risk as a fraction. Defaults to 0.02 (2%).
            stop_loss_pct (float, optional): Stop loss percentage. Defaults to 0.05 (5%).
            take_profit_pct (float, optional): Take profit percentage. Defaults to 0.1 (10%).
        """
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        logger.info(f"Initialized PortfolioRiskManager with max_position_size={max_position_size}, "
                   f"max_portfolio_risk={max_portfolio_risk}")
    
    def calculate_position_size(self, symbol, price, portfolio_value, signal_strength=None,
                                current_positions=None, volatility=None):
        """
        Calculate the appropriate position size based on risk parameters.
        
        Args:
            symbol (str): Symbol
            price (float): Current price
            portfolio_value (float): Current portfolio value
            signal_strength (float, optional): Signal strength (e.g., momentum). Defaults to None.
            current_positions (dict, optional): Current positions dictionary. Defaults to None.
            volatility (float, optional): Asset's volatility. Defaults to None.
            
        Returns:
            float: Position size in dollars
        """
        # Initialize current_positions if None
        if current_positions is None:
            current_positions = {}
        
        # 1. Position size limit based on max_position_size
        max_size = portfolio_value * self.max_position_size
        
        # 2. Calculate position size based on volatility and risk tolerance if volatility is provided
        if volatility is not None:
            # Use volatility-based position sizing (e.g., 2% max daily risk)
            risk_based_size = (portfolio_value * self.max_portfolio_risk) / volatility
            max_size = min(max_size, risk_based_size)
        
        # 3. Adjust based on signal strength (if provided)
        if signal_strength is not None:
            # Scale position size by signal strength (assumed to be between 0 and 1)
            signal_factor = min(max(abs(signal_strength), 0), 1)
            max_size *= signal_factor
        
        # 4. Check current portfolio exposure
        current_exposure = sum(current_positions.values())
        portfolio_utilization = current_exposure / portfolio_value if portfolio_value > 0 else 0
        
        # If portfolio is already heavily allocated, reduce position size
        if portfolio_utilization > 0.7:  # 70% of portfolio already allocated
            reduction_factor = 1 - ((portfolio_utilization - 0.7) / 0.3)
            max_size *= max(reduction_factor, 0)
        
        logger.debug(f"Calculated position size for {symbol}: ${max_size:.2f}")
        return max_size
    
    def calculate_stop_loss(self, entry_price, position_type='long'):
        """
        Calculate stop loss price.
        
        Args:
            entry_price (float): Entry price
            position_type (str, optional): Position type ('long' or 'short'). Defaults to 'long'.
            
        Returns:
            float: Stop loss price
        """
        if position_type.lower() == 'long':
            return entry_price * (1 - self.stop_loss_pct)
        else:  # short position
            return entry_price * (1 + self.stop_loss_pct)
    
    def calculate_take_profit(self, entry_price, position_type='long'):
        """
        Calculate take profit price.
        
        Args:
            entry_price (float): Entry price
            position_type (str, optional): Position type ('long' or 'short'). Defaults to 'long'.
            
        Returns:
            float: Take profit price
        """
        if position_type.lower() == 'long':
            return entry_price * (1 + self.take_profit_pct)
        else:  # short position
            return entry_price * (1 - self.take_profit_pct)
    
    def calculate_portfolio_var(self, positions, returns_data, confidence_level=0.95, time_horizon=1):
        """
        Calculate portfolio Value at Risk (VaR).
        
        Args:
            positions (dict): Dictionary of positions {symbol: quantity}
            returns_data (pandas.DataFrame): Historical returns data
            confidence_level (float, optional): Confidence level. Defaults to 0.95.
            time_horizon (int, optional): Time horizon in days. Defaults to 1.
            
        Returns:
            float: Portfolio VaR
        """
        # Filter returns data to include only symbols in positions
        symbols = list(positions.keys())
        filtered_returns = returns_data[symbols].dropna()
        
        if filtered_returns.empty:
            logger.warning("No returns data available for VaR calculation")
            return 0
        
        # Calculate position values
        position_values = pd.Series({symbol: positions[symbol] for symbol in symbols})
        
        # Calculate portfolio returns
        weighted_returns = filtered_returns.mul(position_values, axis=1)
        portfolio_returns = weighted_returns.sum(axis=1)
        
        # Calculate VaR using historical simulation method
        var = -np.percentile(portfolio_returns, 100 * (1 - confidence_level)) * np.sqrt(time_horizon)
        
        logger.debug(f"Portfolio VaR ({confidence_level:.0%}, {time_horizon} day): {var:.2%}")
        return var
    
    def adjust_for_correlation(self, positions, price_data):
        """
        Adjust position sizes based on correlation between assets.
        
        Args:
            positions (dict): Dictionary of proposed positions {symbol: size_in_dollars}
            price_data (pandas.DataFrame): Historical price data
            
        Returns:
            dict: Adjusted positions
        """
        symbols = list(positions.keys())
        
        if len(symbols) <= 1:
            return positions
        
        # Calculate returns
        returns = price_data[symbols].pct_change().dropna()
        
        # Calculate correlation matrix
        corr_matrix = returns.corr()
        
        # Identify highly correlated pairs
        adjusted_positions = positions.copy()
        
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                correlation = corr_matrix.loc[symbol1, symbol2]
                
                # If assets are highly correlated, reduce position sizes
                if correlation > 0.7:
                    reduction_factor = 1 - ((correlation - 0.7) / 0.3) * 0.5
                    adjusted_positions[symbol1] *= reduction_factor
                    adjusted_positions[symbol2] *= reduction_factor
                    
                    logger.debug(f"Reduced positions for correlated pair {symbol1}/{symbol2} "
                                f"(correlation: {correlation:.2f}, factor: {reduction_factor:.2f})")
        
        return adjusted_positions