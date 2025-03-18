"""
Backtesting engine implementation.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import time
from tqdm import tqdm

from trading_engine.utils.logging import get_logger
from trading_engine.strategies.base import SignalType

logger = get_logger(__name__)


class BacktestEngine:
    """
    Engine for backtesting trading strategies.
    """
    
    def __init__(self, data_connector, strategy, risk_manager, initial_capital=100000,
                 start_date=None, end_date=None, commission_rate=0.0005, slippage=0.0001):
        """
        Initialize the backtest engine.
        
        Args:
            data_connector: Data connector instance
            strategy: Strategy instance
            risk_manager: Risk manager instance
            initial_capital (float, optional): Initial capital. Defaults to 100000.
            start_date (str, optional): Start date for backtest. Defaults to None.
            end_date (str, optional): End date for backtest. Defaults to None.
            commission_rate (float, optional): Commission rate per trade. Defaults to 0.0005 (5 basis points).
            slippage (float, optional): Slippage per trade. Defaults to 0.0001 (1 basis point).
        """
        self.data_connector = data_connector
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.initial_capital = initial_capital
        self.start_date = pd.Timestamp(start_date) if start_date else None
        self.end_date = pd.Timestamp(end_date) if end_date else None
        self.commission_rate = commission_rate
        self.slippage = slippage
        
        # Backtest state
        self.portfolio = {
            'cash': initial_capital,
            'positions': {},
            'equity': initial_capital,
            'trades': []
        }
        
        logger.info(f"Initialized backtest engine with {initial_capital} initial capital")
    
    def run(self, timeframe='1D'):
        """
        Run the backtest.
        
        Args:
            timeframe (str, optional): Data timeframe. Defaults to '1D'.
        
        Returns:
            dict: Backtest results
        """
        logger.info(f"Starting backtest from {self.start_date} to {self.end_date}")
        start_time = time.time()
        
        # Fetch historical data
        data = self.data_connector.get_bars(
            self.strategy.symbols,
            timeframe,
            self.start_date,
            self.end_date
        )
        
        if data.empty:
            logger.error("No data available for backtesting")
            return {'error': 'No data available'}
        
        # Initialize results DataFrame with daily portfolio values
        dates = data.index.get_level_values(0).unique()
        results = pd.DataFrame(index=dates, columns=['portfolio_value', 'cash', 'positions_value'])
        results['portfolio_value'] = self.initial_capital
        results['cash'] = self.initial_capital
        results['positions_value'] = 0
        
        # Track daily positions for each symbol
        for symbol in self.strategy.symbols:
            results[f'position_{symbol}'] = 0
            results[f'value_{symbol}'] = 0
        
        # Generate trading signals
        signals = self.strategy.generate_signals(data)
        
        # Simulate trading for each day
        logger.info(f"Simulating trading for {len(dates)} days")
        
        for i, date in enumerate(tqdm(dates)):
            # Skip the first day (no previous data to make decisions)
            if i == 0:
                continue
            
            # Get data for the current date
            current_data = data.xs(date, level=0)
            
            # Generate orders based on signals and risk management
            orders = self._generate_orders(signals.loc[date], current_data)
            
            # Execute orders
            for order in orders:
                self._execute_order(order, current_data, date)
            
            # Update portfolio value for the day
            self._update_portfolio_value(current_data, date, results)
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(results)
        
        execution_time = time.time() - start_time
        logger.info(f"Backtest completed in {execution_time:.2f} seconds. Final portfolio value: ${metrics['final_value']:.2f}")
        
        return {
            'results': results,
            'metrics': metrics,
            'trades': pd.DataFrame(self.portfolio['trades']),
            'final_value': metrics['final_value'],
            'execution_time': execution_time
        }
    
    def _generate_orders(self, day_signals, current_data):
        """
        Generate orders based on signals and risk management.
        
        Args:
            day_signals (pandas.Series): Signals for the current day
            current_data (pandas.DataFrame): Market data for the current day
            
        Returns:
            list: List of order dictionaries
        """
        orders = []
        
        for symbol in self.strategy.symbols:
            signal_col = f'{symbol}_signal'
            
            if signal_col in day_signals:
                signal = day_signals[signal_col]
                current_position = self.portfolio['positions'].get(symbol, 0)
                
                if signal == SignalType.BUY and current_position <= 0:
                    # Calculate position size with risk management
                    price = current_data.loc[symbol, 'close']
                    position_size = self.risk_manager.calculate_position_size(
                        symbol=symbol,
                        price=price,
                        portfolio_value=self.portfolio['equity'],
                        signal_strength=day_signals.get(f'{symbol}_momentum', 0),
                        current_positions=self.portfolio['positions']
                    )
                    
                    if position_size > 0:
                        orders.append({
                            'symbol': symbol,
                            'action': 'BUY',
                            'quantity': position_size / price,
                            'price': price
                        })
                
                elif signal == SignalType.SELL and current_position > 0:
                    price = current_data.loc[symbol, 'close']
                    orders.append({
                        'symbol': symbol,
                        'action': 'SELL',
                        'quantity': current_position,
                        'price': price
                    })
        
        return orders
    
    def _execute_order(self, order, current_data, date):
        """
        Execute a trading order.
        
        Args:
            order (dict): Order details
            current_data (pandas.DataFrame): Market data for the current day
            date (pandas.Timestamp): Current date
        """
        symbol = order['symbol']
        action = order['action']
        quantity = order['quantity']
        price = order['price']
        
        # Apply slippage - increase price for buys, decrease for sells
        execution_price = price * (1 + self.slippage if action == 'BUY' else 1 - self.slippage)
        
        # Calculate commission
        commission = execution_price * quantity * self.commission_rate
        
        # Calculate total value
        trade_value = execution_price * quantity
        
        if action == 'BUY':
            # Check if we have enough cash
            total_cost = trade_value + commission
            
            if total_cost > self.portfolio['cash']:
                # Adjust quantity if not enough cash
                adjusted_quantity = (self.portfolio['cash'] - commission) / execution_price
                trade_value = execution_price * adjusted_quantity
                quantity = adjusted_quantity
                total_cost = trade_value + commission
                
                if adjusted_quantity <= 0:
                    logger.warning(f"Not enough cash to execute BUY order for {symbol}")
                    return
            
            # Update portfolio
            self.portfolio['cash'] -= total_cost
            self.portfolio['positions'][symbol] = self.portfolio['positions'].get(symbol, 0) + quantity
            
        elif action == 'SELL':
            # Check if we have enough shares
            current_position = self.portfolio['positions'].get(symbol, 0)
            
            if quantity > current_position:
                quantity = current_position
                trade_value = execution_price * quantity
            
            # Update portfolio
            self.portfolio['cash'] += trade_value - commission
            self.portfolio['positions'][symbol] = current_position - quantity
            
            # Remove position if zero
            if self.portfolio['positions'][symbol] <= 0:
                self.portfolio['positions'].pop(symbol, None)
        
        # Record the trade
        self.portfolio['trades'].append({
            'date': date,
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': execution_price,
            'commission': commission,
            'value': trade_value
        })
        
        logger.debug(f"Executed {action} order for {quantity} shares of {symbol} at ${execution_price:.2f}")
    
    def _update_portfolio_value(self, current_data, date, results):
        """
        Update portfolio value for the current day.
        
        Args:
            current_data (pandas.DataFrame): Market data for the current day
            date (pandas.Timestamp): Current date
            results (pandas.DataFrame): Results DataFrame to update
        """
        positions_value = 0
        
        for symbol, quantity in self.portfolio['positions'].items():
            if symbol in current_data.index:
                price = current_data.loc[symbol, 'close']
                position_value = price * quantity
                positions_value += position_value
                
                # Update symbol-specific columns
                results.loc[date, f'position_{symbol}'] = quantity
                results.loc[date, f'value_{symbol}'] = position_value
            else:
                logger.warning(f"Symbol {symbol} not found in data for {date}")
        
        # Update portfolio value
        self.portfolio['equity'] = self.portfolio['cash'] + positions_value
        
        # Update results DataFrame
        results.loc[date, 'portfolio_value'] = self.portfolio['equity']
        results.loc[date, 'cash'] = self.portfolio['cash']
        results.loc[date, 'positions_value'] = positions_value
    
    def _calculate_metrics(self, results):
        """
        Calculate performance metrics from backtest results.
        
        Args:
            results (pandas.DataFrame): Backtest results
            
        Returns:
            dict: Performance metrics
        """
        # Daily returns
        results['daily_return'] = results['portfolio_value'].pct_change()
        
        # Calculate metrics
        initial_value = self.initial_capital
        final_value = results['portfolio_value'].iloc[-1]
        total_return = (final_value / initial_value) - 1
        
        # Annualized return (assuming 252 trading days)
        days = len(results)
        annual_return = (1 + total_return) ** (252 / days) - 1
        
        # Risk metrics
        daily_returns = results['daily_return'].dropna()
        volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        results['cum_return'] = (1 + results['daily_return'].fillna(0)).cumprod()
        results['running_max'] = results['cum_return'].cummax()
        results['drawdown'] = (results['cum_return'] / results['running_max']) - 1
        max_drawdown = results['drawdown'].min()
        
        # Win rate (based on trades)
        if len(self.portfolio['trades']) > 0:
            trades_df = pd.DataFrame(self.portfolio['trades'])
            buy_trades = trades_df[trades_df['action'] == 'BUY']
            sell_trades = trades_df[trades_df['action'] == 'SELL']
            
            # Match buys and sells to calculate P&L for each round trip
            if not buy_trades.empty and not sell_trades.empty:
                # Group by symbol
                profits = []
                losses = []
                
                for symbol in trades_df['symbol'].unique():
                    symbol_buys = buy_trades[buy_trades['symbol'] == symbol]
                    symbol_sells = sell_trades[sell_trades['symbol'] == symbol]
                    
                    shares_bought = symbol_buys['quantity'].sum()
                    cost_basis = (symbol_buys['quantity'] * symbol_buys['price']).sum()
                    
                    shares_sold = symbol_sells['quantity'].sum()
                    sell_value = (symbol_sells['quantity'] * symbol_sells['price']).sum()
                    
                    if shares_sold > 0:
                        avg_buy_price = cost_basis / shares_bought if shares_bought > 0 else 0
                        avg_sell_price = sell_value / shares_sold
                        
                        if avg_sell_price > avg_buy_price:
                            profits.append(1)
                        else:
                            losses.append(1)
                
                win_rate = len(profits) / (len(profits) + len(losses)) if len(profits) + len(losses) > 0 else 0
            else:
                win_rate = 0
        else:
            win_rate = 0
        
        return {
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(self.portfolio['trades'])
        }