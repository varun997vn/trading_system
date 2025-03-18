"""
Order management utilities.
"""
import pandas as pd
from datetime import datetime, timedelta

from trading_engine.utils.logging import get_logger
from trading_engine.execution.broker import Order, OrderType, OrderSide, OrderStatus

logger = get_logger(__name__)


class OrderManager:
    """
    Manager for tracking and analyzing orders.
    """
    
    def __init__(self, broker_manager):
        """
        Initialize the order manager.
        
        Args:
            broker_manager: Broker manager instance
        """
        self.broker = broker_manager
        self.orders = []  # List of all orders
        self.active_orders = {}  # Symbol to active orders mapping
        
        logger.info("Initialized OrderManager")
    
    def create_market_order(self, symbol, side, quantity):
        """
        Create a market order.
        
        Args:
            symbol (str): Symbol
            side (str): Order side ('buy' or 'sell')
            quantity (float): Order quantity
            
        Returns:
            Order: Created order
        """
        side = OrderSide(side.lower())
        order = Order(symbol, side, quantity, OrderType.MARKET)
        
        logger.info(f"Created market order: {order}")
        return order
    
    def create_limit_order(self, symbol, side, quantity, limit_price):
        """
        Create a limit order.
        
        Args:
            symbol (str): Symbol
            side (str): Order side ('buy' or 'sell')
            quantity (float): Order quantity
            limit_price (float): Limit price
            
        Returns:
            Order: Created order
        """
        side = OrderSide(side.lower())
        order = Order(symbol, side, quantity, OrderType.LIMIT, limit_price=limit_price)
        
        logger.info(f"Created limit order: {order}")
        return order
    
    def create_stop_order(self, symbol, side, quantity, stop_price):
        """
        Create a stop order.
        
        Args:
            symbol (str): Symbol
            side (str): Order side ('buy' or 'sell')
            quantity (float): Order quantity
            stop_price (float): Stop price
            
        Returns:
            Order: Created order
        """
        side = OrderSide(side.lower())
        order = Order(symbol, side, quantity, OrderType.STOP, stop_price=stop_price)
        
        logger.info(f"Created stop order: {order}")
        return order
    
    def create_stop_limit_order(self, symbol, side, quantity, stop_price, limit_price):
        """
        Create a stop-limit order.
        
        Args:
            symbol (str): Symbol
            side (str): Order side ('buy' or 'sell')
            quantity (float): Order quantity
            stop_price (float): Stop price
            limit_price (float): Limit price
            
        Returns:
            Order: Created order
        """
        side = OrderSide(side.lower())
        order = Order(
            symbol, side, quantity, OrderType.STOP_LIMIT,
            stop_price=stop_price, limit_price=limit_price
        )
        
        logger.info(f"Created stop-limit order: {order}")
        return order
    
    def submit_order(self, order):
        """
        Submit an order.
        
        Args:
            order (Order): Order to submit
            
        Returns:
            Order: Updated order
        """
        order = self.broker.submit_order(order)
        
        # Add to orders list
        self.orders.append(order)
        
        # Add to active orders if not rejected
        if order.status != OrderStatus.REJECTED:
            if order.symbol not in self.active_orders:
                self.active_orders[order.symbol] = []
            self.active_orders[order.symbol].append(order)
        
        return order
    
    def cancel_order(self, order_id):
        """
        Cancel an order.
        
        Args:
            order_id (str): Order ID
            
        Returns:
            bool: Success flag
        """
        success = self.broker.cancel_order(order_id)
        
        if success:
            # Update the order in our lists
            for order in self.orders:
                if order.id == order_id:
                    order.status = OrderStatus.CANCELED
                    break
            
            # Remove from active orders
            for symbol, orders in self.active_orders.items():
                for i, order in enumerate(orders):
                    if order.id == order_id:
                        orders.pop(i)
                        break
        
        return success
    
    def update_orders(self):
        """
        Update the status of all active orders.
        
        Returns:
            list: List of orders that were filled or canceled
        """
        filled_orders = []
        
        # Flatten active orders into a list
        active_orders = [order for orders in self.active_orders.values() for order in orders]
        
        for order in active_orders:
            updated_order = self.broker.get_order(order.id)
            
            if updated_order is None:
                continue
            
            # Check if the order status changed
            if updated_order.status != order.status:
                logger.info(f"Order status changed: {order.id} {order.status} -> {updated_order.status}")
                
                # Update the order in our lists
                for i, o in enumerate(self.orders):
                    if o.id == order.id:
                        self.orders[i] = updated_order
                        break
                
                # If the order is filled or canceled, remove from active orders
                if updated_order.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED]:
                    symbol = updated_order.symbol
                    self.active_orders[symbol] = [o for o in self.active_orders[symbol] if o.id != order.id]
                    
                    if updated_order.status == OrderStatus.FILLED:
                        filled_orders.append(updated_order)
        
        return filled_orders
    
    def get_orders_history(self, symbol=None, start_date=None, end_date=None):
        """
        Get order history.
        
        Args:
            symbol (str, optional): Filter by symbol. Defaults to None.
            start_date (datetime, optional): Start date. Defaults to None.
            end_date (datetime, optional): End date. Defaults to None.
            
        Returns:
            pandas.DataFrame: Order history
        """
        # Filter orders
        filtered_orders = self.orders
        
        if symbol is not None:
            filtered_orders = [order for order in filtered_orders if order.symbol == symbol]
        
        if start_date is not None:
            filtered_orders = [order for order in filtered_orders if order.submitted_at >= start_date]
        
        if end_date is not None:
            filtered_orders = [order for order in filtered_orders if order.submitted_at <= end_date]
        
        # Convert to DataFrame
        data = []
        for order in filtered_orders:
            data.append({
                'id': order.id,
                'symbol': order.symbol,
                'side': order.side.value,
                'quantity': order.quantity,
                'order_type': order.order_type.value,
                'limit_price': order.limit_price,
                'stop_price': order.stop_price,
                'status': order.status.value,
                'submitted_at': order.submitted_at,
                'filled_at': order.filled_at,
                'filled_quantity': order.filled_quantity,
                'filled_price': order.filled_price,
                'commission': order.commission
            })
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df = df.sort_values('submitted_at', ascending=False)
        
        return df
    
    def get_active_orders(self, symbol=None):
        """
        Get active orders.
        
        Args:
            symbol (str, optional): Filter by symbol. Defaults to None.
            
        Returns:
            list: List of active orders
        """
        if symbol is not None:
            return self.active_orders.get(symbol, [])
        else:
            return [order for orders in self.active_orders.values() for order in orders]
    
    def cancel_all_orders(self, symbol=None):
        """
        Cancel all active orders.
        
        Args:
            symbol (str, optional): Filter by symbol. Defaults to None.
            
        Returns:
            int: Number of orders canceled
        """
        orders_to_cancel = self.get_active_orders(symbol)
        canceled_count = 0
        
        for order in orders_to_cancel:
            if self.cancel_order(order.id):
                canceled_count += 1
        
        return canceled_count