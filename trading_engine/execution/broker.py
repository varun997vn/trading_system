"""
Broker implementation for order execution.
"""
import logging
from enum import Enum
from datetime import datetime

from trading_engine.utils.logging import get_logger

logger = get_logger(__name__)


class OrderType(Enum):
    """
    Enumeration of order types.
    """
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """
    Enumeration of order sides.
    """
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """
    Enumeration of order statuses.
    """
    NEW = "new"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    PENDING = "pending"


class Order:
    """
    Class representing a trading order.
    """
    
    def __init__(self, symbol, side, quantity, order_type=OrderType.MARKET,
                 limit_price=None, stop_price=None, time_in_force="day"):
        """
        Initialize an order.
        
        Args:
            symbol (str): Symbol
            side (OrderSide): Order side (BUY or SELL)
            quantity (float): Order quantity
            order_type (OrderType, optional): Order type. Defaults to OrderType.MARKET.
            limit_price (float, optional): Limit price. Defaults to None.
            stop_price (float, optional): Stop price. Defaults to None.
            time_in_force (str, optional): Time in force. Defaults to "day".
        """
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.order_type = order_type
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.time_in_force = time_in_force
        self.status = OrderStatus.NEW
        self.id = None
        self.submitted_at = None
        self.filled_at = None
        self.filled_quantity = 0
        self.filled_price = None
        self.commission = 0
    
    def __str__(self):
        """
        String representation of the order.
        
        Returns:
            str: String representation
        """
        return (f"{self.side.value.upper()} {self.quantity} {self.symbol} @ "
                f"{self.order_type.value.upper()}"
                f"{f' {self.limit_price}' if self.limit_price else ''}"
                f"{f' (stop: {self.stop_price})' if self.stop_price else ''}")


class BrokerManager:
    """
    Manager for broker interactions and order execution.
    """
    
    def __init__(self, broker_connector, commission_rate=0.0005):
        """
        Initialize the broker manager.
        
        Args:
            broker_connector: Broker connector instance
            commission_rate (float, optional): Commission rate. Defaults to 0.0005 (5 basis points).
        """
        self.broker = broker_connector
        self.commission_rate = commission_rate
        self.orders = {}  # Order ID to Order mapping
        
        logger.info(f"Initialized BrokerManager with commission_rate={commission_rate}")
    
    def submit_order(self, order):
        """
        Submit an order to the broker.
        
        Args:
            order (Order): Order to submit
            
        Returns:
            Order: Updated order
        """
        logger.info(f"Submitting order: {order}")
        
        try:
            # Set submission time
            order.submitted_at = datetime.now()
            
            # Submit the order via the broker connector
            response = self.broker.submit_order(
                symbol=order.symbol,
                side=order.side.value,
                qty=order.quantity,
                type=order.order_type.value,
                limit_price=order.limit_price,
                stop_price=order.stop_price,
                time_in_force=order.time_in_force
            )
            
            # Update order with broker response
            order.id = response.id
            order.status = OrderStatus(response.status)
            
            # Store the order
            self.orders[order.id] = order
            
            logger.info(f"Order submitted successfully: {order.id}")
            return order
            
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            order.status = OrderStatus.REJECTED
            return order
    
    def cancel_order(self, order_id):
        """
        Cancel an order.
        
        Args:
            order_id (str): Order ID
            
        Returns:
            bool: Success flag
        """
        logger.info(f"Canceling order: {order_id}")
        
        if order_id not in self.orders:
            logger.warning(f"Order not found: {order_id}")
            return False
        
        try:
            # Cancel the order via the broker connector
            success = self.broker.cancel_order(order_id)
            
            if success:
                self.orders[order_id].status = OrderStatus.CANCELED
                logger.info(f"Order canceled successfully: {order_id}")
            else:
                logger.warning(f"Failed to cancel order: {order_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error canceling order: {e}")
            return False
    
    def get_order(self, order_id):
        """
        Get an order by ID.
        
        Args:
            order_id (str): Order ID
            
        Returns:
            Order: Order or None if not found
        """
        if order_id not in self.orders:
            logger.warning(f"Order not found: {order_id}")
            return None
        
        try:
            # Get the order from the broker connector
            response = self.broker.get_order(order_id)
            
            # Update local order with latest status
            order = self.orders[order_id]
            order.status = OrderStatus(response.status)
            order.filled_quantity = response.filled_qty
            order.filled_price = response.filled_avg_price
            
            if response.status == "filled":
                order.filled_at = datetime.now()
                order.commission = order.filled_quantity * order.filled_price * self.commission_rate
            
            return order
            
        except Exception as e:
            logger.error(f"Error getting order: {e}")
            return self.orders[order_id]
    
    def get_positions(self):
        """
        Get current positions.
        
        Returns:
            dict: Dictionary mapping symbols to positions
        """
        try:
            # Get positions from the broker connector
            response = self.broker.get_positions()
            
            # Convert to dictionary
            positions = {}
            for position in response:
                positions[position.symbol] = {
                    'quantity': float(position.qty),
                    'avg_price': float(position.avg_entry_price),
                    'market_value': float(position.market_value),
                    'unrealized_pl': float(position.unrealized_pl),
                    'current_price': float(position.current_price)
                }
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}
    
    def get_account(self):
        """
        Get account information.
        
        Returns:
            dict: Account information
        """
        try:
            # Get account from the broker connector
            response = self.broker.get_account()
            
            # Convert to dictionary
            account = {
                'cash': float(response.cash),
                'buying_power': float(response.buying_power),
                'equity': float(response.equity)
            }
            
            return account
            
        except Exception as e:
            logger.error(f"Error getting account: {e}")
            return {'cash': 0, 'buying_power': 0, 'equity': 0}