"""
Alpaca data connector for retrieving market data.
"""
import logging
import pandas as pd
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
from pathlib import Path
import pytz

from trading_engine.utils.logging import get_logger

logger = get_logger(__name__)


class AlpacaDataConnector:
    """
    Data connector for Alpaca Markets API.
    """
    
    def __init__(self, api_key, api_secret, base_url, data_cache_dir=None):
        """
        Initialize the Alpaca data connector.
        
        Args:
            api_key (str): Alpaca API key
            api_secret (str): Alpaca API secret
            base_url (str): Alpaca API base URL
            data_cache_dir (str, optional): Directory to cache data. Defaults to None.
        """
        self.api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
        self.data_cache_dir = data_cache_dir
        
        if data_cache_dir:
            Path(data_cache_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized Alpaca data connector")

    def get_bars(self, symbols, timeframe, start_date, end_date, limit=None, adjustment='raw'):
        """
        Get historical bars data for one or more symbols.
        
        Args:
            symbols (list): List of symbols to get data for
            timeframe (str): Bar timeframe (e.g., '1D', '1H', '5Min')
            start_date (str or datetime): Start date
            end_date (str or datetime): End date
            limit (int, optional): Maximum number of bars to return. Defaults to None.
            adjustment (str, optional): Adjustment mode ('raw', 'split', 'dividend', 'all'). Defaults to 'raw'.
            
        Returns:
            pandas.DataFrame: DataFrame containing bar data with MultiIndex (symbol, timestamp)
        """
        # Validate inputs
        if isinstance(symbols, str):
            symbols = [symbols]
        
        # Convert string dates to datetime objects
        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date, tz='UTC')
        if isinstance(end_date, str):
            end_date = pd.Timestamp(end_date, tz='UTC')
        
        # Check cache first if data_cache_dir is provided
        if self.data_cache_dir:
            cache_file = self._get_cache_filename(symbols, timeframe, start_date, end_date, adjustment)
            if Path(cache_file).exists():
                logger.info(f"Loading data from cache: {cache_file}")
                return pd.read_parquet(cache_file)
        
        logger.info(f"Fetching {timeframe} bars for {len(symbols)} symbols from {start_date} to {end_date}")
        
        try:
            # Fetch data from Alpaca
            bars = self.api.get_bars(
                symbols,
                timeframe,
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                limit=limit,
                adjustment=adjustment
            ).df
            
            # Cache the data if data_cache_dir is provided
            if self.data_cache_dir and not bars.empty:
                cache_file = self._get_cache_filename(symbols, timeframe, start_date, end_date, adjustment)
                bars.to_parquet(cache_file)
                logger.info(f"Cached data to: {cache_file}")
            
            return bars
            
        except Exception as e:
            logger.error(f"Error fetching data from Alpaca: {e}")
            raise
    
    def get_latest_quotes(self, symbols):
        """
        Get the latest quotes for one or more symbols.
        
        Args:
            symbols (list): List of symbols to get quotes for
            
        Returns:
            pandas.DataFrame: DataFrame containing quote data
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        logger.info(f"Fetching latest quotes for {len(symbols)} symbols")
        
        try:
            quotes = {symbol: self.api.get_latest_quote(symbol) for symbol in symbols}
            
            data = []
            for symbol, quote in quotes.items():
                data.append({
                    'symbol': symbol,
                    'bid_price': quote.bp,
                    'bid_size': quote.bs,
                    'ask_price': quote.ap,
                    'ask_size': quote.as_,
                    'timestamp': pd.Timestamp(quote.t).tz_convert('UTC')
                })
            
            return pd.DataFrame(data).set_index('symbol')
            
        except Exception as e:
            logger.error(f"Error fetching quotes from Alpaca: {e}")
            raise
    
    def get_account(self):
        """
        Get account information.
        
        Returns:
            dict: Account information
        """
        try:
            account = self.api.get_account()
            return {
                'id': account.id,
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'long_market_value': float(account.long_market_value),
                'short_market_value': float(account.short_market_value),
                'initial_margin': float(account.initial_margin),
                'maintenance_margin': float(account.maintenance_margin),
                'last_equity': float(account.last_equity),
                'status': account.status
            }
        except Exception as e:
            logger.error(f"Error fetching account information: {e}")
            raise
    
    def _get_cache_filename(self, symbols, timeframe, start_date, end_date, adjustment):
        """
        Generate a cache filename based on query parameters.
        
        Args:
            symbols (list): List of symbols
            timeframe (str): Bar timeframe
            start_date (datetime): Start date
            end_date (datetime): End date
            adjustment (str): Adjustment mode
            
        Returns:
            str: Cache filename
        """
        symbols_str = '-'.join(sorted(symbols)) if len(symbols) <= 5 else f"{len(symbols)}symbols"
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        
        return f"{self.data_cache_dir}/{symbols_str}_{timeframe}_{start_str}_{end_str}_{adjustment}.parquet"