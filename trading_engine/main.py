#!/usr/bin/env python3
"""
Trading Analysis Engine - Main Entry Point
"""
import argparse
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from trading_engine.utils.config import load_config
from trading_engine.utils.logging import setup_logging
from trading_engine.data.connectors.alpaca import AlpacaDataConnector
from trading_engine.strategies.momentum import MomentumStrategy
from trading_engine.backtesting.engine import BacktestEngine
from trading_engine.risk.portfolio import PortfolioRiskManager
from trading_engine.execution.broker import BrokerManager


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Trading Analysis Engine")
    parser.add_argument(
        "--mode",
        choices=["backtest", "paper", "live"],
        default="backtest",
        help="Execution mode (default: backtest)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="momentum",
        help="Strategy to execute",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="AAPL,MSFT,AMZN,GOOGL",
        help="Comma-separated list of symbols to trade",
    )
    return parser.parse_args()


def main():
    """Main function to run the trading engine."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up logging
    setup_logging(config.get("logging", {}))
    logger = logging.getLogger(__name__)
    logger.info(f"Starting trading engine in {args.mode} mode")
    
    # Parse symbols
    symbols = args.symbols.split(",")
    logger.info(f"Trading symbols: {symbols}")
    
    try:
        # Initialize components based on mode
        if args.mode == "backtest":
            # Set up data connector
            data_connector = AlpacaDataConnector(
                api_key=config["alpaca"]["api_key"],
                api_secret=config["alpaca"]["api_secret"],
                base_url=config["alpaca"]["base_url"],
            )
            
            # Set up strategy
            if args.strategy == "momentum":
                strategy = MomentumStrategy(
                    symbols=symbols,
                    lookback_period=config["strategies"]["momentum"]["lookback_period"],
                    threshold=config["strategies"]["momentum"]["threshold"],
                )
            else:
                raise ValueError(f"Unknown strategy: {args.strategy}")
            
            # Set up risk manager
            risk_manager = PortfolioRiskManager(
                max_position_size=config["risk"]["max_position_size"],
                max_portfolio_risk=config["risk"]["max_portfolio_risk"],
            )
            
            # Set up and run backtest
            backtest_engine = BacktestEngine(
                data_connector=data_connector,
                strategy=strategy,
                risk_manager=risk_manager,
                initial_capital=config["backtest"]["initial_capital"],
                start_date=config["backtest"]["start_date"],
                end_date=config["backtest"]["end_date"],
            )
            
            results = backtest_engine.run()
            logger.info(f"Backtest completed. Final portfolio value: ${results['final_value']:.2f}")
            
        elif args.mode in ["paper", "live"]:
            logger.info(f"{args.mode.capitalize()} trading not yet implemented")
            # TODO: Implement paper and live trading
            
    except Exception as e:
        logger.exception(f"Error running trading engine: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())