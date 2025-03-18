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
from trading_engine.strategies.mean_reversion import MeanReversionStrategy
from trading_engine.strategies.ma_crossover import MACrossoverStrategy
from trading_engine.strategies.breakout import BreakoutStrategy
from trading_engine.strategies.combined import CombinedStrategy
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
        choices=["momentum", "mean_reversion", "ma_crossover", "breakout", "combined"],
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
            elif args.strategy == "mean_reversion":
                strategy = MeanReversionStrategy(
                    symbols=symbols,
                    lookback_period=config["strategies"]["mean_reversion"]["lookback_period"],
                    z_score_threshold=config["strategies"]["mean_reversion"]["z_score_threshold"],
                )
            elif args.strategy == "ma_crossover":
                strategy = MACrossoverStrategy(
                    symbols=symbols,
                    fast_period=config["strategies"].get("ma_crossover", {}).get("fast_period", 20),
                    slow_period=config["strategies"].get("ma_crossover", {}).get("slow_period", 50),
                    signal_period=config["strategies"].get("ma_crossover", {}).get("signal_period", 9),
                )
            elif args.strategy == "breakout":
                strategy = BreakoutStrategy(
                    symbols=symbols,
                    lookback_period=config["strategies"].get("breakout", {}).get("lookback_period", 20),
                    breakout_threshold=config["strategies"].get("breakout", {}).get("breakout_threshold", 0.02),
                    atr_periods=config["strategies"].get("breakout", {}).get("atr_periods", 14),
                )
            elif args.strategy == "combined":
                # Create individual strategies to combine
                momentum_strategy = MomentumStrategy(
                    symbols=symbols,
                    lookback_period=config["strategies"]["momentum"]["lookback_period"],
                    threshold=config["strategies"]["momentum"]["threshold"],
                )
                mean_reversion_strategy = MeanReversionStrategy(
                    symbols=symbols,
                    lookback_period=config["strategies"]["mean_reversion"]["lookback_period"],
                    z_score_threshold=config["strategies"]["mean_reversion"]["z_score_threshold"],
                )
                
                strategy = CombinedStrategy(
                    symbols=symbols,
                    strategies=[momentum_strategy, mean_reversion_strategy],
                    aggregation_method=config["strategies"].get("combined", {}).get("aggregation_method", "majority"),
                )
            else:
                raise ValueError(f"Unknown strategy: {args.strategy}")
            
            # Set up risk manager
            risk_manager = PortfolioRiskManager(
                max_position_size=config["risk"]["max_position_size"],
                max_portfolio_risk=config["risk"]["max_portfolio_risk"],
                stop_loss_pct=config["risk"]["stop_loss_percentage"],
                take_profit_pct=config["risk"]["take_profit_percentage"]
            )
            
            # Set up and run backtest
            backtest_engine = BacktestEngine(
                data_connector=data_connector,
                strategy=strategy,
                risk_manager=risk_manager,
                initial_capital=config["backtest"]["initial_capital"],
                start_date=config["backtest"]["start_date"],
                end_date=config["backtest"]["end_date"],
                commission_rate=config["backtest"]["commission_rate"] if config["backtest"]["include_commission"] else 0
            )
            
            results = backtest_engine.run()
            logger.info(f"Backtest completed. Final portfolio value: ${results['final_value']:.2f}")
            
            # Display key performance metrics
            metrics = results['metrics']
            logger.info("Performance Metrics:")
            logger.info(f"Total Return: {metrics['total_return']:.2%}")
            logger.info(f"Annual Return: {metrics['annual_return']:.2%}")
            logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
            logger.info(f"Win Rate: {metrics['win_rate']:.2%}")
            logger.info(f"Number of Trades: {metrics['num_trades']}")
            
        elif args.mode == "paper":
            logger.info("Paper trading mode")
            # Initialize broker connector for paper trading
            broker_connector = AlpacaDataConnector(
                api_key=config["alpaca"]["api_key"],
                api_secret=config["alpaca"]["api_secret"],
                base_url=config["alpaca"]["base_url"],
            )
            
            # TODO: Implement paper trading logic
            logger.info("Paper trading not yet fully implemented")
            
        elif args.mode == "live":
            logger.info("Live trading mode")
            # Initialize broker connector for live trading
            # TODO: Implement live trading logic
            logger.info("Live trading not yet implemented")
            
    except Exception as e:
        logger.exception(f"Error running trading engine: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())