"""
Nifty Options Auto-Trader - Main Entry Point
Automated trading system for Nifty options using Fyers API
"""

import os
import sys
import time
import signal
import yaml
from datetime import datetime, timedelta
from pathlib import Path
import threading
from typing import Optional

import click
import schedule
from loguru import logger
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.strategy import TradingStrategy, SignalType, OptionType
from core.indicators import TechnicalIndicators
from core.logger import TradingLogger, AlertManager
from api.fyers_client import FyersClient
from backtest.engine import BacktestEngine


class NiftyTrader:
    """
    Main trading bot class.

    Orchestrates:
    - Market data fetching
    - Strategy signal generation
    - Order execution
    - Position management
    - Logging and alerts
    """

    def __init__(self, config_path: str = None):
        """
        Initialize the trading bot.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize components
        self.logger = TradingLogger(self.config)
        self.alerts = AlertManager(self.config)
        self.strategy = TradingStrategy(self.config)
        self.fyers = FyersClient(self.config)

        # Trading state
        self.is_running = False
        self.is_paper_mode = self.config.get('system', {}).get('mode', 'paper') == 'paper'
        self.shutdown_event = threading.Event()

        # Data cache
        self.daily_data: Optional[pd.DataFrame] = None
        self.intraday_data: Optional[pd.DataFrame] = None
        self.last_daily_update: Optional[datetime] = None

        logger.info(f"NiftyTrader initialized in {'PAPER' if self.is_paper_mode else 'LIVE'} mode")

    def _load_config(self, config_path: str = None) -> dict:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent / 'config' / 'settings.yaml'

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def authenticate(self) -> bool:
        """Authenticate with Fyers."""
        if self.is_paper_mode:
            logger.info("Paper mode - skipping authentication")
            return True

        if not self.fyers.is_authenticated():
            logger.info("Authenticating with Fyers...")
            if not self.fyers.authenticate():
                logger.error("Authentication failed")
                self.alerts.error_alert("Fyers authentication failed")
                return False

        # Verify authentication
        profile = self.fyers.get_profile()
        if profile:
            logger.info(f"Authenticated as: {profile.get('name', 'Unknown')}")
            return True

        return False

    def fetch_daily_data(self) -> bool:
        """Fetch daily historical data for analysis."""
        try:
            symbol = self.config.get('instrument', {}).get('symbol', 'NSE:NIFTY50-INDEX')
            end_date = datetime.now()
            start_date = end_date - timedelta(days=100)  # 100 days for EMA calculation

            if self.is_paper_mode:
                # Use yfinance for paper mode
                import yfinance as yf
                ticker = yf.Ticker("^NSEI")
                df = ticker.history(start=start_date, end=end_date)
                df.columns = df.columns.str.lower()
            else:
                # Use Fyers API
                candles = self.fyers.get_historical_data(
                    symbol=symbol,
                    resolution="D",
                    from_date=start_date,
                    to_date=end_date
                )

                if not candles:
                    logger.error("Failed to fetch daily data")
                    return False

                # Convert to DataFrame
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)

            # Add indicators
            indicators = TechnicalIndicators(self.config.get('daily_conditions', {}))
            self.daily_data = indicators.add_all_indicators(df)
            self.last_daily_update = datetime.now()

            logger.info(f"Daily data updated - {len(self.daily_data)} candles")
            return True

        except Exception as e:
            logger.error(f"Error fetching daily data: {e}")
            return False

    def fetch_intraday_data(self) -> bool:
        """Fetch 15-min intraday data."""
        try:
            symbol = self.config.get('instrument', {}).get('symbol', 'NSE:NIFTY50-INDEX')
            end_date = datetime.now()
            start_date = end_date - timedelta(days=2)  # 2 days for first candle

            if self.is_paper_mode:
                # Use yfinance for paper mode
                import yfinance as yf
                ticker = yf.Ticker("^NSEI")
                df = ticker.history(start=start_date, end=end_date, interval="15m")
                df.columns = df.columns.str.lower()
            else:
                candles = self.fyers.get_historical_data(
                    symbol=symbol,
                    resolution="15",
                    from_date=start_date,
                    to_date=end_date
                )

                if not candles:
                    logger.error("Failed to fetch intraday data")
                    return False

                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)

            self.intraday_data = df
            logger.debug(f"Intraday data updated - {len(self.intraday_data)} candles")
            return True

        except Exception as e:
            logger.error(f"Error fetching intraday data: {e}")
            return False

    def analyze_market(self) -> SignalType:
        """Analyze market conditions and generate signal."""
        if self.daily_data is None or self.daily_data.empty:
            if not self.fetch_daily_data():
                return SignalType.NO_DATA

        # Update daily analysis
        signal = self.strategy.update_daily_analysis(self.daily_data)

        logger.info(f"Market analysis: {signal.value}")
        return signal

    def check_entry_conditions(self) -> bool:
        """Check if entry conditions are met."""
        # Check if we can take a new trade
        can_trade, reason = self.strategy.can_take_new_trade()
        if not can_trade:
            logger.debug(f"Cannot take new trade: {reason}")
            return False

        # Check trading time
        current_time = datetime.now().time()
        is_valid_time, time_reason = self.strategy.is_trading_time(current_time)
        if not is_valid_time:
            logger.debug(f"Not trading time: {time_reason}")
            return False

        # Fetch fresh intraday data
        if not self.fetch_intraday_data():
            return False

        if self.intraday_data.empty:
            return False

        latest = self.intraday_data.iloc[-1]

        # Get daily signals for prev high/low
        if self.daily_data is None or self.daily_data.empty:
            return False

        daily_latest = self.daily_data.iloc[-1]

        # Check entry trigger
        triggered, option_type, trigger_reason = self.strategy.check_entry_trigger(
            current_price=latest['close'],
            current_volume=latest['volume'],
            prev_day_high=daily_latest['prev_high'],
            prev_day_low=daily_latest['prev_low'],
            avg_volume=daily_latest['volume_ma']
        )

        if triggered:
            logger.info(f"Entry trigger: {trigger_reason} for {option_type.value}")
            return self.execute_entry(option_type, latest['close'], trigger_reason)

        return False

    def execute_entry(self, option_type: OptionType, nifty_price: float,
                     trigger_reason: str) -> bool:
        """Execute trade entry."""
        try:
            # Calculate strike
            strike = self.strategy.calculate_strike(nifty_price, option_type)

            # Get current expiry
            expiry = self.fyers.get_current_expiry() if not self.is_paper_mode else "250213"

            # Build option symbol
            underlying = "NIFTY"
            option_symbol = self.fyers.build_option_symbol(
                underlying, expiry, strike, option_type.value
            ) if not self.is_paper_mode else f"NSE:NIFTY{expiry}{int(strike)}{option_type.value}"

            # Get option premium (simulate for paper mode)
            if self.is_paper_mode:
                # Simulate premium
                if option_type == OptionType.CALL:
                    premium = 100 + (nifty_price - strike) * 0.5 if nifty_price > strike else 100 - (strike - nifty_price) * 0.3
                else:
                    premium = 100 + (strike - nifty_price) * 0.5 if strike > nifty_price else 100 - (nifty_price - strike) * 0.3
                premium = max(80, min(150, premium))
            else:
                quotes = self.fyers.get_quotes([option_symbol])
                if not quotes:
                    logger.error("Failed to get option quote")
                    return False
                premium = quotes[0]['v']['lp']  # Last price

            # Check premium range
            min_premium = self.config.get('capital', {}).get('min_premium', 80)
            max_premium = self.config.get('capital', {}).get('max_premium', 150)

            if not (min_premium <= premium <= max_premium):
                logger.warning(f"Premium ₹{premium} outside range [{min_premium}, {max_premium}]")
                return False

            # Calculate position size
            capital = self.config.get('capital', {}).get('total_capital', 200000)
            risk_percent = self.config.get('capital', {}).get('risk_per_trade_percent', 2)
            lot_size = self.config.get('instrument', {}).get('lot_size', 25)

            lots, quantity = self.strategy.calculate_position_size(
                premium, capital, risk_percent, lot_size
            )

            # Create trade in strategy
            trade = self.strategy.create_trade(
                option_type=option_type,
                strike=strike,
                entry_price=premium,
                lots=lots,
                quantity=quantity,
                trigger_type=trigger_reason
            )

            # Place order (skip for paper mode)
            if not self.is_paper_mode:
                order_response = self.fyers.place_order(
                    symbol=option_symbol,
                    qty=quantity,
                    side=1,  # Buy
                    order_type=2,  # Market
                    product_type="INTRADAY"
                )

                if not order_response:
                    logger.error("Order placement failed")
                    return False

                logger.info(f"Order placed: {order_response}")

            # Log and alert
            self.logger.log_trade_entry({
                'trade_id': trade.trade_id,
                'option_type': option_type.value,
                'strike': strike,
                'entry_price': premium,
                'quantity': quantity,
                'lots': lots,
                'stop_loss': trade.stop_loss,
                'target': trade.target
            })

            self.alerts.trade_entry_alert({
                'option_type': option_type.value,
                'strike': strike,
                'entry_price': premium,
                'stop_loss': trade.stop_loss,
                'target': trade.target
            })

            return True

        except Exception as e:
            logger.error(f"Entry execution error: {e}")
            return False

    def check_exit_conditions(self) -> bool:
        """Check if exit conditions are met for open position."""
        position = self.strategy.current_position
        if position is None:
            return False

        try:
            # Get current premium
            if self.is_paper_mode:
                # Simulate based on latest price
                if not self.fetch_intraday_data():
                    return False
                latest = self.intraday_data.iloc[-1]
                current_nifty = latest['close']

                # Simple simulation
                entry_nifty = position.entry_price * 2  # Rough estimate
                movement = (current_nifty - entry_nifty) / entry_nifty * 100

                if position.option_type == OptionType.PUT:
                    movement = -movement

                current_premium = position.entry_price * (1 + movement / 100)
                current_premium = max(1, current_premium)
            else:
                expiry = self.fyers.get_current_expiry()
                option_symbol = self.fyers.build_option_symbol(
                    "NIFTY", expiry, position.strike, position.option_type.value
                )
                quotes = self.fyers.get_quotes([option_symbol])
                if not quotes:
                    return False
                current_premium = quotes[0]['v']['lp']

            # Update position with current price
            position.current_price = current_premium

            # Update trailing SL
            self.strategy.update_trailing_sl(current_premium)

            # Check exit conditions
            current_time = datetime.now().time()
            should_exit, exit_reason, exit_percent = self.strategy.check_exit_conditions(
                current_premium, current_time
            )

            if should_exit:
                return self.execute_exit(current_premium, exit_reason, exit_percent)

            return False

        except Exception as e:
            logger.error(f"Exit check error: {e}")
            return False

    def execute_exit(self, exit_price: float, exit_reason: str,
                    exit_percent: float) -> bool:
        """Execute trade exit."""
        try:
            position = self.strategy.current_position
            if position is None:
                return False

            # Execute exit in strategy
            exit_result = self.strategy.execute_exit(exit_price, exit_reason, exit_percent)

            if not exit_result.get('success'):
                return False

            # Place exit order (skip for paper mode)
            if not self.is_paper_mode and exit_result['type'] == 'FULL':
                expiry = self.fyers.get_current_expiry()
                option_symbol = self.fyers.build_option_symbol(
                    "NIFTY", expiry, position.strike, position.option_type.value
                )

                order_response = self.fyers.place_order(
                    symbol=option_symbol,
                    qty=exit_result['quantity'],
                    side=-1,  # Sell
                    order_type=2,  # Market
                    product_type="INTRADAY"
                )

                if not order_response:
                    logger.error("Exit order failed")

            # Log and alert
            self.logger.log_trade_exit({
                'trade_id': position.trade_id,
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'pnl': exit_result['pnl'],
                'type': exit_result['type']
            })

            self.alerts.trade_exit_alert({
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'pnl': exit_result['pnl']
            })

            return True

        except Exception as e:
            logger.error(f"Exit execution error: {e}")
            return False

    def trading_loop(self):
        """Main trading loop - called every minute."""
        try:
            now = datetime.now()

            # Check if trading day
            if not self.strategy.is_trading_day(now):
                return

            # Reset daily state
            self.strategy.reset_daily_state(now)

            # Update daily data if needed (once per day before market open)
            if (self.last_daily_update is None or
                self.last_daily_update.date() != now.date()):
                self.fetch_daily_data()

            # Analyze market
            signal = self.analyze_market()

            if signal == SignalType.NEUTRAL or signal == SignalType.NO_DATA:
                return

            # Check for entries or exits
            if self.strategy.current_position is None:
                self.check_entry_conditions()
            else:
                self.check_exit_conditions()

        except Exception as e:
            logger.error(f"Trading loop error: {e}")

    def graceful_shutdown(self):
        """Handle graceful shutdown."""
        logger.info("Initiating graceful shutdown...")

        # Exit all positions
        if self.strategy.current_position is not None:
            logger.warning("Closing open position due to shutdown")

            if not self.is_paper_mode:
                self.fyers.exit_all_positions()

            # Log the forced exit
            position = self.strategy.current_position
            self.strategy.execute_exit(
                position.current_price or position.entry_price,
                "SHUTDOWN",
                100
            )

        # Save daily summary
        summary = self.strategy.get_daily_summary()
        self.logger.save_daily_summary(summary)

        self.is_running = False
        self.shutdown_event.set()

        logger.info("Shutdown complete")

    def run(self):
        """Start the trading bot."""
        logger.info("Starting NiftyTrader...")

        # Set up signal handlers
        signal.signal(signal.SIGINT, lambda s, f: self.graceful_shutdown())
        signal.signal(signal.SIGTERM, lambda s, f: self.graceful_shutdown())

        # Authenticate
        if not self.authenticate():
            logger.error("Failed to authenticate. Exiting.")
            return

        self.is_running = True

        # Schedule trading loop
        schedule.every(30).seconds.do(self.trading_loop)

        logger.info("Trading bot started. Press Ctrl+C to stop.")

        # Run initial analysis
        self.trading_loop()

        # Main loop
        while self.is_running:
            schedule.run_pending()
            time.sleep(1)

        logger.info("Trading bot stopped.")


@click.group()
def cli():
    """Nifty Options Auto-Trader CLI"""
    pass


@cli.command()
@click.option('--config', '-c', default=None, help='Path to config file')
@click.option('--paper', is_flag=True, help='Run in paper trading mode')
def run(config, paper):
    """Start the trading bot."""
    if paper:
        print("Starting in PAPER TRADING mode...")

    trader = NiftyTrader(config_path=config)

    if paper:
        trader.is_paper_mode = True

    trader.run()


@cli.command()
@click.option('--config', '-c', default=None, help='Path to config file')
@click.option('--output', '-o', default='backtest_results.csv', help='Output file for results')
def backtest(config, output):
    """Run backtest on historical data."""
    print("Running backtest...")

    if config:
        with open(config, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        config_path = Path(__file__).parent / 'config' / 'settings.yaml'
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

    engine = BacktestEngine(cfg)
    result = engine.run_backtest()

    engine.print_results(result)

    if output:
        engine.export_results(result, output)
        print(f"\nResults exported to: {output}")


@cli.command()
@click.option('--config', '-c', default=None, help='Path to config file')
def auth(config):
    """Authenticate with Fyers."""
    print("Authenticating with Fyers...")

    if config:
        with open(config, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        config_path = Path(__file__).parent / 'config' / 'settings.yaml'
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

    client = FyersClient(cfg)

    if client.authenticate():
        print("Authentication successful!")
        profile = client.get_profile()
        if profile:
            print(f"Logged in as: {profile.get('name', 'Unknown')}")
            print(f"Client ID: {profile.get('fy_id', 'Unknown')}")
    else:
        print("Authentication failed!")


@cli.command()
def status():
    """Show current status and open positions."""
    config_path = Path(__file__).parent / 'config' / 'settings.yaml'
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    client = FyersClient(cfg)

    if not client.is_authenticated():
        print("Not authenticated. Run 'auth' command first.")
        return

    print("\n=== FYERS STATUS ===")

    # Profile
    profile = client.get_profile()
    if profile:
        print(f"Account: {profile.get('name', 'Unknown')}")

    # Funds
    funds = client.get_funds()
    if funds:
        for fund in funds:
            if fund.get('id') == 10:  # Available margin
                print(f"Available Margin: ₹{fund.get('equityAmount', 0):,.2f}")

    # Positions
    positions = client.get_positions()
    if positions:
        print(f"\nOpen Positions: {len(positions)}")
        for pos in positions:
            print(f"  {pos.get('symbol')}: {pos.get('netQty')} @ ₹{pos.get('avgPrice', 0):.2f}")
    else:
        print("\nNo open positions")

    # Today's orders
    orders = client.get_orders()
    if orders:
        print(f"\nToday's Orders: {len(orders)}")


if __name__ == '__main__':
    cli()
