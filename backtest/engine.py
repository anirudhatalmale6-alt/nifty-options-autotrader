"""
Backtesting Engine
Simulates the trading strategy on historical data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger
import yfinance as yf

from core.indicators import TechnicalIndicators
from core.strategy import (
    TradingStrategy, SignalType, OptionType,
    TradeStatus, TradePosition
)


@dataclass
class BacktestTrade:
    """Record of a backtested trade."""
    trade_id: int
    date: datetime
    option_type: str
    strike: float
    entry_price: float
    entry_time: str
    exit_price: float
    exit_time: str
    exit_reason: str
    quantity: int
    lots: int
    gross_pnl: float
    commission: float
    slippage: float
    net_pnl: float
    return_percent: float
    holding_time: str


@dataclass
class BacktestResult:
    """Complete backtesting results."""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_profit: float
    total_loss: float
    net_pnl: float
    roi_percent: float
    average_win: float
    average_loss: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_percent: float
    sharpe_ratio: float
    avg_holding_time: str
    best_trade: float
    worst_trade: float
    consecutive_wins_max: int
    consecutive_losses_max: int
    monthly_returns: Dict[str, float] = field(default_factory=dict)
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


class BacktestEngine:
    """
    Backtesting engine for Nifty options strategy.

    Simulates option premium movement based on Nifty price movement
    with adjustable slippage and commission.
    """

    def __init__(self, config: dict):
        """
        Initialize backtesting engine.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        backtest_config = config.get('backtest', {})

        self.start_date = datetime.strptime(
            backtest_config.get('start_date', '2022-01-01'), '%Y-%m-%d'
        )
        self.end_date = datetime.strptime(
            backtest_config.get('end_date', '2025-02-07'), '%Y-%m-%d'
        )
        self.initial_capital = backtest_config.get('initial_capital', 200000)

        self.include_slippage = backtest_config.get('include_slippage', True)
        self.slippage_percent = backtest_config.get('slippage_percent', 0.1)
        self.include_commission = backtest_config.get('include_commission', True)
        self.commission_per_lot = backtest_config.get('commission_per_lot', 20)

        # Strategy instance
        self.strategy = TradingStrategy(config)
        self.indicators = TechnicalIndicators(config.get('daily_conditions', {}))

        # Backtest state
        self.current_capital = self.initial_capital
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[float] = []
        self.trade_id = 0

    def download_data(self, symbol: str = "^NSEI") -> pd.DataFrame:
        """
        Download historical data from Yahoo Finance.

        Args:
            symbol: Yahoo Finance symbol (^NSEI for Nifty 50)

        Returns:
            OHLCV DataFrame
        """
        logger.info(f"Downloading data for {symbol} from {self.start_date} to {self.end_date}")

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=self.start_date, end=self.end_date)

            if df.empty:
                logger.error("No data downloaded")
                return pd.DataFrame()

            # Standardize column names
            df.columns = df.columns.str.lower()
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df.index = pd.to_datetime(df.index)

            logger.info(f"Downloaded {len(df)} days of data")

            return df

        except Exception as e:
            logger.error(f"Data download error: {e}")
            return pd.DataFrame()

    def simulate_option_premium(self, nifty_price: float, strike: float,
                                option_type: OptionType,
                                movement_percent: float) -> Tuple[float, float]:
        """
        Simulate option premium based on Nifty movement.

        Uses a simplified delta-based model for premium estimation.

        Args:
            nifty_price: Current Nifty price
            strike: Option strike price
            option_type: CALL or PUT
            movement_percent: Nifty % movement from entry

        Returns:
            Tuple of (entry_premium, current_premium)
        """
        # Calculate moneyness
        if option_type == OptionType.CALL:
            moneyness = nifty_price - strike
        else:
            moneyness = strike - nifty_price

        # Estimate entry premium (ATM ~ 100-150 range)
        # OTM premium is lower, ITM is higher
        if moneyness >= 0:  # ITM
            entry_premium = 100 + abs(moneyness) * 0.6  # Delta ~ 0.6
        else:  # OTM
            entry_premium = max(50, 100 - abs(moneyness) * 0.4)  # Delta ~ 0.4

        # Clamp to realistic range
        entry_premium = min(max(entry_premium, 80), 150)

        # Simulate premium change based on Nifty movement
        # Use approximate delta (0.5 for ATM)
        delta = 0.5 if abs(moneyness) < 50 else (0.6 if moneyness > 0 else 0.35)

        nifty_points_moved = nifty_price * (movement_percent / 100)

        if option_type == OptionType.CALL:
            premium_change = nifty_points_moved * delta
        else:
            premium_change = -nifty_points_moved * delta

        current_premium = entry_premium + premium_change

        # Add some noise (theta decay, IV changes)
        theta_decay = entry_premium * 0.02  # ~2% daily theta
        current_premium = max(1, current_premium - theta_decay)

        return round(entry_premium, 2), round(current_premium, 2)

    def apply_slippage(self, price: float, is_entry: bool) -> float:
        """
        Apply slippage to price.

        Args:
            price: Original price
            is_entry: True for entry, False for exit

        Returns:
            Price with slippage applied
        """
        if not self.include_slippage:
            return price

        slippage = price * (self.slippage_percent / 100)

        if is_entry:
            return price + slippage  # Pay more on entry
        else:
            return price - slippage  # Get less on exit

    def calculate_commission(self, lots: int) -> float:
        """Calculate commission for trade."""
        if not self.include_commission:
            return 0

        return lots * self.commission_per_lot

    def run_backtest(self) -> BacktestResult:
        """
        Run the backtest on historical data.

        Returns:
            BacktestResult with all metrics
        """
        logger.info("Starting backtest...")

        # Download data
        df = self.download_data()
        if df.empty:
            logger.error("No data available for backtest")
            return self._empty_result()

        # Add indicators
        df = self.indicators.add_all_indicators(df)

        # Skip initial period for indicator warmup
        df = df.iloc[60:]  # Skip first 60 days

        # Reset state
        self.current_capital = self.initial_capital
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.trade_id = 0

        # Strategy state
        consecutive_losses = 0
        daily_trades = 0
        current_date = None
        position = None  # Simulated position

        # Iterate through each day
        for i, (date, row) in enumerate(df.iterrows()):
            # Reset daily counters
            if current_date != date.date():
                current_date = date.date()
                daily_trades = 0

            # Check if trading day
            if not self._is_valid_trading_day(date):
                continue

            # Check trading limits
            max_trades = self.config.get('limits', {}).get('max_trades_per_day', 2)
            max_losses = self.config.get('limits', {}).get('max_consecutive_losses', 3)

            if daily_trades >= max_trades or consecutive_losses >= max_losses:
                continue

            # Get signal from indicators
            signal = self._get_signal(row)

            if position is None and signal != SignalType.NEUTRAL:
                # Check for entry
                entry_triggered, trigger_type = self._check_entry_trigger(row, df.iloc[:i+1], signal)

                if entry_triggered:
                    # Calculate position details
                    nifty_price = row['close']
                    option_type = OptionType.CALL if signal == SignalType.BULLISH else OptionType.PUT
                    strike = self._calculate_strike(nifty_price, option_type)

                    # Simulate entry premium
                    entry_premium, _ = self.simulate_option_premium(
                        nifty_price, strike, option_type, 0
                    )
                    entry_premium = self.apply_slippage(entry_premium, is_entry=True)

                    # Calculate position size
                    lots, qty = self._calculate_position_size(entry_premium)

                    # Create position
                    position = {
                        'trade_id': self.trade_id,
                        'date': date,
                        'option_type': option_type,
                        'strike': strike,
                        'entry_price': entry_premium,
                        'entry_nifty': nifty_price,
                        'quantity': qty,
                        'lots': lots,
                        'stop_loss_pct': 33,
                        'target_pct': 100
                    }

                    self.trade_id += 1
                    daily_trades += 1

            elif position is not None:
                # Check for exit
                current_nifty = row['close']
                movement_pct = ((current_nifty - position['entry_nifty']) /
                               position['entry_nifty']) * 100

                if position['option_type'] == OptionType.PUT:
                    movement_pct = -movement_pct

                _, current_premium = self.simulate_option_premium(
                    position['entry_nifty'], position['strike'],
                    position['option_type'], movement_pct
                )

                # Check exit conditions
                entry_price = position['entry_price']
                sl_price = entry_price * (1 - position['stop_loss_pct'] / 100)
                target_price = entry_price * (1 + position['target_pct'] / 100)

                exit_reason = None
                exit_price = current_premium

                if current_premium <= sl_price:
                    exit_reason = "STOP_LOSS"
                    exit_price = sl_price
                elif current_premium >= target_price:
                    exit_reason = "TARGET"
                    exit_price = target_price

                if exit_reason:
                    # Execute exit
                    exit_price = self.apply_slippage(exit_price, is_entry=False)
                    commission = self.calculate_commission(position['lots']) * 2  # Entry + Exit

                    gross_pnl = (exit_price - entry_price) * position['quantity']
                    slippage_cost = entry_price * (self.slippage_percent / 100) * position['quantity'] * 2
                    net_pnl = gross_pnl - commission

                    # Record trade
                    trade = BacktestTrade(
                        trade_id=position['trade_id'],
                        date=position['date'],
                        option_type=position['option_type'].value,
                        strike=position['strike'],
                        entry_price=entry_price,
                        entry_time="10:00",
                        exit_price=exit_price,
                        exit_time="14:00",
                        exit_reason=exit_reason,
                        quantity=position['quantity'],
                        lots=position['lots'],
                        gross_pnl=gross_pnl,
                        commission=commission,
                        slippage=slippage_cost,
                        net_pnl=net_pnl,
                        return_percent=(net_pnl / (entry_price * position['quantity'])) * 100,
                        holding_time="4h"
                    )
                    self.trades.append(trade)

                    # Update capital
                    self.current_capital += net_pnl
                    self.equity_curve.append(self.current_capital)

                    # Update consecutive losses
                    if net_pnl < 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0

                    # Clear position
                    position = None

        # Generate results
        return self._generate_results()

    def _is_valid_trading_day(self, date: datetime) -> bool:
        """Check if date is a valid trading day."""
        day_name = date.strftime('%A').lower()
        trading_days = self.config.get('trading_days', {})
        return trading_days.get(day_name, False)

    def _get_signal(self, row: pd.Series) -> SignalType:
        """Get trading signal from indicators."""
        daily_config = self.config.get('daily_conditions', {})

        rsi_bullish_min = daily_config.get('rsi_bullish_min', 50)
        rsi_bullish_max = daily_config.get('rsi_bullish_max', 70)
        rsi_bearish_min = daily_config.get('rsi_bearish_min', 30)
        rsi_bearish_max = daily_config.get('rsi_bearish_max', 50)

        bullish = (
            row['above_ema_20'] and
            row['above_ema_50'] and
            row['ema_20_above_50'] and
            rsi_bullish_min <= row['rsi'] <= rsi_bullish_max and
            row['macd_bullish'] and
            row['volume_above_avg']
        )

        bearish = (
            not row['above_ema_20'] and
            not row['above_ema_50'] and
            not row['ema_20_above_50'] and
            rsi_bearish_min <= row['rsi'] <= rsi_bearish_max and
            row['macd_bearish'] and
            row['volume_above_avg']
        )

        if bullish:
            return SignalType.BULLISH
        elif bearish:
            return SignalType.BEARISH
        else:
            return SignalType.NEUTRAL

    def _check_entry_trigger(self, row: pd.Series, history: pd.DataFrame,
                            signal: SignalType) -> Tuple[bool, str]:
        """Check if entry trigger conditions are met."""
        if len(history) < 2:
            return False, "NO_DATA"

        prev_row = history.iloc[-2]

        if signal == SignalType.BULLISH:
            # Break above previous day high
            if row['high'] > prev_row['high'] and row['volume_above_avg']:
                return True, "PREV_HIGH_BREAKOUT"
        elif signal == SignalType.BEARISH:
            # Break below previous day low
            if row['low'] < prev_row['low'] and row['volume_above_avg']:
                return True, "PREV_LOW_BREAKOUT"

        return False, "NO_TRIGGER"

    def _calculate_strike(self, nifty_price: float, option_type: OptionType) -> float:
        """Calculate strike price."""
        offset = self.config.get('option_selection', {}).get('strike_offset', 50)
        atm_strike = round(nifty_price / 50) * 50

        if option_type == OptionType.CALL:
            return atm_strike + offset
        else:
            return atm_strike - offset

    def _calculate_position_size(self, premium: float) -> Tuple[int, int]:
        """Calculate position size based on risk."""
        capital_config = self.config.get('capital', {})
        risk_percent = capital_config.get('risk_per_trade_percent', 2)
        lot_size = self.config.get('instrument', {}).get('lot_size', 25)
        max_lots = capital_config.get('max_lots', 3)

        risk_amount = self.current_capital * (risk_percent / 100)
        sl_percent = self.config.get('risk_reward', {}).get('stop_loss_percent', 33) / 100

        max_loss_per_qty = premium * sl_percent
        max_loss_per_lot = max_loss_per_qty * lot_size

        lots = int(risk_amount / max_loss_per_lot)
        lots = min(lots, max_lots)
        lots = max(lots, 1)

        return lots, lots * lot_size

    def _generate_results(self) -> BacktestResult:
        """Generate backtest results from trades."""
        if not self.trades:
            return self._empty_result()

        # Basic stats
        winning_trades = [t for t in self.trades if t.net_pnl > 0]
        losing_trades = [t for t in self.trades if t.net_pnl <= 0]

        total_profit = sum(t.net_pnl for t in winning_trades)
        total_loss = abs(sum(t.net_pnl for t in losing_trades))

        # Calculate drawdown
        peak = self.initial_capital
        max_drawdown = 0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            drawdown = peak - equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # Calculate consecutive wins/losses
        max_consec_wins = 0
        max_consec_losses = 0
        current_wins = 0
        current_losses = 0

        for trade in self.trades:
            if trade.net_pnl > 0:
                current_wins += 1
                current_losses = 0
                max_consec_wins = max(max_consec_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consec_losses = max(max_consec_losses, current_losses)

        # Monthly returns
        monthly_returns = {}
        for trade in self.trades:
            month_key = trade.date.strftime('%Y-%m')
            if month_key not in monthly_returns:
                monthly_returns[month_key] = 0
            monthly_returns[month_key] += trade.net_pnl

        # Convert to percentages
        for month in monthly_returns:
            monthly_returns[month] = (monthly_returns[month] / self.initial_capital) * 100

        # Sharpe ratio (simplified)
        returns = [t.return_percent for t in self.trades]
        if len(returns) > 1:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        else:
            sharpe = 0

        return BacktestResult(
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=self.initial_capital,
            final_capital=self.current_capital,
            total_trades=len(self.trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=(len(winning_trades) / len(self.trades)) * 100 if self.trades else 0,
            total_profit=total_profit,
            total_loss=total_loss,
            net_pnl=self.current_capital - self.initial_capital,
            roi_percent=((self.current_capital - self.initial_capital) / self.initial_capital) * 100,
            average_win=total_profit / len(winning_trades) if winning_trades else 0,
            average_loss=total_loss / len(losing_trades) if losing_trades else 0,
            profit_factor=total_profit / total_loss if total_loss > 0 else float('inf'),
            max_drawdown=max_drawdown,
            max_drawdown_percent=(max_drawdown / self.initial_capital) * 100,
            sharpe_ratio=sharpe,
            avg_holding_time="4h",
            best_trade=max(t.net_pnl for t in self.trades) if self.trades else 0,
            worst_trade=min(t.net_pnl for t in self.trades) if self.trades else 0,
            consecutive_wins_max=max_consec_wins,
            consecutive_losses_max=max_consec_losses,
            monthly_returns=monthly_returns,
            trades=self.trades,
            equity_curve=self.equity_curve
        )

    def _empty_result(self) -> BacktestResult:
        """Return empty result when no trades."""
        return BacktestResult(
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=self.initial_capital,
            final_capital=self.initial_capital,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            total_profit=0,
            total_loss=0,
            net_pnl=0,
            roi_percent=0,
            average_win=0,
            average_loss=0,
            profit_factor=0,
            max_drawdown=0,
            max_drawdown_percent=0,
            sharpe_ratio=0,
            avg_holding_time="0",
            best_trade=0,
            worst_trade=0,
            consecutive_wins_max=0,
            consecutive_losses_max=0
        )

    def print_results(self, result: BacktestResult):
        """Print backtest results in a formatted way."""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)

        print(f"\nPeriod: {result.start_date.date()} to {result.end_date.date()}")
        print(f"Initial Capital: ₹{result.initial_capital:,.2f}")
        print(f"Final Capital: ₹{result.final_capital:,.2f}")

        print("\n--- PERFORMANCE ---")
        print(f"Net P&L: ₹{result.net_pnl:,.2f}")
        print(f"ROI: {result.roi_percent:.2f}%")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")

        print("\n--- TRADE STATISTICS ---")
        print(f"Total Trades: {result.total_trades}")
        print(f"Winning Trades: {result.winning_trades} ({result.win_rate:.1f}%)")
        print(f"Losing Trades: {result.losing_trades}")

        print("\n--- PROFIT/LOSS ---")
        print(f"Total Profit: ₹{result.total_profit:,.2f}")
        print(f"Total Loss: ₹{result.total_loss:,.2f}")
        print(f"Average Win: ₹{result.average_win:,.2f}")
        print(f"Average Loss: ₹{result.average_loss:,.2f}")
        print(f"Profit Factor: {result.profit_factor:.2f}")

        print("\n--- RISK METRICS ---")
        print(f"Max Drawdown: ₹{result.max_drawdown:,.2f} ({result.max_drawdown_percent:.1f}%)")
        print(f"Best Trade: ₹{result.best_trade:,.2f}")
        print(f"Worst Trade: ₹{result.worst_trade:,.2f}")
        print(f"Max Consecutive Wins: {result.consecutive_wins_max}")
        print(f"Max Consecutive Losses: {result.consecutive_losses_max}")

        if result.monthly_returns:
            print("\n--- MONTHLY RETURNS ---")
            for month, ret in sorted(result.monthly_returns.items()):
                indicator = "+" if ret > 0 else ""
                print(f"  {month}: {indicator}{ret:.2f}%")

        print("\n" + "=" * 60)

    def export_results(self, result: BacktestResult, filename: str):
        """Export results to CSV."""
        trades_df = pd.DataFrame([
            {
                'Trade ID': t.trade_id,
                'Date': t.date,
                'Type': t.option_type,
                'Strike': t.strike,
                'Entry Price': t.entry_price,
                'Exit Price': t.exit_price,
                'Exit Reason': t.exit_reason,
                'Quantity': t.quantity,
                'Lots': t.lots,
                'Gross P&L': t.gross_pnl,
                'Commission': t.commission,
                'Net P&L': t.net_pnl,
                'Return %': t.return_percent
            }
            for t in result.trades
        ])

        trades_df.to_csv(filename, index=False)
        logger.info(f"Results exported to {filename}")
