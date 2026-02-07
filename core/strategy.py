"""
Trading Strategy Module
Implements the Nifty Options Trend Momentum Strategy
"""

import pandas as pd
from datetime import datetime, time
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from .indicators import TechnicalIndicators, IntradayTrigger


class SignalType(Enum):
    """Trading signal types."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    NO_DATA = "NO_DATA"


class OptionType(Enum):
    """Option types."""
    CALL = "CE"
    PUT = "PE"


class TradeStatus(Enum):
    """Trade status."""
    PENDING = "PENDING"
    OPEN = "OPEN"
    PARTIAL_EXIT = "PARTIAL_EXIT"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


@dataclass
class TradePosition:
    """Represents an open trade position."""
    trade_id: str
    option_type: OptionType
    strike: float
    entry_price: float
    entry_time: datetime
    quantity: int
    lots: int
    stop_loss: float
    target: float
    status: TradeStatus = TradeStatus.OPEN
    current_price: float = 0.0
    exit_price: float = 0.0
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    trigger_type: str = ""
    partial_exit_price: float = 0.0
    partial_exit_quantity: int = 0
    trailing_sl: float = 0.0
    notes: List[str] = field(default_factory=list)


class TradingStrategy:
    """
    Nifty Options Trend Momentum Strategy.

    Entry Rules:
    - CALL: Price > 20 EMA > 50 EMA, RSI 50-70, MACD bullish, Volume above avg
    - PUT: Price < 20 EMA < 50 EMA, RSI 30-50, MACD bearish, Volume above avg

    Exit Rules:
    - SL: 33% below entry
    - Target: 100% above entry (1:3 R:R)
    - Trailing SL after 50% profit
    - Time-based exit at 3:00 PM
    """

    def __init__(self, config: dict):
        """
        Initialize strategy with configuration.

        Args:
            config: Full configuration dictionary
        """
        self.config = config

        # Initialize indicators
        daily_config = config.get('daily_conditions', {})
        self.indicators = TechnicalIndicators(daily_config)

        # Initialize intraday trigger
        intraday_config = config.get('intraday_trigger', {})
        self.intraday_trigger = IntradayTrigger(intraday_config)

        # Trading parameters
        self.trading_hours = config.get('trading_hours', {})
        self.trading_days = config.get('trading_days', {})
        self.limits = config.get('limits', {})
        self.filters = config.get('filters', {})
        self.risk_reward = config.get('risk_reward', {})
        self.trailing = config.get('trailing', {})
        self.option_selection = config.get('option_selection', {})

        # State tracking
        self.current_position: Optional[TradePosition] = None
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.today_date: Optional[datetime] = None

        # Daily signals cache
        self.daily_signal: Optional[SignalType] = None
        self.daily_data: Optional[pd.DataFrame] = None

    def reset_daily_state(self, date: datetime):
        """Reset daily tracking variables."""
        if self.today_date is None or self.today_date.date() != date.date():
            self.today_date = date
            self.daily_trades = 0
            self.intraday_trigger.reset_first_candle()
            logger.info(f"Daily state reset for {date.date()}")

    def is_trading_day(self, date: datetime) -> bool:
        """Check if the given date is a valid trading day."""
        day_name = date.strftime('%A').lower()
        return self.trading_days.get(day_name, False)

    def is_trading_time(self, current_time: time) -> Tuple[bool, str]:
        """
        Check if current time is within trading hours.

        Returns:
            Tuple of (is_valid, reason)
        """
        entry_start = datetime.strptime(
            self.trading_hours.get('entry_start', '09:30'), '%H:%M'
        ).time()
        entry_end = datetime.strptime(
            self.trading_hours.get('entry_end', '10:30'), '%H:%M'
        ).time()
        market_open = datetime.strptime(
            self.trading_hours.get('market_open', '09:15'), '%H:%M'
        ).time()

        if current_time < market_open:
            return False, "MARKET_NOT_OPEN"

        if current_time < entry_start:
            return False, "BEFORE_ENTRY_WINDOW"

        if current_time > entry_end:
            return False, "AFTER_ENTRY_WINDOW"

        return True, "OK"

    def is_exit_time(self, current_time: time) -> bool:
        """Check if it's time-based exit time."""
        exit_time = datetime.strptime(
            self.trading_hours.get('time_based_exit', '15:00'), '%H:%M'
        ).time()
        return current_time >= exit_time

    def can_take_new_trade(self) -> Tuple[bool, str]:
        """
        Check if a new trade can be taken based on limits.

        Returns:
            Tuple of (can_trade, reason)
        """
        # Check if position already open
        if self.current_position is not None:
            return False, "POSITION_OPEN"

        # Check daily trade limit
        max_trades = self.limits.get('max_trades_per_day', 2)
        if self.daily_trades >= max_trades:
            return False, "MAX_DAILY_TRADES"

        # Check consecutive losses
        max_losses = self.limits.get('max_consecutive_losses', 3)
        if (self.limits.get('stop_after_consecutive_losses', True) and
            self.consecutive_losses >= max_losses):
            return False, "MAX_CONSECUTIVE_LOSSES"

        return True, "OK"

    def update_daily_analysis(self, daily_df: pd.DataFrame) -> SignalType:
        """
        Update daily analysis and determine trend direction.

        Args:
            daily_df: Daily OHLCV dataframe

        Returns:
            SignalType indicating market direction
        """
        # Add indicators
        df = self.indicators.add_all_indicators(daily_df)
        self.daily_data = df

        # Get latest signals
        daily_config = self.config.get('daily_conditions', {})
        signals = self.indicators.get_latest_signals(df, daily_config)

        if signals['bullish']:
            self.daily_signal = SignalType.BULLISH
        elif signals['bearish']:
            self.daily_signal = SignalType.BEARISH
        else:
            self.daily_signal = SignalType.NEUTRAL

        logger.info(f"Daily analysis updated: {self.daily_signal.value}")
        logger.debug(f"Indicators - RSI: {signals['rsi']:.2f}, "
                    f"MACD: {signals['macd']:.2f}, "
                    f"Close: {signals['close']:.2f}")

        return self.daily_signal

    def check_entry_trigger(self, current_price: float, current_volume: float,
                           prev_day_high: float, prev_day_low: float,
                           avg_volume: float) -> Tuple[bool, OptionType, str]:
        """
        Check if entry trigger conditions are met.

        Args:
            current_price: Current Nifty price
            current_volume: Current candle volume
            prev_day_high: Previous day high
            prev_day_low: Previous day low
            avg_volume: Average volume

        Returns:
            Tuple of (trigger_met, option_type, trigger_reason)
        """
        if self.daily_signal == SignalType.BULLISH:
            triggered, trigger_type = self.intraday_trigger.check_bullish_trigger(
                current_price, prev_day_high, current_volume, avg_volume
            )
            if triggered:
                return True, OptionType.CALL, trigger_type

        elif self.daily_signal == SignalType.BEARISH:
            triggered, trigger_type = self.intraday_trigger.check_bearish_trigger(
                current_price, prev_day_low, current_volume, avg_volume
            )
            if triggered:
                return True, OptionType.PUT, trigger_type

        return False, None, "NO_TRIGGER"

    def calculate_strike(self, nifty_price: float, option_type: OptionType) -> float:
        """
        Calculate the strike price for option.

        Args:
            nifty_price: Current Nifty price
            option_type: CALL or PUT

        Returns:
            Strike price (rounded to nearest 50)
        """
        offset = self.option_selection.get('strike_offset', 50)

        # Round to nearest 50
        atm_strike = round(nifty_price / 50) * 50

        if option_type == OptionType.CALL:
            # Slightly OTM call (50-100 points above ATM)
            strike = atm_strike + offset
        else:
            # Slightly OTM put (50-100 points below ATM)
            strike = atm_strike - offset

        return strike

    def calculate_position_size(self, premium: float, capital: float,
                               risk_percent: float, lot_size: int) -> Tuple[int, int]:
        """
        Calculate position size based on risk management.

        Args:
            premium: Option premium
            capital: Total capital
            risk_percent: Risk percentage per trade
            lot_size: Lot size for the instrument

        Returns:
            Tuple of (lots, total_quantity)
        """
        risk_amount = capital * (risk_percent / 100)
        sl_percent = self.risk_reward.get('stop_loss_percent', 33) / 100

        # Maximum loss per lot
        max_loss_per_qty = premium * sl_percent
        max_loss_per_lot = max_loss_per_qty * lot_size

        # Calculate lots based on risk
        lots = int(risk_amount / max_loss_per_lot)

        # Apply limits
        max_lots = self.config.get('capital', {}).get('max_lots', 3)
        lots = min(lots, max_lots)
        lots = max(lots, 1)  # Minimum 1 lot

        total_quantity = lots * lot_size

        logger.debug(f"Position size: {lots} lots ({total_quantity} qty) "
                    f"for premium {premium}, risk {risk_amount}")

        return lots, total_quantity

    def calculate_sl_target(self, entry_price: float) -> Tuple[float, float]:
        """
        Calculate stop loss and target prices.

        Args:
            entry_price: Entry premium

        Returns:
            Tuple of (stop_loss, target)
        """
        sl_percent = self.risk_reward.get('stop_loss_percent', 33) / 100
        target_percent = self.risk_reward.get('target_percent', 100) / 100

        stop_loss = entry_price * (1 - sl_percent)
        target = entry_price * (1 + target_percent)

        return round(stop_loss, 2), round(target, 2)

    def create_trade(self, option_type: OptionType, strike: float,
                    entry_price: float, lots: int, quantity: int,
                    trigger_type: str) -> TradePosition:
        """
        Create a new trade position.

        Args:
            option_type: CALL or PUT
            strike: Strike price
            entry_price: Entry premium
            lots: Number of lots
            quantity: Total quantity
            trigger_type: Entry trigger reason

        Returns:
            TradePosition object
        """
        stop_loss, target = self.calculate_sl_target(entry_price)

        trade = TradePosition(
            trade_id=f"TRADE_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            option_type=option_type,
            strike=strike,
            entry_price=entry_price,
            entry_time=datetime.now(),
            quantity=quantity,
            lots=lots,
            stop_loss=stop_loss,
            target=target,
            trigger_type=trigger_type,
            trailing_sl=stop_loss
        )

        self.current_position = trade
        self.daily_trades += 1

        logger.info(f"Trade created: {trade.trade_id}")
        logger.info(f"  {option_type.value} {strike} @ {entry_price}")
        logger.info(f"  SL: {stop_loss}, Target: {target}, Qty: {quantity}")

        return trade

    def update_trailing_sl(self, current_price: float) -> Optional[float]:
        """
        Update trailing stop loss based on current price.

        Args:
            current_price: Current option premium

        Returns:
            New trailing SL or None if no update
        """
        if self.current_position is None:
            return None

        trade = self.current_position
        entry = trade.entry_price

        if not self.trailing.get('enabled', True):
            return None

        profit_percent = ((current_price - entry) / entry) * 100

        breakeven_at = self.trailing.get('move_to_breakeven_at', 50)
        lock_at = self.trailing.get('lock_profit_at', 75)

        new_sl = trade.trailing_sl

        # Move to breakeven at 50% profit
        if profit_percent >= breakeven_at and trade.trailing_sl < entry:
            new_sl = entry
            trade.notes.append(f"SL moved to breakeven at {profit_percent:.1f}% profit")

        # Lock 50% profit at 75% gain
        if profit_percent >= lock_at:
            lock_price = entry * 1.5  # 50% profit locked
            if trade.trailing_sl < lock_price:
                new_sl = lock_price
                trade.notes.append(f"50% profit locked at {profit_percent:.1f}% gain")

        if new_sl != trade.trailing_sl:
            trade.trailing_sl = new_sl
            logger.info(f"Trailing SL updated to {new_sl}")
            return new_sl

        return None

    def check_exit_conditions(self, current_price: float,
                             current_time: time) -> Tuple[bool, str, float]:
        """
        Check if exit conditions are met.

        Args:
            current_price: Current option premium
            current_time: Current time

        Returns:
            Tuple of (should_exit, reason, exit_quantity_percent)
        """
        if self.current_position is None:
            return False, "NO_POSITION", 0

        trade = self.current_position

        # Check stop loss
        if current_price <= trade.trailing_sl:
            return True, "STOP_LOSS", 100

        # Check target
        if current_price >= trade.target:
            partial_at = self.trailing.get('partial_booking_at', 100)
            if trade.status != TradeStatus.PARTIAL_EXIT:
                return True, "TARGET_PARTIAL", 50  # Book 50% at target
            else:
                return True, "TARGET_FULL", 100  # Exit remaining

        # Check time-based exit
        if self.is_exit_time(current_time):
            return True, "TIME_EXIT", 100

        return False, "HOLD", 0

    def execute_exit(self, exit_price: float, exit_reason: str,
                    exit_percent: float) -> Dict:
        """
        Execute exit and calculate P&L.

        Args:
            exit_price: Exit premium
            exit_reason: Reason for exit
            exit_percent: Percentage of position to exit

        Returns:
            Dictionary with exit details
        """
        if self.current_position is None:
            return {'success': False, 'reason': 'NO_POSITION'}

        trade = self.current_position
        exit_quantity = int(trade.quantity * (exit_percent / 100))

        # Calculate P&L for this exit
        pnl_per_qty = exit_price - trade.entry_price
        exit_pnl = pnl_per_qty * exit_quantity

        if exit_percent < 100:
            # Partial exit
            trade.status = TradeStatus.PARTIAL_EXIT
            trade.partial_exit_price = exit_price
            trade.partial_exit_quantity = exit_quantity
            trade.pnl += exit_pnl

            # Update trailing SL for remaining position
            trail_points = self.trailing.get('trail_points_after_partial', 20)
            trade.trailing_sl = exit_price - trail_points

            trade.notes.append(f"Partial exit: {exit_quantity} qty @ {exit_price}")

            logger.info(f"Partial exit: {exit_quantity} qty @ {exit_price}, "
                       f"PnL: {exit_pnl:.2f}")

            return {
                'success': True,
                'type': 'PARTIAL',
                'quantity': exit_quantity,
                'price': exit_price,
                'pnl': exit_pnl,
                'reason': exit_reason
            }

        else:
            # Full exit
            remaining_qty = trade.quantity - trade.partial_exit_quantity
            remaining_pnl = pnl_per_qty * remaining_qty
            trade.pnl += remaining_pnl

            trade.exit_price = exit_price
            trade.exit_time = datetime.now()
            trade.status = TradeStatus.CLOSED

            # Update consecutive losses tracker
            if trade.pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0

            trade.notes.append(f"Full exit @ {exit_price}, Total PnL: {trade.pnl:.2f}")

            logger.info(f"Trade closed: {trade.trade_id}")
            logger.info(f"  Exit @ {exit_price}, Reason: {exit_reason}")
            logger.info(f"  Total PnL: {trade.pnl:.2f}")

            # Clear current position
            closed_trade = trade
            self.current_position = None

            return {
                'success': True,
                'type': 'FULL',
                'quantity': remaining_qty,
                'price': exit_price,
                'pnl': trade.pnl,
                'reason': exit_reason,
                'trade': closed_trade
            }

    def get_position_summary(self) -> Optional[Dict]:
        """Get current position summary."""
        if self.current_position is None:
            return None

        trade = self.current_position
        unrealized_pnl = (trade.current_price - trade.entry_price) * trade.quantity

        return {
            'trade_id': trade.trade_id,
            'option_type': trade.option_type.value,
            'strike': trade.strike,
            'entry_price': trade.entry_price,
            'current_price': trade.current_price,
            'stop_loss': trade.stop_loss,
            'trailing_sl': trade.trailing_sl,
            'target': trade.target,
            'quantity': trade.quantity,
            'lots': trade.lots,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': trade.pnl,
            'status': trade.status.value
        }

    def get_daily_summary(self) -> Dict:
        """Get daily trading summary."""
        return {
            'date': self.today_date.date() if self.today_date else None,
            'daily_signal': self.daily_signal.value if self.daily_signal else None,
            'trades_taken': self.daily_trades,
            'consecutive_losses': self.consecutive_losses,
            'position_open': self.current_position is not None
        }
