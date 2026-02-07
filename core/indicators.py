"""
Technical Indicators Module
Calculates EMA, RSI, MACD, Volume indicators for Nifty trading strategy
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from loguru import logger


class TechnicalIndicators:
    """Calculate technical indicators for the trading strategy."""

    def __init__(self, config: dict):
        """
        Initialize with configuration settings.

        Args:
            config: Dictionary containing indicator parameters
        """
        self.ema_fast = config.get('ema_fast', 20)
        self.ema_slow = config.get('ema_slow', 50)
        self.rsi_period = config.get('rsi_period', 14)
        self.macd_fast = config.get('macd_fast', 12)
        self.macd_slow = config.get('macd_slow', 26)
        self.macd_signal = config.get('macd_signal', 9)
        self.volume_ma_period = config.get('volume_ma_period', 20)

    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average.

        Args:
            data: Price series (typically close prices)
            period: EMA period

        Returns:
            EMA series
        """
        return data.ewm(span=period, adjust=False).mean()

    def calculate_rsi(self, data: pd.Series, period: int = None) -> pd.Series:
        """
        Calculate Relative Strength Index.

        Args:
            data: Price series (typically close prices)
            period: RSI period (default from config)

        Returns:
            RSI series (0-100)
        """
        if period is None:
            period = self.rsi_period

        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_macd(self, data: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            data: Price series (typically close prices)

        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        ema_fast = self.calculate_ema(data, self.macd_fast)
        ema_slow = self.calculate_ema(data, self.macd_slow)

        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, self.macd_signal)
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def calculate_volume_ma(self, volume: pd.Series, period: int = None) -> pd.Series:
        """
        Calculate Volume Moving Average.

        Args:
            volume: Volume series
            period: MA period (default from config)

        Returns:
            Volume MA series
        """
        if period is None:
            period = self.volume_ma_period

        return volume.rolling(window=period).mean()

    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all required indicators to the dataframe.

        Args:
            df: OHLCV dataframe with columns: open, high, low, close, volume

        Returns:
            DataFrame with all indicators added
        """
        df = df.copy()

        # Ensure column names are lowercase
        df.columns = df.columns.str.lower()

        # EMAs
        df['ema_20'] = self.calculate_ema(df['close'], self.ema_fast)
        df['ema_50'] = self.calculate_ema(df['close'], self.ema_slow)

        # RSI
        df['rsi'] = self.calculate_rsi(df['close'])

        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = self.calculate_macd(df['close'])

        # Volume MA
        df['volume_ma'] = self.calculate_volume_ma(df['volume'])

        # Price position relative to EMAs
        df['above_ema_20'] = df['close'] > df['ema_20']
        df['above_ema_50'] = df['close'] > df['ema_50']
        df['ema_20_above_50'] = df['ema_20'] > df['ema_50']

        # Volume above average
        df['volume_above_avg'] = df['volume'] > df['volume_ma']

        # MACD crossover signals
        df['macd_bullish'] = df['macd'] > df['macd_signal']
        df['macd_bearish'] = df['macd'] < df['macd_signal']

        # Higher highs and higher lows (for trend confirmation)
        df['higher_high'] = df['high'] > df['high'].shift(1)
        df['higher_low'] = df['low'] > df['low'].shift(1)
        df['lower_high'] = df['high'] < df['high'].shift(1)
        df['lower_low'] = df['low'] < df['low'].shift(1)

        # Previous day high/low for intraday triggers
        df['prev_high'] = df['high'].shift(1)
        df['prev_low'] = df['low'].shift(1)

        logger.debug(f"Added indicators to dataframe with {len(df)} rows")

        return df

    def check_bullish_trend(self, df: pd.DataFrame, rsi_min: float = 50,
                            rsi_max: float = 70) -> pd.Series:
        """
        Check if bullish trend conditions are met.

        Conditions:
        1. Price > 20 EMA > 50 EMA
        2. RSI between 50-70
        3. MACD line above signal line
        4. Volume above 20-day average

        Args:
            df: DataFrame with indicators
            rsi_min: Minimum RSI for bullish
            rsi_max: Maximum RSI for bullish

        Returns:
            Boolean series indicating bullish trend
        """
        bullish = (
            df['above_ema_20'] &
            df['above_ema_50'] &
            df['ema_20_above_50'] &
            (df['rsi'] >= rsi_min) &
            (df['rsi'] <= rsi_max) &
            df['macd_bullish'] &
            df['volume_above_avg']
        )

        return bullish

    def check_bearish_trend(self, df: pd.DataFrame, rsi_min: float = 30,
                            rsi_max: float = 50) -> pd.Series:
        """
        Check if bearish trend conditions are met.

        Conditions:
        1. Price < 20 EMA < 50 EMA
        2. RSI between 30-50
        3. MACD line below signal line
        4. Volume above 20-day average

        Args:
            df: DataFrame with indicators
            rsi_min: Minimum RSI for bearish
            rsi_max: Maximum RSI for bearish

        Returns:
            Boolean series indicating bearish trend
        """
        bearish = (
            ~df['above_ema_20'] &
            ~df['above_ema_50'] &
            ~df['ema_20_above_50'] &
            (df['rsi'] >= rsi_min) &
            (df['rsi'] <= rsi_max) &
            df['macd_bearish'] &
            df['volume_above_avg']
        )

        return bearish

    def get_latest_signals(self, df: pd.DataFrame, config: dict) -> dict:
        """
        Get the latest trading signals from the dataframe.

        Args:
            df: DataFrame with indicators (must have at least 1 row)
            config: Configuration with RSI thresholds

        Returns:
            Dictionary with signal information
        """
        if df.empty:
            return {'signal': 'NO_DATA', 'bullish': False, 'bearish': False}

        latest = df.iloc[-1]

        rsi_bullish_min = config.get('rsi_bullish_min', 50)
        rsi_bullish_max = config.get('rsi_bullish_max', 70)
        rsi_bearish_min = config.get('rsi_bearish_min', 30)
        rsi_bearish_max = config.get('rsi_bearish_max', 50)

        bullish = (
            latest['above_ema_20'] and
            latest['above_ema_50'] and
            latest['ema_20_above_50'] and
            rsi_bullish_min <= latest['rsi'] <= rsi_bullish_max and
            latest['macd_bullish'] and
            latest['volume_above_avg']
        )

        bearish = (
            not latest['above_ema_20'] and
            not latest['above_ema_50'] and
            not latest['ema_20_above_50'] and
            rsi_bearish_min <= latest['rsi'] <= rsi_bearish_max and
            latest['macd_bearish'] and
            latest['volume_above_avg']
        )

        signal = 'BULLISH' if bullish else ('BEARISH' if bearish else 'NEUTRAL')

        return {
            'signal': signal,
            'bullish': bullish,
            'bearish': bearish,
            'close': latest['close'],
            'ema_20': latest['ema_20'],
            'ema_50': latest['ema_50'],
            'rsi': latest['rsi'],
            'macd': latest['macd'],
            'macd_signal': latest['macd_signal'],
            'volume': latest['volume'],
            'volume_ma': latest['volume_ma'],
            'prev_high': latest['prev_high'],
            'prev_low': latest['prev_low']
        }


class IntradayTrigger:
    """Handle intraday entry triggers on 15-min timeframe."""

    def __init__(self, config: dict):
        """
        Initialize with configuration.

        Args:
            config: Dictionary with intraday trigger settings
        """
        self.use_prev_day_breakout = config.get('use_prev_day_breakout', True)
        self.use_first_candle_breakout = config.get('use_first_candle_breakout', True)
        self.first_candle_high = None
        self.first_candle_low = None

    def set_first_candle(self, high: float, low: float):
        """
        Set the first 15-min candle high/low for the day.

        Args:
            high: First candle high
            low: First candle low
        """
        self.first_candle_high = high
        self.first_candle_low = low
        logger.info(f"First candle set - High: {high}, Low: {low}")

    def reset_first_candle(self):
        """Reset first candle values for new day."""
        self.first_candle_high = None
        self.first_candle_low = None

    def check_bullish_trigger(self, current_price: float, prev_day_high: float,
                              current_candle_volume: float, avg_volume: float) -> Tuple[bool, str]:
        """
        Check if bullish entry trigger is met.

        Triggers:
        1. Break above previous day high, OR
        2. Break above first 15-min candle high
        Plus: Strong volume confirmation

        Args:
            current_price: Current price
            prev_day_high: Previous day's high
            current_candle_volume: Current candle volume
            avg_volume: Average volume

        Returns:
            Tuple of (trigger_met, trigger_type)
        """
        volume_confirmed = current_candle_volume > avg_volume

        if not volume_confirmed:
            return False, "NO_VOLUME"

        if self.use_prev_day_breakout and current_price > prev_day_high:
            return True, "PREV_DAY_HIGH_BREAKOUT"

        if (self.use_first_candle_breakout and
            self.first_candle_high is not None and
            current_price > self.first_candle_high):
            return True, "FIRST_CANDLE_BREAKOUT"

        return False, "NO_TRIGGER"

    def check_bearish_trigger(self, current_price: float, prev_day_low: float,
                              current_candle_volume: float, avg_volume: float) -> Tuple[bool, str]:
        """
        Check if bearish entry trigger is met.

        Triggers:
        1. Break below previous day low, OR
        2. Break below first 15-min candle low
        Plus: Strong volume confirmation

        Args:
            current_price: Current price
            prev_day_low: Previous day's low
            current_candle_volume: Current candle volume
            avg_volume: Average volume

        Returns:
            Tuple of (trigger_met, trigger_type)
        """
        volume_confirmed = current_candle_volume > avg_volume

        if not volume_confirmed:
            return False, "NO_VOLUME"

        if self.use_prev_day_breakout and current_price < prev_day_low:
            return True, "PREV_DAY_LOW_BREAKOUT"

        if (self.use_first_candle_breakout and
            self.first_candle_low is not None and
            current_price < self.first_candle_low):
            return True, "FIRST_CANDLE_BREAKOUT"

        return False, "NO_TRIGGER"
