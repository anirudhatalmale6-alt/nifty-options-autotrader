"""
Logging and Trade Journal Module
Handles all logging, trade recording, and reporting
"""

import os
import json
import csv
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from loguru import logger
import sys


class TradingLogger:
    """
    Centralized logging for the trading system.

    Handles:
    - Console and file logging
    - Trade journal with detailed records
    - Daily summaries
    - Performance reports
    """

    def __init__(self, config: dict):
        """
        Initialize the logger.

        Args:
            config: Logging configuration
        """
        log_config = config.get('logging', {})

        self.log_level = log_config.get('level', 'INFO')
        self.log_to_file = log_config.get('log_to_file', True)
        self.log_dir = log_config.get('log_dir', 'logs')
        self.trade_journal_enabled = log_config.get('trade_journal', True)

        # Create log directory
        self.log_path = Path(self.log_dir)
        self.log_path.mkdir(parents=True, exist_ok=True)

        # Journal file paths
        self.journal_file = self.log_path / 'trade_journal.csv'
        self.daily_summary_file = self.log_path / 'daily_summary.json'

        # Configure loguru
        self._configure_logger()

    def _configure_logger(self):
        """Configure loguru logger."""
        # Remove default handler
        logger.remove()

        # Console handler with colors
        console_format = (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        logger.add(
            sys.stderr,
            format=console_format,
            level=self.log_level,
            colorize=True
        )

        if self.log_to_file:
            # Main log file (rotates daily)
            log_file = self.log_path / "trader_{time:YYYY-MM-DD}.log"
            file_format = (
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message}"
            )
            logger.add(
                str(log_file),
                format=file_format,
                level=self.log_level,
                rotation="00:00",  # New file at midnight
                retention="30 days",
                compression="gz"
            )

            # Separate error log
            error_file = self.log_path / "errors_{time:YYYY-MM-DD}.log"
            logger.add(
                str(error_file),
                format=file_format,
                level="ERROR",
                rotation="00:00",
                retention="30 days"
            )

            # Trade-specific log
            trade_file = self.log_path / "trades_{time:YYYY-MM-DD}.log"
            logger.add(
                str(trade_file),
                format=file_format,
                level="INFO",
                filter=lambda record: "TRADE" in record["message"],
                rotation="00:00",
                retention="90 days"
            )

        logger.info("Logger initialized")

    def log_trade_entry(self, trade_data: Dict):
        """
        Log a trade entry.

        Args:
            trade_data: Dictionary with trade details
        """
        logger.info(f"TRADE ENTRY: {trade_data.get('option_type')} "
                   f"{trade_data.get('strike')} @ ₹{trade_data.get('entry_price')}")

        if self.trade_journal_enabled:
            self._write_journal_entry({
                'timestamp': datetime.now().isoformat(),
                'event': 'ENTRY',
                **trade_data
            })

    def log_trade_exit(self, trade_data: Dict):
        """
        Log a trade exit.

        Args:
            trade_data: Dictionary with trade details including P&L
        """
        pnl = trade_data.get('pnl', 0)
        pnl_str = f"+₹{pnl:.2f}" if pnl >= 0 else f"-₹{abs(pnl):.2f}"

        logger.info(f"TRADE EXIT: {trade_data.get('exit_reason')} | "
                   f"P&L: {pnl_str}")

        if self.trade_journal_enabled:
            self._write_journal_entry({
                'timestamp': datetime.now().isoformat(),
                'event': 'EXIT',
                **trade_data
            })

    def log_signal(self, signal_type: str, details: Dict):
        """Log a trading signal."""
        logger.info(f"SIGNAL: {signal_type} | RSI: {details.get('rsi', 0):.1f} | "
                   f"MACD: {details.get('macd', 0):.2f}")

    def log_order(self, order_type: str, order_data: Dict):
        """Log order placement/modification/cancellation."""
        logger.info(f"ORDER {order_type}: {order_data}")

    def log_error(self, error_type: str, error_message: str, details: Dict = None):
        """Log an error with context."""
        logger.error(f"{error_type}: {error_message}")
        if details:
            logger.error(f"Error details: {details}")

    def log_system_event(self, event: str, details: Dict = None):
        """Log system events (startup, shutdown, connection changes)."""
        logger.info(f"SYSTEM: {event}")
        if details:
            logger.debug(f"Details: {details}")

    def _write_journal_entry(self, entry: Dict):
        """Write entry to trade journal CSV."""
        file_exists = self.journal_file.exists()

        with open(self.journal_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=entry.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(entry)

    def get_trade_journal(self, from_date: datetime = None,
                         to_date: datetime = None) -> List[Dict]:
        """
        Read trade journal entries.

        Args:
            from_date: Filter from date
            to_date: Filter to date

        Returns:
            List of trade journal entries
        """
        if not self.journal_file.exists():
            return []

        entries = []
        with open(self.journal_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry_date = datetime.fromisoformat(row['timestamp'])
                if from_date and entry_date < from_date:
                    continue
                if to_date and entry_date > to_date:
                    continue
                entries.append(row)

        return entries

    def save_daily_summary(self, summary: Dict):
        """
        Save daily trading summary.

        Args:
            summary: Dictionary with daily statistics
        """
        summaries = {}

        # Load existing summaries
        if self.daily_summary_file.exists():
            with open(self.daily_summary_file, 'r') as f:
                summaries = json.load(f)

        # Add today's summary
        date_key = datetime.now().strftime('%Y-%m-%d')
        summaries[date_key] = summary

        # Save
        with open(self.daily_summary_file, 'w') as f:
            json.dump(summaries, f, indent=2, default=str)

        logger.info(f"Daily summary saved for {date_key}")

    def get_daily_summaries(self, days: int = 30) -> Dict:
        """
        Get recent daily summaries.

        Args:
            days: Number of days to retrieve

        Returns:
            Dictionary of date -> summary
        """
        if not self.daily_summary_file.exists():
            return {}

        with open(self.daily_summary_file, 'r') as f:
            summaries = json.load(f)

        # Filter to recent days
        cutoff = datetime.now().date() - timedelta(days=days)
        return {
            k: v for k, v in summaries.items()
            if datetime.strptime(k, '%Y-%m-%d').date() >= cutoff
        }

    def generate_performance_report(self) -> Dict:
        """
        Generate overall performance report from journal.

        Returns:
            Performance statistics
        """
        entries = self.get_trade_journal()

        if not entries:
            return {'total_trades': 0}

        # Filter exits only
        exits = [e for e in entries if e.get('event') == 'EXIT']

        if not exits:
            return {'total_trades': 0}

        # Calculate stats
        pnls = [float(e.get('pnl', 0)) for e in exits]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        return {
            'total_trades': len(exits),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': (len(wins) / len(exits)) * 100 if exits else 0,
            'total_pnl': sum(pnls),
            'average_pnl': sum(pnls) / len(pnls) if pnls else 0,
            'average_win': sum(wins) / len(wins) if wins else 0,
            'average_loss': abs(sum(losses)) / len(losses) if losses else 0,
            'profit_factor': sum(wins) / abs(sum(losses)) if losses else float('inf'),
            'best_trade': max(pnls) if pnls else 0,
            'worst_trade': min(pnls) if pnls else 0
        }


class AlertManager:
    """
    Manage trading alerts and notifications.

    Supports:
    - Console alerts
    - Sound alerts (optional)
    - Future: Email, Telegram, etc.
    """

    def __init__(self, config: dict):
        """
        Initialize alert manager.

        Args:
            config: Alert configuration
        """
        self.enabled = True

    def send_alert(self, alert_type: str, message: str, priority: str = "normal"):
        """
        Send an alert.

        Args:
            alert_type: Type of alert (ENTRY, EXIT, ERROR, etc.)
            message: Alert message
            priority: Alert priority (low, normal, high, critical)
        """
        timestamp = datetime.now().strftime('%H:%M:%S')

        # Console alert with visual emphasis
        if priority == "critical":
            print(f"\n{'!'*60}")
            print(f"!!! CRITICAL ALERT - {alert_type} !!!")
            print(f"    {message}")
            print(f"    Time: {timestamp}")
            print(f"{'!'*60}\n")
        elif priority == "high":
            print(f"\n*** ALERT: {alert_type} ***")
            print(f"    {message}")
            print(f"    Time: {timestamp}\n")
        else:
            print(f"[{timestamp}] {alert_type}: {message}")

        logger.info(f"ALERT ({priority}): {alert_type} - {message}")

    def trade_entry_alert(self, trade_details: Dict):
        """Send trade entry alert."""
        msg = (f"{trade_details.get('option_type')} {trade_details.get('strike')} "
               f"@ ₹{trade_details.get('entry_price')} | "
               f"SL: ₹{trade_details.get('stop_loss')} | "
               f"Target: ₹{trade_details.get('target')}")
        self.send_alert("TRADE ENTRY", msg, "high")

    def trade_exit_alert(self, trade_details: Dict):
        """Send trade exit alert."""
        pnl = trade_details.get('pnl', 0)
        pnl_str = f"+₹{pnl:.2f}" if pnl >= 0 else f"-₹{abs(pnl):.2f}"
        msg = (f"Exit @ ₹{trade_details.get('exit_price')} | "
               f"Reason: {trade_details.get('exit_reason')} | "
               f"P&L: {pnl_str}")

        priority = "high" if abs(pnl) > 1000 else "normal"
        self.send_alert("TRADE EXIT", msg, priority)

    def error_alert(self, error_message: str):
        """Send error alert."""
        self.send_alert("ERROR", error_message, "critical")

    def connection_alert(self, status: str):
        """Send connection status alert."""
        priority = "critical" if status == "LOST" else "normal"
        self.send_alert("CONNECTION", f"Status: {status}", priority)

    def daily_summary_alert(self, summary: Dict):
        """Send daily summary alert."""
        msg = (f"Trades: {summary.get('trades', 0)} | "
               f"P&L: ₹{summary.get('pnl', 0):.2f} | "
               f"Win Rate: {summary.get('win_rate', 0):.1f}%")
        self.send_alert("DAILY SUMMARY", msg, "normal")


# Import timedelta for get_daily_summaries
from datetime import timedelta
