"""
Fyers API Client Module
Handles authentication, market data, and order placement with Fyers broker
"""

import os
import json
import webbrowser
import urllib.parse
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time

import requests
from loguru import logger

try:
    from fyers_apiv3 import fyersModel
    from fyers_apiv3.FyersWebsocket import data_ws
    FYERS_SDK_AVAILABLE = True
except ImportError:
    FYERS_SDK_AVAILABLE = False
    logger.warning("Fyers SDK not installed. Install with: pip install fyers-apiv3")


class AuthCallbackHandler(BaseHTTPRequestHandler):
    """Handle OAuth callback from Fyers."""

    auth_code = None

    def do_GET(self):
        """Handle GET request with auth code."""
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)

        if 'auth_code' in params:
            AuthCallbackHandler.auth_code = params['auth_code'][0]
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            response = b"""
            <html>
            <body style="font-family: Arial; text-align: center; padding: 50px;">
                <h1 style="color: green;">Authentication Successful!</h1>
                <p>You can close this window and return to the trading system.</p>
            </body>
            </html>
            """
            self.wfile.write(response)
        else:
            self.send_response(400)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"Authentication failed. No auth code received.")

    def log_message(self, format, *args):
        """Suppress HTTP server logs."""
        pass


class FyersClient:
    """
    Fyers API client for trading operations.

    Handles:
    - OAuth2 authentication
    - Market data fetching
    - Order placement and management
    - WebSocket streaming for real-time data
    """

    def __init__(self, config: dict):
        """
        Initialize Fyers client.

        Args:
            config: Configuration with Fyers credentials
        """
        self.config = config
        fyers_config = config.get('fyers', {})

        self.app_id = fyers_config.get('app_id', '')
        self.secret_key = fyers_config.get('secret_key', '')
        self.redirect_uri = fyers_config.get('redirect_uri', 'http://127.0.0.1:8000/')

        self.access_token = None
        self.fyers = None
        self.ws_client = None

        self.token_file = os.path.join(
            os.path.dirname(__file__), '..', 'config', '.fyers_token'
        )

        # Try to load existing token
        self._load_token()

    def _save_token(self, token: str):
        """Save access token to file."""
        try:
            token_data = {
                'access_token': token,
                'saved_at': datetime.now().isoformat(),
                'app_id': self.app_id
            }
            with open(self.token_file, 'w') as f:
                json.dump(token_data, f)
            logger.info("Access token saved")
        except Exception as e:
            logger.error(f"Failed to save token: {e}")

    def _load_token(self) -> bool:
        """Load access token from file if valid."""
        try:
            if os.path.exists(self.token_file):
                with open(self.token_file, 'r') as f:
                    token_data = json.load(f)

                # Check if token is from today and same app
                saved_at = datetime.fromisoformat(token_data['saved_at'])
                if (saved_at.date() == datetime.now().date() and
                    token_data.get('app_id') == self.app_id):
                    self.access_token = token_data['access_token']
                    self._init_fyers()
                    logger.info("Loaded existing access token")
                    return True
        except Exception as e:
            logger.debug(f"Could not load token: {e}")

        return False

    def _init_fyers(self):
        """Initialize Fyers model with access token."""
        if not FYERS_SDK_AVAILABLE:
            logger.error("Fyers SDK not available")
            return

        if self.access_token:
            self.fyers = fyersModel.FyersModel(
                client_id=self.app_id,
                token=self.access_token,
                is_async=False,
                log_path=""
            )

    def authenticate(self, auto_open_browser: bool = True) -> bool:
        """
        Authenticate with Fyers using OAuth2.

        Args:
            auto_open_browser: Auto-open browser for login

        Returns:
            True if authentication successful
        """
        if not FYERS_SDK_AVAILABLE:
            logger.error("Fyers SDK not installed")
            return False

        if not self.app_id or not self.secret_key:
            logger.error("Fyers credentials not configured")
            return False

        try:
            # Step 1: Generate auth code URL
            session = fyersModel.SessionModel(
                client_id=self.app_id,
                secret_key=self.secret_key,
                redirect_uri=self.redirect_uri,
                response_type="code",
                grant_type="authorization_code"
            )

            auth_url = session.generate_authcode()
            logger.info(f"Auth URL: {auth_url}")

            # Step 2: Start local server to capture callback
            server_address = ('127.0.0.1', 8000)
            httpd = HTTPServer(server_address, AuthCallbackHandler)

            # Run server in background thread
            server_thread = threading.Thread(target=httpd.handle_request)
            server_thread.start()

            # Open browser for login
            if auto_open_browser:
                webbrowser.open(auth_url)
                logger.info("Browser opened for Fyers login")
            else:
                print(f"\nPlease open this URL in your browser:\n{auth_url}\n")

            # Wait for callback
            server_thread.join(timeout=120)

            if AuthCallbackHandler.auth_code is None:
                logger.error("Did not receive auth code")
                return False

            auth_code = AuthCallbackHandler.auth_code
            AuthCallbackHandler.auth_code = None  # Reset for next time

            # Step 3: Exchange auth code for access token
            session.set_token(auth_code)
            response = session.generate_token()

            if 'access_token' in response:
                self.access_token = response['access_token']
                self._save_token(self.access_token)
                self._init_fyers()
                logger.info("Authentication successful")
                return True
            else:
                logger.error(f"Token generation failed: {response}")
                return False

        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False

    def is_authenticated(self) -> bool:
        """Check if client is authenticated."""
        return self.fyers is not None and self.access_token is not None

    def get_profile(self) -> Optional[Dict]:
        """Get user profile."""
        if not self.is_authenticated():
            logger.error("Not authenticated")
            return None

        try:
            response = self.fyers.get_profile()
            if response.get('s') == 'ok':
                return response.get('data')
            else:
                logger.error(f"Profile fetch failed: {response}")
                return None
        except Exception as e:
            logger.error(f"Profile error: {e}")
            return None

    def get_funds(self) -> Optional[Dict]:
        """Get available funds."""
        if not self.is_authenticated():
            return None

        try:
            response = self.fyers.funds()
            if response.get('s') == 'ok':
                return response.get('fund_limit', [])
            return None
        except Exception as e:
            logger.error(f"Funds error: {e}")
            return None

    def get_historical_data(self, symbol: str, resolution: str,
                           from_date: datetime, to_date: datetime) -> Optional[Dict]:
        """
        Get historical OHLCV data.

        Args:
            symbol: Trading symbol (e.g., NSE:NIFTY50-INDEX)
            resolution: Timeframe (1, 5, 15, 30, 60, D, W, M)
            from_date: Start date
            to_date: End date

        Returns:
            Dictionary with candle data
        """
        if not self.is_authenticated():
            return None

        try:
            data = {
                "symbol": symbol,
                "resolution": resolution,
                "date_format": "1",
                "range_from": int(from_date.timestamp()),
                "range_to": int(to_date.timestamp()),
                "cont_flag": "1"
            }

            response = self.fyers.history(data=data)

            if response.get('s') == 'ok':
                return response.get('candles', [])
            else:
                logger.error(f"Historical data failed: {response}")
                return None

        except Exception as e:
            logger.error(f"Historical data error: {e}")
            return None

    def get_quotes(self, symbols: List[str]) -> Optional[Dict]:
        """
        Get current quotes for symbols.

        Args:
            symbols: List of symbols

        Returns:
            Quote data
        """
        if not self.is_authenticated():
            return None

        try:
            data = {"symbols": ",".join(symbols)}
            response = self.fyers.quotes(data=data)

            if response.get('s') == 'ok':
                return response.get('d', [])
            return None
        except Exception as e:
            logger.error(f"Quotes error: {e}")
            return None

    def get_option_chain(self, symbol: str, strike_count: int = 10) -> Optional[Dict]:
        """
        Get option chain data.

        Args:
            symbol: Underlying symbol
            strike_count: Number of strikes around ATM

        Returns:
            Option chain data
        """
        if not self.is_authenticated():
            return None

        try:
            data = {
                "symbol": symbol,
                "strikecount": strike_count,
                "timestamp": ""
            }
            response = self.fyers.optionchain(data=data)

            if response.get('s') == 'ok':
                return response.get('data', {})
            return None
        except Exception as e:
            logger.error(f"Option chain error: {e}")
            return None

    def place_order(self, symbol: str, qty: int, side: int,
                   order_type: int = 2, product_type: str = "INTRADAY",
                   limit_price: float = 0, stop_price: float = 0,
                   disclosed_qty: int = 0, validity: str = "DAY",
                   offline_order: bool = False,
                   stop_loss: float = 0, take_profit: float = 0) -> Optional[Dict]:
        """
        Place an order.

        Args:
            symbol: Trading symbol (e.g., NSE:NIFTY24FEB48000CE)
            qty: Quantity
            side: 1 for Buy, -1 for Sell
            order_type: 1=Limit, 2=Market, 3=Stop, 4=Stop-Limit
            product_type: INTRADAY, CNC, MARGIN, BO, CO
            limit_price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            disclosed_qty: Disclosed quantity
            validity: DAY, IOC
            offline_order: True for AMO
            stop_loss: Stop loss points (for BO/CO)
            take_profit: Take profit points (for BO/CO)

        Returns:
            Order response with order_id
        """
        if not self.is_authenticated():
            return None

        try:
            data = {
                "symbol": symbol,
                "qty": qty,
                "type": order_type,
                "side": side,
                "productType": product_type,
                "limitPrice": limit_price,
                "stopPrice": stop_price,
                "validity": validity,
                "disclosedQty": disclosed_qty,
                "offlineOrder": offline_order,
                "stopLoss": stop_loss,
                "takeProfit": take_profit
            }

            response = self.fyers.place_order(data=data)
            logger.info(f"Order response: {response}")

            if response.get('s') == 'ok':
                return response
            else:
                logger.error(f"Order failed: {response}")
                return None

        except Exception as e:
            logger.error(f"Order error: {e}")
            return None

    def modify_order(self, order_id: str, qty: int = None,
                    order_type: int = None, limit_price: float = None,
                    stop_price: float = None) -> Optional[Dict]:
        """
        Modify an existing order.

        Args:
            order_id: Order ID to modify
            qty: New quantity (optional)
            order_type: New order type (optional)
            limit_price: New limit price (optional)
            stop_price: New stop price (optional)

        Returns:
            Modification response
        """
        if not self.is_authenticated():
            return None

        try:
            data = {"id": order_id}

            if qty is not None:
                data["qty"] = qty
            if order_type is not None:
                data["type"] = order_type
            if limit_price is not None:
                data["limitPrice"] = limit_price
            if stop_price is not None:
                data["stopPrice"] = stop_price

            response = self.fyers.modify_order(data=data)

            if response.get('s') == 'ok':
                return response
            return None
        except Exception as e:
            logger.error(f"Modify order error: {e}")
            return None

    def cancel_order(self, order_id: str) -> Optional[Dict]:
        """Cancel an order."""
        if not self.is_authenticated():
            return None

        try:
            data = {"id": order_id}
            response = self.fyers.cancel_order(data=data)

            if response.get('s') == 'ok':
                return response
            return None
        except Exception as e:
            logger.error(f"Cancel order error: {e}")
            return None

    def get_orders(self) -> Optional[List]:
        """Get all orders for the day."""
        if not self.is_authenticated():
            return None

        try:
            response = self.fyers.orderbook()
            if response.get('s') == 'ok':
                return response.get('orderBook', [])
            return None
        except Exception as e:
            logger.error(f"Orders error: {e}")
            return None

    def get_positions(self) -> Optional[List]:
        """Get all open positions."""
        if not self.is_authenticated():
            return None

        try:
            response = self.fyers.positions()
            if response.get('s') == 'ok':
                return response.get('netPositions', [])
            return None
        except Exception as e:
            logger.error(f"Positions error: {e}")
            return None

    def get_trades(self) -> Optional[List]:
        """Get all trades for the day."""
        if not self.is_authenticated():
            return None

        try:
            response = self.fyers.tradebook()
            if response.get('s') == 'ok':
                return response.get('tradeBook', [])
            return None
        except Exception as e:
            logger.error(f"Trades error: {e}")
            return None

    def exit_all_positions(self) -> bool:
        """
        Exit all open positions (market order).

        Returns:
            True if successful
        """
        if not self.is_authenticated():
            return False

        try:
            response = self.fyers.exit_positions()
            if response.get('s') == 'ok':
                logger.info("All positions exited")
                return True
            return False
        except Exception as e:
            logger.error(f"Exit positions error: {e}")
            return False

    def get_market_status(self) -> Optional[Dict]:
        """Get market status."""
        if not self.is_authenticated():
            return None

        try:
            response = self.fyers.market_status()
            if response.get('s') == 'ok':
                return response.get('marketStatus', [])
            return None
        except Exception as e:
            logger.error(f"Market status error: {e}")
            return None

    def build_option_symbol(self, underlying: str, expiry_date: str,
                           strike: float, option_type: str) -> str:
        """
        Build Fyers option symbol.

        Args:
            underlying: NIFTY or BANKNIFTY
            expiry_date: Expiry in YYMMDD format (e.g., 240215)
            strike: Strike price
            option_type: CE or PE

        Returns:
            Full option symbol (e.g., NSE:NIFTY24FEB48000CE)
        """
        # Format: NSE:NIFTY{YY}{MON}{STRIKE}{CE/PE}
        # For weekly: NSE:NIFTY{YY}{M}{DD}{STRIKE}{CE/PE}

        # Parse expiry
        expiry = datetime.strptime(expiry_date, '%y%m%d')
        month_codes = {
            1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR',
            5: 'MAY', 6: 'JUN', 7: 'JUL', 8: 'AUG',
            9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'
        }

        # Weekly expiry format: YY + M (single char) + DD
        week_month_codes = {
            1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6',
            7: '7', 8: '8', 9: '9', 10: 'O', 11: 'N', 12: 'D'
        }

        # Check if it's monthly expiry (last Thursday of month)
        # For simplicity, use weekly format for now
        yy = expiry.strftime('%y')
        month_code = week_month_codes[expiry.month]
        dd = expiry.strftime('%d')

        strike_str = str(int(strike))

        symbol = f"NSE:{underlying}{yy}{month_code}{dd}{strike_str}{option_type}"

        return symbol

    def get_current_expiry(self, weekly: bool = True) -> str:
        """
        Get current or next expiry date.

        Args:
            weekly: True for weekly, False for monthly

        Returns:
            Expiry date in YYMMDD format
        """
        today = datetime.now()

        # Find next Thursday (Nifty weekly expiry)
        days_until_thursday = (3 - today.weekday()) % 7
        if days_until_thursday == 0 and today.hour >= 15:
            # If it's Thursday after market close, get next week
            days_until_thursday = 7

        next_expiry = today + timedelta(days=days_until_thursday)

        return next_expiry.strftime('%y%m%d')

    def start_websocket(self, symbols: List[str],
                       on_message_callback, on_error_callback=None,
                       on_close_callback=None):
        """
        Start WebSocket for real-time data.

        Args:
            symbols: List of symbols to subscribe
            on_message_callback: Callback for price updates
            on_error_callback: Callback for errors
            on_close_callback: Callback for connection close
        """
        if not FYERS_SDK_AVAILABLE or not self.is_authenticated():
            logger.error("Cannot start WebSocket - not authenticated")
            return

        try:
            self.ws_client = data_ws.FyersDataSocket(
                access_token=self.access_token,
                log_path="",
                litemode=False,
                write_to_file=False,
                reconnect=True,
                on_connect=lambda: logger.info("WebSocket connected"),
                on_close=on_close_callback or (lambda: logger.info("WebSocket closed")),
                on_error=on_error_callback or (lambda e: logger.error(f"WebSocket error: {e}")),
                on_message=on_message_callback
            )

            # Connect and subscribe
            self.ws_client.connect()
            self.ws_client.subscribe(symbols=symbols, data_type="SymbolUpdate")

            logger.info(f"Subscribed to WebSocket: {symbols}")

        except Exception as e:
            logger.error(f"WebSocket error: {e}")

    def stop_websocket(self):
        """Stop WebSocket connection."""
        if self.ws_client:
            try:
                self.ws_client.close_connection()
                logger.info("WebSocket stopped")
            except Exception as e:
                logger.error(f"WebSocket close error: {e}")
