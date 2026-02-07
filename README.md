# Nifty Options Auto-Trader

Automated trading system for Nifty Options using the Trend Momentum Strategy with Fyers API integration.

## Features

- **Fully Automated Trading**: Analyzes market, places orders, manages positions automatically
- **Trend Momentum Strategy**: Uses EMA crossovers, RSI, MACD, and volume for signal generation
- **Risk Management**: Fixed 1:3 risk-reward, trailing stop-loss, position sizing based on % risk
- **Backtesting Engine**: Test the strategy on 3 years of historical data
- **Trade Journal**: Complete logging of all trades and daily summaries
- **Paper Trading Mode**: Test without real money

## Strategy Overview

### Entry Conditions

**CALL Options (Bullish)**:
1. Nifty > 20 EMA > 50 EMA on Daily chart
2. RSI between 50-70
3. MACD line above signal line
4. Volume above 20-day average
5. Break above previous day high OR first 15-min candle high

**PUT Options (Bearish)**:
1. Nifty < 20 EMA < 50 EMA on Daily chart
2. RSI between 30-50
3. MACD line below signal line
4. Volume above 20-day average
5. Break below previous day low OR first 15-min candle low

### Exit Rules

- **Stop Loss**: 33% below entry premium
- **Target**: 100% above entry premium (1:3 R:R)
- **Trailing SL**: Move to breakeven at 50% profit, lock 50% at 75% profit
- **Partial Booking**: Exit 50% at target, trail remaining with 20-point SL
- **Time Exit**: Exit all positions at 3:00 PM if no movement

### Trading Rules

- Trade only Tuesday, Wednesday, Thursday
- Entry window: 9:30 AM - 10:30 AM
- Maximum 2 trades per day
- Stop after 3 consecutive losses
- Skip high VIX (>20) days

## Installation

```bash
# Clone/Download the project
cd nifty_trader

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Edit `config/settings.yaml`:

```yaml
# Fyers API Credentials
fyers:
  app_id: "YOUR_APP_ID"
  secret_key: "YOUR_SECRET_KEY"
  redirect_uri: "http://127.0.0.1:8000/"

# Capital Settings
capital:
  total_capital: 200000
  risk_per_trade_percent: 2.0
  max_lots: 3

# System Mode
system:
  mode: "paper"  # "paper" or "live"
```

## Usage

### 1. First-time Setup - Authenticate with Fyers

```bash
python main.py auth
```

This opens a browser for Fyers login. Complete the login to get access token.

### 2. Run Backtest

Test the strategy on historical data:

```bash
python main.py backtest --output results.csv
```

### 3. Paper Trading (Recommended First)

Run without real money to test:

```bash
python main.py run --paper
```

### 4. Live Trading

**CAUTION**: Only use after thorough testing!

```bash
# First, change config/settings.yaml:
# system:
#   mode: "live"

python main.py run
```

### 5. Check Status

```bash
python main.py status
```

## Project Structure

```
nifty_trader/
├── main.py              # Main entry point and CLI
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── config/
│   └── settings.yaml   # Configuration file
├── core/
│   ├── indicators.py   # Technical indicators (EMA, RSI, MACD)
│   ├── strategy.py     # Trading strategy logic
│   └── logger.py       # Logging and trade journal
├── api/
│   └── fyers_client.py # Fyers API integration
├── backtest/
│   └── engine.py       # Backtesting engine
├── logs/               # Log files
└── data/               # Market data cache
```

## Fyers API Setup

1. Go to https://myapi.fyers.in/dashboard
2. Login with your Fyers credentials
3. Create a new app:
   - App Name: NiftyTrader
   - Redirect URL: http://127.0.0.1:8000/
   - App Type: Trading
4. Note down App ID and Secret Key
5. Add them to `config/settings.yaml`

## Risk Warning

⚠️ **IMPORTANT DISCLAIMERS**:

1. Trading involves substantial risk of loss
2. Past backtest performance does NOT guarantee future results
3. Options trading can result in 100% loss of capital
4. Never trade with money you cannot afford to lose
5. This software is provided as-is with no guarantees
6. Always paper trade first for at least 1-2 months
7. The developer is not responsible for any trading losses

## Troubleshooting

### Authentication Issues
- Ensure App ID and Secret Key are correct
- Check redirect URL matches exactly
- Try clearing browser cookies and re-authenticating

### No Trades Being Taken
- Check if it's a valid trading day (Tue-Wed-Thu)
- Verify trading time window (9:30 AM - 10:30 AM)
- Check if daily trade limit reached
- Verify market conditions meet all entry criteria

### Connection Lost
- The system auto-reconnects for WebSocket
- All positions are exited on connection loss (configurable)

## Support

For issues or questions, please check the logs in `logs/` directory first.

## License

For personal use only. Not for redistribution.
