# Nifty Options Auto-Trader

Automated trading system for Nifty Options using Fyers API.

## Quick Start (Windows)

### Step 1: Install Python
1. Download Python 3.11 from: https://www.python.org/downloads/
2. During installation, CHECK the box **"Add Python to PATH"**
3. Click "Install Now"

### Step 2: Extract the ZIP
Extract this folder to a location like `C:\TradingBot\nifty_trader`

### Step 3: Open Command Prompt
1. Press `Windows + R`
2. Type `cmd` and press Enter

### Step 4: Navigate to Bot Folder
```
cd C:\TradingBot\nifty_trader
```
(Use the actual path where you extracted)

### Step 5: Install Dependencies
```
pip install -r requirements.txt
```

### Step 6: Run the Bot
```
python main.py run
```

**First time:** Browser will open for Fyers login. Login with your Fyers account. After login, trading starts automatically.

---

## Commands

| Command | Description |
|---------|-------------|
| `python main.py run` | Start trading (paper/live based on config) |
| `python main.py backtest` | Run backtest on historical data |
| `python main.py auth` | Re-authenticate with Fyers |

---

## Configuration

Edit `config/settings.yaml` to change settings:

### Trading Mode
```yaml
system:
  mode: "paper"  # Change to "live" for real trading
```

### Capital Settings
```yaml
capital:
  total_capital: 200000  # Your trading capital
  risk_per_trade_percent: 5.0  # Risk per trade
```

### Trading Hours
```yaml
trading_hours:
  entry_start: "09:20"
  entry_end: "14:30"
  time_based_exit: "15:15"
```

---

## Strategy Overview

**Entry Conditions:**
- Daily trend: EMA 20 crosses above EMA 50 (bullish) or below (bearish)
- RSI between 35-85 (bullish) or 15-65 (bearish)
- MACD confirms direction
- 15-min candle breakout trigger

**Exit Conditions:**
- Stop Loss: 25% of premium
- Target: 50% profit (1:2 risk-reward)
- Trailing stop activates at 25% profit
- Time-based exit at 3:15 PM

**Risk Management:**
- Max 5% risk per trade
- Max 6 trades per day
- Stops after 4 consecutive losses

---

## Paper Trading Mode (Default)

Default mode is **paper** - no real orders are placed. Bot simulates trades and shows P&L.

**To switch to LIVE trading:**
1. Open `config/settings.yaml`
2. Change `mode: "paper"` to `mode: "live"`
3. Save and restart bot

---

## Logs

Check `logs/` folder for:
- Daily trade logs
- Error logs
- Performance reports

---

## Troubleshooting

### "Python not found"
- Reinstall Python and CHECK "Add Python to PATH"

### "Module not found"
- Run: `pip install -r requirements.txt`

### Browser doesn't open for login
- Run: `python main.py auth`

### No trades happening
- Check if market is open
- Check trading hours in config
- Check logs for errors

---

## Daily Usage

1. Open Command Prompt
2. Navigate to bot folder: `cd C:\TradingBot\nifty_trader`
3. Start bot: `python main.py run`
4. First time each day: Login to Fyers when browser opens
5. Bot trades automatically during market hours
6. Press `Ctrl+C` to stop

---

## Risk Warning

- Trading involves substantial risk of loss
- Past backtest results do NOT guarantee future returns
- Options can result in 100% loss
- Never trade money you cannot afford to lose
- Always test in paper mode first

---

## Support

If you face any issues, contact me on Freelancer.
