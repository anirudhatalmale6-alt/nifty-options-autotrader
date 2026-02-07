"""
Strategy Optimizer
Tests multiple parameter combinations to find optimal settings
Target: 6-7% monthly returns, <20% drawdown
"""

import sys
import yaml
import copy
from pathlib import Path
from datetime import datetime
from itertools import product

sys.path.insert(0, str(Path(__file__).parent))

from backtest.engine import BacktestEngine


def run_optimization():
    """Run parameter optimization."""

    # Load base config
    with open('config/settings.yaml', 'r') as f:
        base_config = yaml.safe_load(f)

    # Parameters to test
    param_grid = {
        'rsi_bullish_range': [(40, 80), (45, 75), (35, 85)],
        'rsi_bearish_range': [(20, 60), (25, 55), (15, 65)],
        'trading_days': [
            {'monday': False, 'tuesday': True, 'wednesday': True, 'thursday': True, 'friday': False},
            {'monday': True, 'tuesday': True, 'wednesday': True, 'thursday': True, 'friday': False},
            {'monday': True, 'tuesday': True, 'wednesday': True, 'thursday': True, 'friday': True},
        ],
        'entry_end': ['10:30', '12:00', '14:00'],
        'max_trades_per_day': [2, 3, 5],
        'risk_per_trade': [2.0, 3.0, 4.0],
        'stop_loss_percent': [25, 33, 40],
        'target_percent': [75, 100, 150],
    }

    results = []

    # Test key combinations (not all - that would be too many)
    test_configs = [
        # Config 1: Aggressive - wider RSI, all days, longer window
        {
            'name': 'Aggressive',
            'rsi_bullish': (35, 85),
            'rsi_bearish': (15, 65),
            'days': {'monday': True, 'tuesday': True, 'wednesday': True, 'thursday': True, 'friday': True},
            'entry_end': '14:00',
            'max_trades': 5,
            'risk': 3.0,
            'sl': 30,
            'target': 90
        },
        # Config 2: Moderate - relaxed RSI, weekdays, extended window
        {
            'name': 'Moderate',
            'rsi_bullish': (40, 80),
            'rsi_bearish': (20, 60),
            'days': {'monday': True, 'tuesday': True, 'wednesday': True, 'thursday': True, 'friday': False},
            'entry_end': '12:00',
            'max_trades': 3,
            'risk': 2.5,
            'sl': 33,
            'target': 100
        },
        # Config 3: Balanced - moderate settings
        {
            'name': 'Balanced',
            'rsi_bullish': (45, 75),
            'rsi_bearish': (25, 55),
            'days': {'monday': True, 'tuesday': True, 'wednesday': True, 'thursday': True, 'friday': True},
            'entry_end': '13:00',
            'max_trades': 4,
            'risk': 3.0,
            'sl': 25,
            'target': 75
        },
        # Config 4: High frequency - very relaxed
        {
            'name': 'High Frequency',
            'rsi_bullish': (30, 90),
            'rsi_bearish': (10, 70),
            'days': {'monday': True, 'tuesday': True, 'wednesday': True, 'thursday': True, 'friday': True},
            'entry_end': '14:30',
            'max_trades': 6,
            'risk': 2.0,
            'sl': 25,
            'target': 75
        },
        # Config 5: Scalping - quick trades
        {
            'name': 'Scalping',
            'rsi_bullish': (40, 80),
            'rsi_bearish': (20, 60),
            'days': {'monday': True, 'tuesday': True, 'wednesday': True, 'thursday': True, 'friday': True},
            'entry_end': '14:00',
            'max_trades': 8,
            'risk': 2.0,
            'sl': 20,
            'target': 50
        },
        # Config 6: Trend following - EMA focused
        {
            'name': 'Trend Following',
            'rsi_bullish': (45, 80),
            'rsi_bearish': (20, 55),
            'days': {'monday': True, 'tuesday': True, 'wednesday': True, 'thursday': True, 'friday': True},
            'entry_end': '13:00',
            'max_trades': 4,
            'risk': 3.5,
            'sl': 30,
            'target': 90
        },
    ]

    print("=" * 70)
    print("STRATEGY OPTIMIZATION")
    print("Target: 6-7% monthly returns, <20% max drawdown")
    print("=" * 70)

    for tc in test_configs:
        config = copy.deepcopy(base_config)

        # Apply test config
        config['daily_conditions']['rsi_bullish_min'] = tc['rsi_bullish'][0]
        config['daily_conditions']['rsi_bullish_max'] = tc['rsi_bullish'][1]
        config['daily_conditions']['rsi_bearish_min'] = tc['rsi_bearish'][0]
        config['daily_conditions']['rsi_bearish_max'] = tc['rsi_bearish'][1]
        config['trading_days'] = tc['days']
        config['trading_hours']['entry_end'] = tc['entry_end']
        config['limits']['max_trades_per_day'] = tc['max_trades']
        config['capital']['risk_per_trade_percent'] = tc['risk']
        config['risk_reward']['stop_loss_percent'] = tc['sl']
        config['risk_reward']['target_percent'] = tc['target']

        # Run backtest
        print(f"\nTesting: {tc['name']}...")
        engine = BacktestEngine(config)
        result = engine.run_backtest()

        # Calculate monthly returns
        months = len(result.monthly_returns) if result.monthly_returns else 1
        avg_monthly = result.roi_percent / max(months, 1) if months > 0 else 0

        # Calculate actual monthly avg from data
        if result.monthly_returns:
            monthly_vals = list(result.monthly_returns.values())
            avg_monthly = sum(monthly_vals) / len(monthly_vals)

        results.append({
            'name': tc['name'],
            'trades': result.total_trades,
            'win_rate': result.win_rate,
            'roi': result.roi_percent,
            'avg_monthly': avg_monthly,
            'max_dd': result.max_drawdown_percent,
            'profit_factor': result.profit_factor,
            'config': tc
        })

        print(f"  Trades: {result.total_trades}, Win Rate: {result.win_rate:.1f}%")
        print(f"  ROI: {result.roi_percent:.2f}%, Avg Monthly: {avg_monthly:.2f}%")
        print(f"  Max Drawdown: {result.max_drawdown_percent:.1f}%, PF: {result.profit_factor:.2f}")

    # Sort by average monthly return
    results.sort(key=lambda x: x['avg_monthly'], reverse=True)

    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS (Sorted by Avg Monthly Return)")
    print("=" * 70)
    print(f"{'Config':<20} {'Trades':>8} {'Win%':>8} {'ROI%':>10} {'Monthly%':>10} {'MaxDD%':>8} {'PF':>6}")
    print("-" * 70)

    for r in results:
        meets_target = r['avg_monthly'] >= 5 and r['max_dd'] < 20
        marker = " ✓" if meets_target else ""
        print(f"{r['name']:<20} {r['trades']:>8} {r['win_rate']:>7.1f}% {r['roi']:>9.2f}% {r['avg_monthly']:>9.2f}% {r['max_dd']:>7.1f}% {r['profit_factor']:>5.2f}{marker}")

    # Find best config that meets criteria
    best = None
    for r in results:
        if r['max_dd'] < 20 and r['avg_monthly'] >= 5:
            best = r
            break

    if best:
        print(f"\n✓ BEST CONFIG: {best['name']}")
        print(f"  Average Monthly Return: {best['avg_monthly']:.2f}%")
        print(f"  Max Drawdown: {best['max_dd']:.1f}%")
        print(f"  Settings: {best['config']}")
    else:
        # Pick the one with highest returns under 20% DD
        valid = [r for r in results if r['max_dd'] < 20]
        if valid:
            best = max(valid, key=lambda x: x['avg_monthly'])
            print(f"\n⚠ CLOSEST TO TARGET: {best['name']}")
            print(f"  Average Monthly Return: {best['avg_monthly']:.2f}%")
            print(f"  Max Drawdown: {best['max_dd']:.1f}%")
        else:
            print("\n⚠ No config meets both criteria. Consider further tuning.")

    return results, best


if __name__ == '__main__':
    results, best = run_optimization()
