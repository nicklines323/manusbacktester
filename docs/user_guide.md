# docs/user_guide.md

# Crypto Algotrading Platform User Guide

## Introduction

Welcome to the Crypto Algotrading Platform! This platform allows you to backtest and deploy the envelope trading strategy for cryptocurrency trading. This user guide will help you understand how to use the platform effectively.

## Getting Started

### Installation

Please refer to the installation.md document for detailed installation instructions.

### Dashboard Overview

The platform consists of four main tabs:

1. **Dashboard**: Provides an overview of your trading performance
2. **Backtesting**: Allows you to test the envelope strategy with historical data
3. **Live Trading**: Enables you to deploy the strategy for real-time trading
4. **Settings**: Provides options to configure exchange connections and platform preferences

## Envelope Trading Strategy

The envelope trading strategy is based on creating bands (envelopes) around a moving average:

- When price crosses below the lower band, a buy signal is generated
- When price crosses above the upper band, a sell signal is generated
- The strategy can be configured with different types of moving averages and envelope percentages

### Strategy Parameters

- **Average Type**: The type of moving average to use (SMA, EMA, WMA, Donchian)
- **Average Period**: The period for calculating the moving average
- **Envelope Percentage**: The percentage deviation from the average to create the bands
- **Trading Mode**: Whether to trade long only, short only, or both
- **Stop Loss**: The percentage at which to trigger a stop loss

## Backtesting

The backtesting tab allows you to evaluate the strategy's performance with historical data:

1. Select a trading pair (e.g., BTC/USDT)
2. Choose a timeframe (e.g., 1h, 4h, 1d)
3. Set the date range for the backtest
4. Configure strategy parameters
5. Click "Run Backtest" to execute

### Interpreting Results

The backtest results include:

- **Total Return**: The overall percentage return
- **Sharpe Ratio**: Risk-adjusted return metric
- **Max Drawdown**: Maximum percentage decline from a peak
- **Win Rate**: Percentage of profitable trades
- **Equity Curve**: Chart showing balance over time
- **Drawdown Chart**: Visualization of drawdowns
- **Trade Analysis**: Breakdown of individual trades

## Live Trading

The live trading tab allows you to deploy the strategy in real-time:

1. Configure the strategy parameters
2. Set position sizing
3. Click "Start Trading" to begin

### Monitoring Trades

The platform provides real-time monitoring of:

- Current price and position
- Unrealized profit/loss
- Trade history
- Price chart with strategy indicators

### Risk Management

The platform includes several risk management features:

- Stop loss orders
- Position sizing based on account balance
- Trading mode selection (long only, short only, both)

## Settings

The settings tab allows you to configure:

- Exchange API connections
- Notification preferences
- Platform appearance

## Troubleshooting

If you encounter issues:

1. Check the logs for error messages
2. Verify exchange API credentials
3. Ensure internet connectivity
4. Restart the application if necessary

## Support

For additional support, please refer to the GitHub repository or contact the development team.
