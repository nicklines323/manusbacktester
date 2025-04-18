# src/ui/dashboard.py

import logging
import os
import json
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta

import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from ..data.data_manager import DataManager
from ..strategy.strategy_manager import StrategyManager
from ..backtesting.backtest_manager import BacktestManager
from ..trading.trading_engine import TradingEngine
from .styles import COLORS, STYLES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Dashboard:
    """
    Web-based dashboard for the crypto algotrading platform.
    
    This class:
    - Creates a Dash web application
    - Provides UI for configuring and monitoring trading strategies
    - Visualizes backtest results and live trading performance
    """
    
    def __init__(self, 
                data_manager=None, 
                strategy_manager=None, 
                backtest_manager=None,
                trading_engine=None):
        """
        Initialize the Dashboard.
        
        Args:
            data_manager: Data manager instance
            strategy_manager: Strategy manager instance
            backtest_manager: Backtest manager instance
            trading_engine: Trading engine instance
        """
        # Initialize components
        self.data_manager = data_manager or DataManager()
        self.strategy_manager = strategy_manager or StrategyManager()
        self.backtest_manager = backtest_manager or BacktestManager(
            strategy_manager=self.strategy_manager,
            data_manager=self.data_manager
        )
        self.trading_engine = trading_engine or TradingEngine(
            strategy_manager=self.strategy_manager,
            data_manager=self.data_manager
        )
        
        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY],
            suppress_callback_exceptions=True
        )
        
        # Set app title
        self.app.title = 'Crypto Algotrading Platform'
        
        # Create layout
        self.app.layout = self._create_layout()
        
        # Register callbacks
        self._register_callbacks()
        
        logger.info("Initialized Dashboard")
    
    def _create_layout(self):
        """
        Create the dashboard layout.
        
        Returns:
            dash.html.Div: Dashboard layout
        """
        # Create navbar
        navbar = dbc.Navbar(
            dbc.Container(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Img(
                                    src='/assets/logo.png',
                                    height="30px",
                                    className="me-2"
                                ),
                                width="auto"
                            ),
                            dbc.Col(
                                dbc.NavbarBrand("Crypto Algotrading Platform", className="ms-2"),
                                width="auto"
                            )
                        ],
                        align="center",
                        className="g-0"
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.NavbarToggler(id="navbar-toggler"),
                                    dbc.Collapse(
                                        dbc.Nav(
                                            [
                                                dbc.NavItem(dbc.NavLink("Dashboard", href="#")),
                                                dbc.NavItem(dbc.NavLink("Backtesting", href="#")),
                                                dbc.NavItem(dbc.NavLink("Live Trading", href="#")),
                                                dbc.NavItem(dbc.NavLink("Settings", href="#"))
                                            ],
                                            className="ms-auto",
                                            navbar=True
                                        ),
                                        id="navbar-collapse",
                                        navbar=True
                                    )
                                ]
                            )
                        ],
                        align="center"
                    )
                ],
                fluid=True
            ),
            color="dark",
            dark=True
        )
        
        # Create tabs
        tabs = dbc.Tabs(
            [
                dbc.Tab(self._create_dashboard_tab(), label="Dashboard", tab_id="tab-dashboard"),
                dbc.Tab(self._create_backtesting_tab(), label="Backtesting", tab_id="tab-backtesting"),
                dbc.Tab(self._create_live_trading_tab(), label="Live Trading", tab_id="tab-live-trading"),
                dbc.Tab(self._create_settings_tab(), label="Settings", tab_id="tab-settings")
            ],
            id="tabs",
            active_tab="tab-dashboard"
        )
        
        # Create footer
        footer = html.Footer(
            dbc.Container(
                [
                    html.Hr(),
                    html.P(
                        [
                            "Crypto Algotrading Platform © 2025 | ",
                            html.A("Documentation", href="#"),
                            " | ",
                            html.A("GitHub", href="#")
                        ],
                        className="text-center text-muted"
                    )
                ],
                fluid=True
            )
        )
        
        # Create layout
        layout = html.Div(
            [
                navbar,
                dbc.Container(
                    [
                        html.Div(id="page-content", children=[tabs]),
                        dcc.Store(id="backtest-results"),
                        dcc.Store(id="live-trading-state"),
                        dcc.Interval(
                            id="interval-component",
                            interval=5 * 1000,  # 5 seconds
                            n_intervals=0
                        )
                    ],
                    fluid=True,
                    className="mt-4"
                ),
                footer
            ]
        )
        
        return layout
    
    def _create_dashboard_tab(self):
        """
        Create the dashboard tab.
        
        Returns:
            dash.html.Div: Dashboard tab layout
        """
        # Create cards
        balance_card = dbc.Card(
            [
                dbc.CardHeader("Account Balance"),
                dbc.CardBody(
                    [
                        html.H3(id="balance-value", children="$10,000.00"),
                        html.P(id="balance-change", children="0.00%", className="text-success")
                    ]
                )
            ],
            className="mb-4"
        )
        
        active_trades_card = dbc.Card(
            [
                dbc.CardHeader("Active Trades"),
                dbc.CardBody(
                    [
                        html.H3(id="active-trades-value", children="0"),
                        html.P("Open positions", className="text-muted")
                    ]
                )
            ],
            className="mb-4"
        )
        
        profit_card = dbc.Card(
            [
                dbc.CardHeader("Total Profit"),
                dbc.CardBody(
                    [
                        html.H3(id="profit-value", children="$0.00"),
                        html.P(id="profit-change", children="0.00%", className="text-success")
                    ]
                )
            ],
            className="mb-4"
        )
        
        win_rate_card = dbc.Card(
            [
                dbc.CardHeader("Win Rate"),
                dbc.CardBody(
                    [
                        html.H3(id="win-rate-value", children="0.00%"),
                        html.P(id="win-rate-trades", children="0 trades", className="text-muted")
                    ]
                )
            ],
            className="mb-4"
        )
        
        # Create charts
        equity_chart = dbc.Card(
            [
                dbc.CardHeader("Equity Curve"),
                dbc.CardBody(
                    [
                        dcc.Graph(
                            id="equity-chart",
                            figure=go.Figure(
                                layout=dict(
                                    template="plotly_dark",
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    margin=dict(l=10, r=10, t=10, b=10),
                                    height=300
                                )
                            )
                        )
                    ]
                )
            ],
            className="mb-4"
        )
        
        trades_chart = dbc.Card(
            [
                dbc.CardHeader("Recent Trades"),
                dbc.CardBody(
                    [
                        dcc.Graph(
                            id="trades-chart",
                            figure=go.Figure(
                                layout=dict(
                                    template="plotly_dark",
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    margin=dict(l=10, r=10, t=10, b=10),
                                    height=300
                                )
                            )
                        )
                    ]
                )
            ],
            className="mb-4"
        )
        
        # Create recent trades table
        recent_trades_table = dbc.Card(
            [
                dbc.CardHeader("Recent Trades"),
                dbc.CardBody(
                    [
                        html.Div(id="recent-trades-table")
                    ]
                )
            ],
            className="mb-4"
        )
        
        # Create layout
        layout = html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(balance_card, width=3),
                        dbc.Col(active_trades_card, width=3),
                        dbc.Col(profit_card, width=3),
                        dbc.Col(win_rate_card, width=3)
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(equity_chart, width=6),
                        dbc.Col(trades_chart, width=6)
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(recent_trades_table, width=12)
                    ]
                )
            ]
        )
        
        return layout
    
    def _create_backtesting_tab(self):
        """
        Create the backtesting tab.
        
        Returns:
            dash.html.Div: Backtesting tab layout
        """
        # Create strategy configuration card
        strategy_config_card = dbc.Card(
            [
                dbc.CardHeader("Strategy Configuration"),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Trading Pair"),
                                        dcc.Dropdown(
                                            id="backtest-pair-dropdown",
                                            options=[
                                                {"label": "BTC/USDT", "value": "BTC/USDT"},
                                                {"label": "ETH/USDT", "value": "ETH/USDT"},
                                                {"label": "BNB/USDT", "value": "BNB/USDT"},
                                                {"label": "SOL/USDT", "value": "SOL/USDT"},
                                                {"label": "ADA/USDT", "value": "ADA/USDT"}
                                            ],
                                            value="BTC/USDT",
                                            clearable=False
                                        )
                                    ],
                                    width=6
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Timeframe"),
                                        dcc.Dropdown(
                                            id="backtest-timeframe-dropdown",
                                            options=[
                                                {"label": "1 minute", "value": "1m"},
                                                {"label": "5 minutes", "value": "5m"},
                                                {"label": "15 minutes", "value": "15m"},
                                                {"label": "1 hour", "value": "1h"},
                                                {"label": "4 hours", "value": "4h"},
                                                {"label": "1 day", "value": "1d"}
                                            ],
                                            value="1h",
                                            clearable=False
                                        )
                                    ],
                                    width=6
                                )
                            ],
                            className="mb-3"
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Date Range"),
                                        dcc.DatePickerRange(
                                            id="backtest-date-range",
                                            start_date=(datetime.now() - timedelta(days=30)).date(),
                                            end_date=datetime.now().date(),
                                            display_format="YYYY-MM-DD"
                                        )
                                    ],
                                    width=12
                                )
                            ],
                            className="mb-3"
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Average Type"),
                                        dcc.Dropdown(
                                            id="backtest-average-type-dropdown",
                                            options=[
                                                {"label": "Simple Moving Average (SMA)", "value": "sma"},
                                                {"label": "Exponential Moving Average (EMA)", "value": "ema"},
                                                {"label": "Weighted Moving Average (WMA)", "value": "wma"},
                                                {"label": "Donchian Channel Middle (DCM)", "value": "dcm"}
                                            ],
                                            value="sma",
                                            clearable=False
                                        )
                                    ],
                                    width=6
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Average Period"),
                                        dcc.Slider(
                                            id="backtest-average-period-slider",
                                            min=5,
                                            max=50,
                                            step=1,
                                            value=20,
                                            marks={i: str(i) for i in range(5, 51, 5)},
                                            tooltip={"placement": "bottom", "always_visible": True}
                                        )
                                    ],
                                    width=6
                                )
                            ],
                            className="mb-3"
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Envelope Percentage"),
                                        dcc.Slider(
                                            id="backtest-envelope-slider",
                                            min=0.1,
                                            max=5.0,
                                            step=0.1,
                                            value=1.0,
                                            marks={i/10: f"{i/10}%" for i in range(1, 51, 5)},
                                            tooltip={"placement": "bottom", "always_visible": True}
                                        )
                                    ],
                                    width=12
                                )
                            ],
                            className="mb-3"
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Trading Mode"),
                                        dcc.Dropdown(
                                            id="backtest-mode-dropdown",
                                            options=[
                                                {"label": "Long Only", "value": "long"},
                                                {"label": "Short Only", "value": "short"},
                                                {"label": "Long & Short", "value": "both"}
                                            ],
                                            value="both",
                                            clearable=False
                                        )
                                    ],
                                    width=6
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Stop Loss (%)"),
                                        dcc.Slider(
                                            id="backtest-stop-loss-slider",
                                            min=0.5,
                                            max=10.0,
                                            step=0.5,
                                            value=3.0,
                                            marks={i/2: f"{i/2}%" for i in range(1, 21, 2)},
                                            tooltip={"placement": "bottom", "always_visible": True}
                                        )
                                    ],
                                    width=6
                                )
                            ],
                            className="mb-3"
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Initial Balance"),
                                        dbc.Input(
                                            id="backtest-initial-balance-input",
                                            type="number",
                                            value=10000,
                                            min=100,
                                            step=100
                                        )
                                    ],
                                    width=6
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Commission (%)"),
                                        dbc.Input(
                                            id="backtest-commission-input",
                                            type="number",
                                            value=0.1,
                                            min=0,
                                            max=1,
                                            step=0.01
                                        )
                                    ],
                                    width=6
                                )
                            ],
                            className="mb-3"
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Button(
                                            "Run Backtest",
                                            id="run-backtest-button",
                                            color="primary",
                                            className="w-100"
                                        )
                                    ],
                                    width=12
                                )
                            ]
                        )
                    ]
                )
            ],
            className="mb-4"
        )
        
        # Create backtest results card
        backtest_results_card = dbc.Card(
            [
                dbc.CardHeader("Backtest Results"),
                dbc.CardBody(
                    [
                        dbc.Spinner(
                            html.Div(id="backtest-results-content")
                        )
                    ]
                )
            ],
            className="mb-4"
        )
        
        # Create layout
        layout = html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(strategy_config_card, width=4),
                        dbc.Col(backtest_results_card, width=8)
                    ]
                )
            ]
        )
        
        return layout
    
    def _create_live_trading_tab(self):
        """
        Create the live trading tab.
        
        Returns:
            dash.html.Div: Live trading tab layout
        """
        # Create strategy configuration card
        strategy_config_card = dbc.Card(
            [
                dbc.CardHeader("Strategy Configuration"),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Trading Pair"),
                                        dcc.Dropdown(
                                            id="live-pair-dropdown",
                                            options=[
                                                {"label": "BTC/USDT", "value": "BTC/USDT"},
                                                {"label": "ETH/USDT", "value": "ETH/USDT"},
                                                {"label": "BNB/USDT", "value": "BNB/USDT"},
                                                {"label": "SOL/USDT", "value": "SOL/USDT"},
                                                {"label": "ADA/USDT", "value": "ADA/USDT"}
                                            ],
                                            value="BTC/USDT",
                                            clearable=False
                                        )
                                    ],
                                    width=6
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Timeframe"),
                                        dcc.Dropdown(
                                            id="live-timeframe-dropdown",
                                            options=[
                                                {"label": "1 minute", "value": "1m"},
                                                {"label": "5 minutes", "value": "5m"},
                                                {"label": "15 minutes", "value": "15m"},
                                                {"label": "1 hour", "value": "1h"},
                                                {"label": "4 hours", "value": "4h"},
                                                {"label": "1 day", "value": "1d"}
                                            ],
                                            value="1h",
                                            clearable=False
                                        )
                                    ],
                                    width=6
                                )
                            ],
                            className="mb-3"
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Average Type"),
                                        dcc.Dropdown(
                                            id="live-average-type-dropdown",
                                            options=[
                                                {"label": "Simple Moving Average (SMA)", "value": "sma"},
                                                {"label": "Exponential Moving Average (EMA)", "value": "ema"},
                                                {"label": "Weighted Moving Average (WMA)", "value": "wma"},
                                                {"label": "Donchian Channel Middle (DCM)", "value": "dcm"}
                                            ],
                                            value="sma",
                                            clearable=False
                                        )
                                    ],
                                    width=6
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Average Period"),
                                        dcc.Slider(
                                            id="live-average-period-slider",
                                            min=5,
                                            max=50,
                                            step=1,
                                            value=20,
                                            marks={i: str(i) for i in range(5, 51, 5)},
                                            tooltip={"placement": "bottom", "always_visible": True}
                                        )
                                    ],
                                    width=6
                                )
                            ],
                            className="mb-3"
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Envelope Percentage"),
                                        dcc.Slider(
                                            id="live-envelope-slider",
                                            min=0.1,
                                            max=5.0,
                                            step=0.1,
                                            value=1.0,
                                            marks={i/10: f"{i/10}%" for i in range(1, 51, 5)},
                                            tooltip={"placement": "bottom", "always_visible": True}
                                        )
                                    ],
                                    width=12
                                )
                            ],
                            className="mb-3"
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Trading Mode"),
                                        dcc.Dropdown(
                                            id="live-mode-dropdown",
                                            options=[
                                                {"label": "Long Only", "value": "long"},
                                                {"label": "Short Only", "value": "short"},
                                                {"label": "Long & Short", "value": "both"}
                                            ],
                                            value="both",
                                            clearable=False
                                        )
                                    ],
                                    width=6
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Stop Loss (%)"),
                                        dcc.Slider(
                                            id="live-stop-loss-slider",
                                            min=0.5,
                                            max=10.0,
                                            step=0.5,
                                            value=3.0,
                                            marks={i/2: f"{i/2}%" for i in range(1, 21, 2)},
                                            tooltip={"placement": "bottom", "always_visible": True}
                                        )
                                    ],
                                    width=6
                                )
                            ],
                            className="mb-3"
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Position Size (%)"),
                                        dcc.Slider(
                                            id="live-position-size-slider",
                                            min=1,
                                            max=100,
                                            step=1,
                                            value=10,
                                            marks={i: f"{i}%" for i in range(0, 101, 10)},
                                            tooltip={"placement": "bottom", "always_visible": True}
                                        )
                                    ],
                                    width=12
                                )
                            ],
                            className="mb-3"
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Button(
                                            "Start Trading",
                                            id="start-trading-button",
                                            color="success",
                                            className="w-100"
                                        )
                                    ],
                                    width=6
                                ),
                                dbc.Col(
                                    [
                                        dbc.Button(
                                            "Stop Trading",
                                            id="stop-trading-button",
                                            color="danger",
                                            className="w-100",
                                            disabled=True
                                        )
                                    ],
                                    width=6
                                )
                            ]
                        )
                    ]
                )
            ],
            className="mb-4"
        )
        
        # Create trading status card
        trading_status_card = dbc.Card(
            [
                dbc.CardHeader("Trading Status"),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.H5("Status"),
                                        html.P(id="trading-status", children="Inactive", className="text-muted")
                                    ],
                                    width=4
                                ),
                                dbc.Col(
                                    [
                                        html.H5("Running Time"),
                                        html.P(id="trading-running-time", children="00:00:00", className="text-muted")
                                    ],
                                    width=4
                                ),
                                dbc.Col(
                                    [
                                        html.H5("Last Update"),
                                        html.P(id="trading-last-update", children="-", className="text-muted")
                                    ],
                                    width=4
                                )
                            ],
                            className="mb-3"
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.H5("Current Price"),
                                        html.P(id="trading-current-price", children="$0.00", className="text-muted")
                                    ],
                                    width=4
                                ),
                                dbc.Col(
                                    [
                                        html.H5("Position"),
                                        html.P(id="trading-position", children="No position", className="text-muted")
                                    ],
                                    width=4
                                ),
                                dbc.Col(
                                    [
                                        html.H5("Unrealized P/L"),
                                        html.P(id="trading-unrealized-pl", children="$0.00", className="text-muted")
                                    ],
                                    width=4
                                )
                            ]
                        )
                    ]
                )
            ],
            className="mb-4"
        )
        
        # Create price chart card
        price_chart_card = dbc.Card(
            [
                dbc.CardHeader("Price Chart"),
                dbc.CardBody(
                    [
                        dcc.Graph(
                            id="live-price-chart",
                            figure=go.Figure(
                                layout=dict(
                                    template="plotly_dark",
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    margin=dict(l=10, r=10, t=10, b=10),
                                    height=400
                                )
                            )
                        )
                    ]
                )
            ],
            className="mb-4"
        )
        
        # Create trade history card
        trade_history_card = dbc.Card(
            [
                dbc.CardHeader("Trade History"),
                dbc.CardBody(
                    [
                        html.Div(id="live-trade-history")
                    ]
                )
            ],
            className="mb-4"
        )
        
        # Create layout
        layout = html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(strategy_config_card, width=4),
                        dbc.Col(
                            [
                                trading_status_card,
                                price_chart_card
                            ],
                            width=8
                        )
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(trade_history_card, width=12)
                    ]
                )
            ]
        )
        
        return layout
    
    def _create_settings_tab(self):
        """
        Create the settings tab.
        
        Returns:
            dash.html.Div: Settings tab layout
        """
        # Create exchange settings card
        exchange_settings_card = dbc.Card(
            [
                dbc.CardHeader("Exchange Settings"),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Exchange"),
                                        dcc.Dropdown(
                                            id="exchange-dropdown",
                                            options=[
                                                {"label": "Binance", "value": "binance"},
                                                {"label": "Bybit", "value": "bybit"},
                                                {"label": "Coinbase", "value": "coinbase"},
                                                {"label": "Kraken", "value": "kraken"},
                                                {"label": "KuCoin", "value": "kucoin"}
                                            ],
                                            value="binance",
                                            clearable=False
                                        )
                                    ],
                                    width=6
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Use Testnet"),
                                        dbc.Checklist(
                                            options=[
                                                {"label": "Yes", "value": 1}
                                            ],
                                            value=[1],
                                            id="testnet-checklist",
                                            switch=True
                                        )
                                    ],
                                    width=6
                                )
                            ],
                            className="mb-3"
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("API Key"),
                                        dbc.Input(
                                            id="api-key-input",
                                            type="password",
                                            placeholder="Enter API Key"
                                        )
                                    ],
                                    width=12
                                )
                            ],
                            className="mb-3"
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("API Secret"),
                                        dbc.Input(
                                            id="api-secret-input",
                                            type="password",
                                            placeholder="Enter API Secret"
                                        )
                                    ],
                                    width=12
                                )
                            ],
                            className="mb-3"
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Button(
                                            "Save Exchange Settings",
                                            id="save-exchange-settings-button",
                                            color="primary",
                                            className="w-100"
                                        )
                                    ],
                                    width=12
                                )
                            ]
                        )
                    ]
                )
            ],
            className="mb-4"
        )
        
        # Create notification settings card
        notification_settings_card = dbc.Card(
            [
                dbc.CardHeader("Notification Settings"),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Email Notifications"),
                                        dbc.Checklist(
                                            options=[
                                                {"label": "Enable", "value": 1}
                                            ],
                                            value=[],
                                            id="email-notifications-checklist",
                                            switch=True
                                        )
                                    ],
                                    width=6
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Email Address"),
                                        dbc.Input(
                                            id="email-address-input",
                                            type="email",
                                            placeholder="Enter Email Address",
                                            disabled=True
                                        )
                                    ],
                                    width=6
                                )
                            ],
                            className="mb-3"
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Notification Events"),
                                        dbc.Checklist(
                                            options=[
                                                {"label": "Trade Entry", "value": "entry"},
                                                {"label": "Trade Exit", "value": "exit"},
                                                {"label": "Stop Loss", "value": "stop_loss"},
                                                {"label": "Error", "value": "error"}
                                            ],
                                            value=["entry", "exit", "stop_loss", "error"],
                                            id="notification-events-checklist",
                                            inline=True
                                        )
                                    ],
                                    width=12
                                )
                            ],
                            className="mb-3"
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Button(
                                            "Save Notification Settings",
                                            id="save-notification-settings-button",
                                            color="primary",
                                            className="w-100"
                                        )
                                    ],
                                    width=12
                                )
                            ]
                        )
                    ]
                )
            ],
            className="mb-4"
        )
        
        # Create appearance settings card
        appearance_settings_card = dbc.Card(
            [
                dbc.CardHeader("Appearance Settings"),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Theme"),
                                        dbc.RadioItems(
                                            options=[
                                                {"label": "Dark", "value": "dark"},
                                                {"label": "Light", "value": "light"}
                                            ],
                                            value="dark",
                                            id="theme-radio",
                                            inline=True
                                        )
                                    ],
                                    width=6
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Chart Style"),
                                        dbc.RadioItems(
                                            options=[
                                                {"label": "Candlestick", "value": "candlestick"},
                                                {"label": "OHLC", "value": "ohlc"},
                                                {"label": "Line", "value": "line"}
                                            ],
                                            value="candlestick",
                                            id="chart-style-radio",
                                            inline=True
                                        )
                                    ],
                                    width=6
                                )
                            ],
                            className="mb-3"
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Button(
                                            "Save Appearance Settings",
                                            id="save-appearance-settings-button",
                                            color="primary",
                                            className="w-100"
                                        )
                                    ],
                                    width=12
                                )
                            ]
                        )
                    ]
                )
            ],
            className="mb-4"
        )
        
        # Create layout
        layout = html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(exchange_settings_card, width=6),
                        dbc.Col(notification_settings_card, width=6)
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(appearance_settings_card, width=6)
                    ]
                )
            ]
        )
        
        return layout
    
    def _register_callbacks(self):
        """
        Register callbacks for the dashboard.
        """
        # Register callback for running backtest
        @self.app.callback(
            [
                Output("backtest-results-content", "children"),
                Output("backtest-results", "data")
            ],
            [
                Input("run-backtest-button", "n_clicks")
            ],
            [
                State("backtest-pair-dropdown", "value"),
                State("backtest-timeframe-dropdown", "value"),
                State("backtest-date-range", "start_date"),
                State("backtest-date-range", "end_date"),
                State("backtest-average-type-dropdown", "value"),
                State("backtest-average-period-slider", "value"),
                State("backtest-envelope-slider", "value"),
                State("backtest-mode-dropdown", "value"),
                State("backtest-stop-loss-slider", "value"),
                State("backtest-initial-balance-input", "value"),
                State("backtest-commission-input", "value")
            ],
            prevent_initial_call=True
        )
        def run_backtest(n_clicks, pair, timeframe, start_date, end_date, average_type, average_period, envelope, mode, stop_loss, initial_balance, commission):
            if n_clicks is None:
                return html.Div("Run a backtest to see results."), {}
            
            # Convert dates
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            
            # Create strategy parameters
            strategy_params = {
                'average_type': average_type,
                'average_period': average_period,
                'envelopes': [envelope / 100],  # Convert to decimal
                'stop_loss_pct': stop_loss / 100,  # Convert to decimal
                'mode': mode
            }
            
            # Run backtest
            backtest_results = self.backtest_manager.run_backtest(
                strategy_name='envelope',
                symbol=pair,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                strategy_params=strategy_params,
                initial_balance=initial_balance,
                commission=commission / 100  # Convert to decimal
            )
            
            # Create results content
            if not backtest_results:
                return html.Div("Error running backtest. Check logs for details."), {}
            
            # Create metrics cards
            total_return = backtest_results["metrics"].get("total_return", 0)
            return_class = "text-success" if total_return >= 0 else "text-danger"
            
            metrics_cards = [
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.H5("Total Return", className="card-title"),
                                html.H3(f"{total_return:.2f}%", className=return_class)
                            ]
                        )
                    ],
                    className="mb-4"
                ),
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.H5("Sharpe Ratio", className="card-title"),
                                html.H3(f"{backtest_results['metrics'].get('sharpe_ratio', 0):.2f}")
                            ]
                        )
                    ],
                    className="mb-4"
                ),
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.H5("Max Drawdown", className="card-title"),
                                html.H3(f"{backtest_results['metrics'].get('max_drawdown', 0):.2f}%", className="text-danger")
                            ]
                        )
                    ],
                    className="mb-4"
                ),
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.H5("Win Rate", className="card-title"),
                                html.H3(f"{backtest_results['metrics'].get('win_rate', 0) * 100:.2f}%")
                            ]
                        )
                    ],
                    className="mb-4"
                )
            ]
            
            # Create equity curve
            equity_curve = self._create_equity_curve_figure(backtest_results)
            
            # Create drawdown chart
            drawdown_chart = self._create_drawdown_figure(backtest_results)
            
            # Create trade analysis
            trade_analysis = self._create_trade_analysis_figure(backtest_results)
            
            # Create trade table
            trade_table = self._create_trade_table(backtest_results)
            
            # Create results content
            results_content = html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(metrics_cards[0], width=3),
                            dbc.Col(metrics_cards[1], width=3),
                            dbc.Col(metrics_cards[2], width=3),
                            dbc.Col(metrics_cards[3], width=3)
                        ],
                        className="mb-4"
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Equity Curve"),
                                            dbc.CardBody(
                                                [
                                                    dcc.Graph(figure=equity_curve)
                                                ]
                                            )
                                        ],
                                        className="mb-4"
                                    )
                                ],
                                width=6
                            ),
                            dbc.Col(
                                [
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Drawdown"),
                                            dbc.CardBody(
                                                [
                                                    dcc.Graph(figure=drawdown_chart)
                                                ]
                                            )
                                        ],
                                        className="mb-4"
                                    )
                                ],
                                width=6
                            )
                        ],
                        className="mb-4"
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Trade Analysis"),
                                            dbc.CardBody(
                                                [
                                                    dcc.Graph(figure=trade_analysis)
                                                ]
                                            )
                                        ],
                                        className="mb-4"
                                    )
                                ],
                                width=12
                            )
                        ],
                        className="mb-4"
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Trades"),
                                            dbc.CardBody(
                                                [
                                                    trade_table
                                                ]
                                            )
                                        ],
                                        className="mb-4"
                                    )
                                ],
                                width=12
                            )
                        ]
                    )
                ]
            )
            
            return results_content, backtest_results
        
        # Register callback for starting/stopping trading
        @self.app.callback(
            [
                Output("start-trading-button", "disabled"),
                Output("stop-trading-button", "disabled"),
                Output("live-trading-state", "data")
            ],
            [
                Input("start-trading-button", "n_clicks"),
                Input("stop-trading-button", "n_clicks")
            ],
            [
                State("live-pair-dropdown", "value"),
                State("live-timeframe-dropdown", "value"),
                State("live-average-type-dropdown", "value"),
                State("live-average-period-slider", "value"),
                State("live-envelope-slider", "value"),
                State("live-mode-dropdown", "value"),
                State("live-stop-loss-slider", "value"),
                State("live-position-size-slider", "value"),
                State("live-trading-state", "data")
            ],
            prevent_initial_call=True
        )
        def toggle_trading(start_clicks, stop_clicks, pair, timeframe, average_type, average_period, envelope, mode, stop_loss, position_size, trading_state):
            ctx = dash.callback_context
            
            if not ctx.triggered:
                return False, True, {}
            
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            
            if button_id == "start-trading-button" and start_clicks:
                # Create strategy parameters
                strategy_params = {
                    'average_type': average_type,
                    'average_period': average_period,
                    'envelopes': [envelope / 100],  # Convert to decimal
                    'stop_loss_pct': stop_loss / 100,  # Convert to decimal
                    'mode': mode,
                    'position_size_percentage': position_size
                }
                
                # Start trading
                self.trading_engine.start_trading(
                    strategy_name='envelope',
                    symbol=pair,
                    timeframe=timeframe,
                    strategy_params=strategy_params
                )
                
                # Create trading state
                trading_state = {
                    'active': True,
                    'start_time': datetime.now().isoformat(),
                    'pair': pair,
                    'timeframe': timeframe,
                    'strategy_params': strategy_params
                }
                
                return True, False, trading_state
                
            elif button_id == "stop-trading-button" and stop_clicks:
                # Stop trading
                self.trading_engine.stop_trading()
                
                # Update trading state
                if trading_state:
                    trading_state['active'] = False
                    trading_state['stop_time'] = datetime.now().isoformat()
                
                return False, True, trading_state
            
            # Default return
            return dash.no_update, dash.no_update, dash.no_update
        
        # Register callback for updating trading status
        @self.app.callback(
            [
                Output("trading-status", "children"),
                Output("trading-status", "className"),
                Output("trading-running-time", "children"),
                Output("trading-last-update", "children"),
                Output("trading-current-price", "children"),
                Output("trading-position", "children"),
                Output("trading-position", "className"),
                Output("trading-unrealized-pl", "children"),
                Output("trading-unrealized-pl", "className"),
                Output("live-price-chart", "figure"),
                Output("live-trade-history", "children")
            ],
            [
                Input("interval-component", "n_intervals")
            ],
            [
                State("live-trading-state", "data")
            ]
        )
        def update_trading_status(n_intervals, trading_state):
            if not trading_state or not trading_state.get('active', False):
                # Default values when not trading
                return (
                    "Inactive", "text-muted",
                    "00:00:00", "-",
                    "$0.00", "No position", "text-muted",
                    "$0.00", "text-muted",
                    go.Figure(layout=dict(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=10, r=10, t=10, b=10),
                        height=400
                    )),
                    html.Div("No trade history available.")
                )
            
            # Get trading status
            status = self.trading_engine.get_status()
            
            if not status:
                return dash.no_update
            
            # Calculate running time
            start_time = datetime.fromisoformat(trading_state['start_time'])
            running_time = datetime.now() - start_time
            running_time_str = str(running_time).split('.')[0]  # Remove microseconds
            
            # Format last update
            last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Format current price
            current_price = f"${status.get('current_price', 0):.2f}"
            
            # Format position
            position = status.get('position', {})
            position_size = position.get('size', 0)
            position_type = position.get('type', '')
            
            if position_size > 0:
                position_str = f"{position_type.capitalize()}: {position_size:.6f}"
                position_class = "text-success" if position_type == 'long' else "text-danger"
            else:
                position_str = "No position"
                position_class = "text-muted"
            
            # Format unrealized P/L
            unrealized_pl = position.get('unrealized_pl', 0)
            unrealized_pl_str = f"${unrealized_pl:.2f}"
            unrealized_pl_class = "text-success" if unrealized_pl >= 0 else "text-danger"
            
            # Create price chart
            price_chart = self._create_live_price_chart(status)
            
            # Create trade history table
            trade_history = self._create_trade_history_table(status)
            
            return (
                "Active", "text-success",
                running_time_str, last_update,
                current_price, position_str, position_class,
                unrealized_pl_str, unrealized_pl_class,
                price_chart, trade_history
            )
        
        # Register callback for email notifications toggle
        @self.app.callback(
            Output("email-address-input", "disabled"),
            [Input("email-notifications-checklist", "value")],
            prevent_initial_call=True
        )
        def toggle_email_input(values):
            return not (1 in values)
    
    def _create_equity_curve_figure(self, backtest_results):
        """
        Create equity curve figure.
        
        Args:
            backtest_results (Dict[str, Any]): Backtest results
            
        Returns:
            go.Figure: Equity curve figure
        """
        # Extract trades
        trades = backtest_results.get('trades', [])
        
        if not trades:
            return go.Figure()
        
        # Create DataFrame with balance history
        balance_history = []
        
        for trade in trades:
            balance_history.append({
                'date': trade['date'],
                'balance': trade['balance']
            })
        
        balance_df = pd.DataFrame(balance_history)
        balance_df['date'] = pd.to_datetime(balance_df['date'])
        
        # Create figure
        fig = go.Figure()
        
        # Add equity curve
        fig.add_trace(
            go.Scatter(
                x=balance_df['date'],
                y=balance_df['balance'],
                mode='lines',
                name='Equity',
                line=dict(color=COLORS['primary'], width=2)
            )
        )
        
        # Add initial balance
        initial_balance = backtest_results.get('initial_balance', 0)
        fig.add_trace(
            go.Scatter(
                x=[balance_df['date'].min(), balance_df['date'].max()],
                y=[initial_balance, initial_balance],
                mode='lines',
                name='Initial Balance',
                line=dict(color=COLORS['danger'], width=1, dash='dash')
            )
        )
        
        # Update layout
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(
                title="Date",
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.1)"
            ),
            yaxis=dict(
                title="Balance",
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.1)"
            )
        )
        
        return fig
    
    def _create_drawdown_figure(self, backtest_results):
        """
        Create drawdown figure.
        
        Args:
            backtest_results (Dict[str, Any]): Backtest results
            
        Returns:
            go.Figure: Drawdown figure
        """
        # Extract trades
        trades = backtest_results.get('trades', [])
        
        if not trades:
            return go.Figure()
        
        # Create DataFrame with balance history
        balance_history = []
        
        for trade in trades:
            balance_history.append({
                'date': trade['date'],
                'balance': trade['balance']
            })
        
        balance_df = pd.DataFrame(balance_history)
        balance_df['date'] = pd.to_datetime(balance_df['date'])
        
        # Calculate running maximum
        balance_df['running_max'] = balance_df['balance'].cummax()
        
        # Calculate drawdown
        balance_df['drawdown'] = (balance_df['running_max'] - balance_df['balance']) / balance_df['running_max'] * 100
        
        # Create figure
        fig = go.Figure()
        
        # Add drawdown
        fig.add_trace(
            go.Scatter(
                x=balance_df['date'],
                y=balance_df['drawdown'],
                mode='lines',
                name='Drawdown',
                line=dict(color=COLORS['danger'], width=2),
                fill='tozeroy'
            )
        )
        
        # Update layout
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
            xaxis=dict(
                title="Date",
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.1)"
            ),
            yaxis=dict(
                title="Drawdown (%)",
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.1)",
                autorange="reversed"
            )
        )
        
        return fig
    
    def _create_trade_analysis_figure(self, backtest_results):
        """
        Create trade analysis figure.
        
        Args:
            backtest_results (Dict[str, Any]): Backtest results
            
        Returns:
            go.Figure: Trade analysis figure
        """
        # Extract trades
        trades = backtest_results.get('trades', [])
        
        if not trades:
            return go.Figure()
        
        # Filter trades
        close_trades = [t for t in trades if t['type'] in ['close_long', 'close_short']]
        
        if not close_trades:
            return go.Figure()
        
        # Extract profits
        profits = [t.get('profit', 0) for t in close_trades]
        
        # Create figure with subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Profit Distribution", "Cumulative Profit"),
            specs=[[{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # Add profit distribution
        fig.add_trace(
            go.Histogram(
                x=profits,
                name="Profit Distribution",
                marker=dict(color=COLORS['primary']),
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Add zero line to profit distribution
        fig.add_vline(
            x=0,
            line=dict(color=COLORS['danger'], width=1, dash='dash'),
            row=1, col=1
        )
        
        # Add cumulative profit
        cumulative_profit = np.cumsum(profits)
        fig.add_trace(
            go.Scatter(
                x=list(range(len(cumulative_profit))),
                y=cumulative_profit,
                mode='lines',
                name="Cumulative Profit",
                line=dict(color=COLORS['success'], width=2)
            ),
            row=1, col=2
        )
        
        # Add zero line to cumulative profit
        fig.add_hline(
            y=0,
            line=dict(color=COLORS['danger'], width=1, dash='dash'),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=30, b=10),
            height=300,
            showlegend=False,
            xaxis=dict(
                title="Profit",
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.1)"
            ),
            yaxis=dict(
                title="Frequency",
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.1)"
            ),
            xaxis2=dict(
                title="Trade Number",
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.1)"
            ),
            yaxis2=dict(
                title="Cumulative Profit",
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.1)"
            )
        )
        
        return fig
    
    def _create_trade_table(self, backtest_results):
        """
        Create trade table.
        
        Args:
            backtest_results (Dict[str, Any]): Backtest results
            
        Returns:
            dash.html.Div: Trade table
        """
        # Extract trades
        trades = backtest_results.get('trades', [])
        
        if not trades:
            return html.Div("No trades to display.")
        
        # Create table header
        header = html.Thead(
            html.Tr(
                [
                    html.Th("Date"),
                    html.Th("Type"),
                    html.Th("Price"),
                    html.Th("Position"),
                    html.Th("Profit"),
                    html.Th("Balance")
                ]
            )
        )
        
        # Create table rows
        rows = []
        
        for trade in trades:
            # Format date
            date = trade.get('date')
            if isinstance(date, str):
                date = datetime.fromisoformat(date)
            date_str = date.strftime("%Y-%m-%d %H:%M:%S")
            
            # Format type
            trade_type = trade.get('type', '')
            
            # Format profit
            profit = trade.get('profit', 0)
            profit_class = "text-success" if profit > 0 else "text-danger" if profit < 0 else ""
            
            # Create row
            row = html.Tr(
                [
                    html.Td(date_str),
                    html.Td(trade_type),
                    html.Td(f"${trade.get('price', 0):.2f}"),
                    html.Td(f"{trade.get('position', 0):.6f}"),
                    html.Td(f"${profit:.2f}", className=profit_class) if 'profit' in trade else html.Td("-"),
                    html.Td(f"${trade.get('balance', 0):.2f}")
                ]
            )
            
            rows.append(row)
        
        # Create table body
        body = html.Tbody(rows)
        
        # Create table
        table = dbc.Table(
            [header, body],
            bordered=True,
            hover=True,
            responsive=True,
            striped=True
        )
        
        return table
    
    def _create_live_price_chart(self, status):
        """
        Create live price chart.
        
        Args:
            status (Dict[str, Any]): Trading status
            
        Returns:
            go.Figure: Live price chart
        """
        # Extract data
        data = status.get('data', pd.DataFrame())
        
        if data.empty:
            return go.Figure(layout=dict(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10),
                height=400
            ))
        
        # Convert to DataFrame if necessary
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        
        # Create figure
        fig = go.Figure()
        
        # Add price
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name="Price"
            )
        )
        
        # Add average
        if 'average' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['average'],
                    mode='lines',
                    name='Average',
                    line=dict(color=COLORS['info'], width=1)
                )
            )
        
        # Add envelope bands
        for col in data.columns:
            if col.startswith('band_high_'):
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data[col],
                        mode='lines',
                        name=f'Upper Band',
                        line=dict(color=COLORS['danger'], width=1, dash='dash')
                    )
                )
            
            if col.startswith('band_low_'):
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data[col],
                        mode='lines',
                        name=f'Lower Band',
                        line=dict(color=COLORS['success'], width=1, dash='dash')
                    )
                )
        
        # Add buy signals
        if 'buy_signal' in data.columns:
            buy_signals = data[data['buy_signal']]
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['close'],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(
                        symbol='triangle-up',
                        size=10,
                        color=COLORS['success']
                    )
                )
            )
        
        # Add sell signals
        if 'sell_signal' in data.columns:
            sell_signals = data[data['sell_signal']]
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['close'],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(
                        symbol='triangle-down',
                        size=10,
                        color=COLORS['danger']
                    )
                )
            )
        
        # Add trades
        trades = status.get('trades', [])
        
        for trade in trades:
            if trade['type'] == 'buy':
                fig.add_trace(
                    go.Scatter(
                        x=[trade['date']],
                        y=[trade['price']],
                        mode='markers',
                        name='Buy',
                        marker=dict(
                            symbol='circle',
                            size=8,
                            color=COLORS['success']
                        ),
                        showlegend=False
                    )
                )
            elif trade['type'] == 'sell':
                fig.add_trace(
                    go.Scatter(
                        x=[trade['date']],
                        y=[trade['price']],
                        mode='markers',
                        name='Sell',
                        marker=dict(
                            symbol='circle',
                            size=8,
                            color=COLORS['danger']
                        ),
                        showlegend=False
                    )
                )
            elif trade['type'] == 'close_long':
                fig.add_trace(
                    go.Scatter(
                        x=[trade['date']],
                        y=[trade['price']],
                        mode='markers',
                        name='Close Long',
                        marker=dict(
                            symbol='x',
                            size=8,
                            color=COLORS['info']
                        ),
                        showlegend=False
                    )
                )
            elif trade['type'] == 'close_short':
                fig.add_trace(
                    go.Scatter(
                        x=[trade['date']],
                        y=[trade['price']],
                        mode='markers',
                        name='Close Short',
                        marker=dict(
                            symbol='x',
                            size=8,
                            color=COLORS['warning']
                        ),
                        showlegend=False
                    )
                )
        
        # Update layout
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(
                title="Date",
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.1)",
                rangeslider=dict(visible=False)
            ),
            yaxis=dict(
                title="Price",
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.1)"
            )
        )
        
        return fig
    
    def _create_trade_history_table(self, status):
        """
        Create trade history table.
        
        Args:
            status (Dict[str, Any]): Trading status
            
        Returns:
            dash.html.Div: Trade history table
        """
        # Extract trades
        trades = status.get('trades', [])
        
        if not trades:
            return html.Div("No trade history available.")
        
        # Create table header
        header = html.Thead(
            html.Tr(
                [
                    html.Th("Date"),
                    html.Th("Type"),
                    html.Th("Price"),
                    html.Th("Position"),
                    html.Th("Profit"),
                    html.Th("Balance")
                ]
            )
        )
        
        # Create table rows
        rows = []
        
        for trade in trades:
            # Format date
            date = trade.get('date')
            if isinstance(date, str):
                date = datetime.fromisoformat(date)
            date_str = date.strftime("%Y-%m-%d %H:%M:%S")
            
            # Format type
            trade_type = trade.get('type', '')
            
            # Format profit
            profit = trade.get('profit', 0)
            profit_class = "text-success" if profit > 0 else "text-danger" if profit < 0 else ""
            
            # Create row
            row = html.Tr(
                [
                    html.Td(date_str),
                    html.Td(trade_type),
                    html.Td(f"${trade.get('price', 0):.2f}"),
                    html.Td(f"{trade.get('position', 0):.6f}"),
                    html.Td(f"${profit:.2f}", className=profit_class) if 'profit' in trade else html.Td("-"),
                    html.Td(f"${trade.get('balance', 0):.2f}")
                ]
            )
            
            rows.append(row)
        
        # Create table body
        body = html.Tbody(rows)
        
        # Create table
        table = dbc.Table(
            [header, body],
            bordered=True,
            hover=True,
            responsive=True,
            striped=True
        )
        
        return table
    
    def run(self, host='0.0.0.0', port=8050, debug=False):
        """
        Run the dashboard.
        
        Args:
            host (str): Host to run the dashboard on
            port (int): Port to run the dashboard on
            debug (bool): Whether to run in debug mode
        """
        self.app.run_server(host=host, port=port, debug=debug)
