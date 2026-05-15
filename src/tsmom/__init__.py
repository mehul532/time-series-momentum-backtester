"""Time-Series Momentum Backtester.

Research-inspired tools for building ETF trend-following signals, running
simple backtests, and checking robustness without look-ahead bias.
"""

from .backtest import backtest_single_asset
from .metrics import summarize_performance
from .signals import shift_signal, trailing_return_signal

__all__ = [
    "backtest_single_asset",
    "shift_signal",
    "summarize_performance",
    "trailing_return_signal",
]

__version__ = "0.1.0"
