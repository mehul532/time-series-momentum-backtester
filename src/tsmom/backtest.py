"""Backtesting utilities with explicit signal/return alignment."""

from __future__ import annotations

import pandas as pd

from .signals import shift_signal


def compute_turnover(position: pd.Series | pd.DataFrame) -> pd.Series:
    """Compute absolute position changes.

    For a single asset this is ``abs(position.diff())``. For a DataFrame of
    weights, turnover is the row-wise sum of absolute weight changes.
    """

    previous = position.shift(1).fillna(0)
    changes = (position.fillna(0) - previous).abs()
    if isinstance(changes, pd.DataFrame):
        return changes.sum(axis=1)
    return changes


def compute_strategy_returns(
    asset_returns: pd.Series,
    signal: pd.Series,
    cost_bps: float = 0.0,
    cash_return: float | pd.Series = 0.0,
    signal_is_position: bool = False,
) -> pd.Series:
    """Apply a raw signal to returns with one-day lag and transaction costs.

    By default, ``signal`` is treated as an unshifted research signal. Set
    ``signal_is_position=True`` only when the input has already been explicitly
    shifted into a tradable position.
    """

    returns = asset_returns.astype(float).fillna(0.0)
    raw_signal = signal.reindex(returns.index).ffill().fillna(0.0).astype(float)
    position = raw_signal if signal_is_position else shift_signal(raw_signal, lag=1).astype(float)
    cash = (
        cash_return.reindex(returns.index).fillna(0.0)
        if isinstance(cash_return, pd.Series)
        else pd.Series(float(cash_return), index=returns.index)
    )
    turnover = compute_turnover(position)
    cost = turnover * (float(cost_bps) / 10_000.0)
    strategy_returns = position * returns + (1.0 - position) * cash - cost
    strategy_returns.name = "strategy_return"
    return strategy_returns


def benchmark_buy_hold(close: pd.Series) -> pd.Series:
    """Compute buy-and-hold returns from close prices."""

    returns = close.astype(float).pct_change(fill_method=None).fillna(0.0)
    returns.name = "benchmark_return"
    return returns


def align_signal_and_returns(close: pd.Series, signal: pd.Series) -> pd.DataFrame:
    """Align close, raw signal, shifted position, and asset returns.

    The ``position`` column is always ``signal.shift(1)`` with missing values
    filled by cash exposure, which prevents same-day signal use by default.
    """

    close_series = close.dropna().astype(float).sort_index()
    raw_signal = signal.reindex(close_series.index).ffill().fillna(0).astype(float)
    asset_returns = close_series.pct_change(fill_method=None).fillna(0.0)
    position = shift_signal(raw_signal, lag=1).astype(float)
    return pd.DataFrame(
        {
            "close": close_series,
            "asset_return": asset_returns,
            "signal": raw_signal,
            "position": position,
        },
        index=close_series.index,
    )


def backtest_single_asset(
    close: pd.Series,
    signal: pd.Series,
    cost_bps: float = 0.0,
    cash_return: float | pd.Series = 0.0,
    signal_is_position: bool = False,
) -> pd.DataFrame:
    """Backtest a single-asset long/cash time-series momentum strategy."""

    aligned = align_signal_and_returns(close, signal)
    if signal_is_position:
        aligned["position"] = signal.reindex(aligned.index).ffill().fillna(0).astype(float)
    aligned["turnover"] = compute_turnover(aligned["position"])
    aligned["strategy_return"] = compute_strategy_returns(
        aligned["asset_return"],
        aligned["position"],
        cost_bps=cost_bps,
        cash_return=cash_return,
        signal_is_position=True,
    )
    aligned["benchmark_return"] = benchmark_buy_hold(aligned["close"])
    aligned["strategy_equity"] = (1.0 + aligned["strategy_return"]).cumprod()
    aligned["benchmark_equity"] = (1.0 + aligned["benchmark_return"]).cumprod()
    aligned.insert(0, "date", aligned.index)
    return aligned[
        [
            "date",
            "close",
            "asset_return",
            "signal",
            "position",
            "strategy_return",
            "benchmark_return",
            "strategy_equity",
            "benchmark_equity",
            "turnover",
        ]
    ]
