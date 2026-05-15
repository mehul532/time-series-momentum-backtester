"""Portfolio construction helpers for multi-asset trend following."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import compute_turnover
from .signals import shift_signal, trailing_return_signal


def build_equal_weight_portfolio(strategy_returns_df: pd.DataFrame) -> pd.Series:
    """Average available strategy return streams into an equal-weight portfolio."""

    portfolio_returns = strategy_returns_df.mean(axis=1, skipna=True).fillna(0.0)
    portfolio_returns.name = "portfolio_return"
    return portfolio_returns


def build_vol_target_portfolio(
    returns_df: pd.DataFrame,
    target_vol: float = 0.10,
    lookback: int = 63,
    max_leverage: float = 1.0,
) -> pd.DataFrame:
    """Scale an equal-weight return stream using lagged realized volatility."""

    if lookback <= 1:
        raise ValueError("lookback must be greater than one.")
    base_returns = build_equal_weight_portfolio(returns_df)
    realized_vol = base_returns.rolling(lookback, min_periods=lookback).std(ddof=0) * np.sqrt(252)
    leverage = (target_vol / realized_vol).clip(lower=0.0, upper=max_leverage)
    leverage = leverage.shift(1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    targeted_returns = base_returns * leverage
    return pd.DataFrame(
        {
            "base_return": base_returns,
            "volatility_estimate": realized_vol,
            "leverage": leverage,
            "portfolio_return": targeted_returns,
            "equity": (1.0 + targeted_returns).cumprod(),
        }
    )


def multi_asset_tsmom(
    close_df: pd.DataFrame,
    lookback: int = 252,
    cost_bps: float = 5.0,
    vol_target: float | None = None,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Run independent ETF time-series momentum and equal-weight active assets.

    Signals are generated per asset and shifted by one trading day before
    weights are computed. Active assets receive equal weight; if no asset is
    active, the portfolio holds cash for that day.
    """

    close = close_df.sort_index().astype(float)
    asset_returns_raw = close.pct_change(fill_method=None)
    asset_returns = asset_returns_raw.fillna(0.0)
    signals = trailing_return_signal(close, lookback=lookback)
    positions = shift_signal(signals, lag=1).astype(float)
    active_counts = positions.sum(axis=1).replace(0.0, np.nan)
    weights = positions.div(active_counts, axis=0).fillna(0.0)
    turnover = compute_turnover(weights)
    cost = turnover * (float(cost_bps) / 10_000.0)
    strategy_returns = (weights * asset_returns).sum(axis=1) - cost
    benchmark_returns = asset_returns_raw.mean(axis=1, skipna=True).fillna(0.0)

    results = pd.DataFrame(
        {
            "strategy_return": strategy_returns,
            "benchmark_return": benchmark_returns,
            "strategy_equity": (1.0 + strategy_returns).cumprod(),
            "benchmark_equity": (1.0 + benchmark_returns).cumprod(),
            "turnover": turnover,
            "active_assets": positions.sum(axis=1),
        },
        index=close.index,
    )
    results.index.name = "date"

    output: dict[str, pd.DataFrame | pd.Series] = {
        "results": results,
        "weights": weights,
        "signals": signals,
        "positions": positions,
        "asset_returns": asset_returns,
    }
    if vol_target is not None:
        output["vol_target"] = build_vol_target_portfolio(
            strategy_returns.to_frame("strategy"),
            target_vol=vol_target,
            max_leverage=1.0,
        )
    return output
