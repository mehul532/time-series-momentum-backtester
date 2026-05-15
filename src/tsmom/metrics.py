"""Performance metrics for daily return series and equity curves."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clean_returns(returns: pd.Series) -> pd.Series:
    return pd.Series(returns).dropna().astype(float)


def cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compound annual growth rate from periodic returns."""

    clean = _clean_returns(returns)
    if clean.empty:
        return float("nan")
    equity = (1.0 + clean).cumprod()
    years = len(clean) / periods_per_year
    if years <= 0 or equity.iloc[-1] <= 0:
        return float("nan")
    return float(equity.iloc[-1] ** (1.0 / years) - 1.0)


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized standard deviation of periodic returns."""

    clean = _clean_returns(returns)
    if clean.empty:
        return float("nan")
    return float(clean.std(ddof=0) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio using a constant annual risk-free rate."""

    clean = _clean_returns(returns)
    if clean.empty:
        return float("nan")
    excess = clean - (risk_free_rate / periods_per_year)
    vol = excess.std(ddof=0)
    if np.isclose(vol, 0.0):
        return float("nan")
    return float(excess.mean() / vol * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum drawdown as a negative decimal."""

    equity = pd.Series(equity_curve).dropna().astype(float)
    if equity.empty:
        return float("nan")
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series) -> float:
    """CAGR divided by absolute maximum drawdown."""

    clean = _clean_returns(returns)
    if clean.empty:
        return float("nan")
    equity = (1.0 + clean).cumprod()
    drawdown = abs(max_drawdown(equity))
    if np.isclose(drawdown, 0.0):
        return float("nan")
    return float(cagr(clean) / drawdown)


def hit_rate(returns: pd.Series) -> float:
    """Fraction of periods with positive returns."""

    clean = _clean_returns(returns)
    if clean.empty:
        return float("nan")
    return float((clean > 0).mean())


def summarize_performance(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """Summarize strategy and optional benchmark performance."""

    def stats(series: pd.Series) -> dict[str, float]:
        equity = (1.0 + _clean_returns(series)).cumprod()
        return {
            "cagr": cagr(series),
            "annualized_volatility": annualized_volatility(series),
            "sharpe_ratio": sharpe_ratio(series),
            "max_drawdown": max_drawdown(equity),
            "calmar_ratio": calmar_ratio(series),
            "hit_rate": hit_rate(series),
        }

    summary = {"strategy": stats(returns)}
    if benchmark_returns is not None:
        summary["benchmark"] = stats(benchmark_returns)
    return pd.DataFrame(summary)
