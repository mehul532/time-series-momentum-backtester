"""Parameter robustness and walk-forward testing helpers."""

from __future__ import annotations

import pandas as pd

from .backtest import backtest_single_asset
from .metrics import summarize_performance
from .signals import trailing_return_signal


def parameter_sweep(
    close: pd.Series,
    lookbacks: list[int],
    cost_bps_values: list[float],
) -> pd.DataFrame:
    """Evaluate strategy performance across lookbacks and transaction costs."""

    rows: list[dict[str, float]] = []
    for lookback in lookbacks:
        signal = trailing_return_signal(close, lookback=lookback)
        for cost_bps in cost_bps_values:
            result = backtest_single_asset(close, signal, cost_bps=cost_bps)
            summary = summarize_performance(
                result["strategy_return"],
                result["benchmark_return"],
            )
            row = {
                "lookback": lookback,
                "cost_bps": cost_bps,
                "final_equity": float(result["strategy_equity"].iloc[-1]),
            }
            row.update(summary["strategy"].to_dict())
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_parameter_grid(results: pd.DataFrame) -> pd.DataFrame:
    """Pivot parameter-sweep Sharpe ratios for table and heatmap output."""

    required = {"lookback", "cost_bps", "sharpe_ratio"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return results.pivot(index="lookback", columns="cost_bps", values="sharpe_ratio")


def walk_forward_parameter_test(
    close: pd.Series,
    train_years: int = 5,
    test_years: int = 1,
    lookbacks: list[int] | None = None,
) -> pd.DataFrame:
    """Choose the best training lookback, then evaluate it out of sample."""

    if lookbacks is None:
        lookbacks = [63, 126, 189, 252]
    close = close.dropna().sort_index()
    train_size = train_years * 252
    test_size = test_years * 252
    rows: list[dict[str, float | int | pd.Timestamp]] = []

    start = 0
    while start + train_size + test_size <= len(close):
        train = close.iloc[start : start + train_size]
        test = close.iloc[start + train_size : start + train_size + test_size]

        train_scores = []
        for lookback in lookbacks:
            signal = trailing_return_signal(train, lookback=lookback)
            result = backtest_single_asset(train, signal)
            score = summarize_performance(result["strategy_return"]).loc["sharpe_ratio", "strategy"]
            train_scores.append((lookback, score))
        best_lookback = max(train_scores, key=lambda item: item[1] if pd.notna(item[1]) else -float("inf"))[0]

        combined = close.loc[train.index[0] : test.index[-1]]
        combined_signal = trailing_return_signal(combined, lookback=best_lookback)
        combined_result = backtest_single_asset(combined, combined_signal)
        test_result = combined_result.loc[test.index]
        test_summary = summarize_performance(test_result["strategy_return"]).loc[:, "strategy"]
        rows.append(
            {
                "train_start": train.index[0],
                "train_end": train.index[-1],
                "test_start": test.index[0],
                "test_end": test.index[-1],
                "best_lookback": best_lookback,
                "test_cagr": float(test_summary["cagr"]),
                "test_sharpe": float(test_summary["sharpe_ratio"]),
                "test_max_drawdown": float(test_summary["max_drawdown"]),
            }
        )
        start += test_size

    return pd.DataFrame(rows)
