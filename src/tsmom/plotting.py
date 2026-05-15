"""Matplotlib plotting helpers for reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save(fig: plt.Figure, output_path: str | Path | None) -> plt.Figure:
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_equity_curves(results_df: pd.DataFrame, output_path: str | Path | None = None) -> plt.Figure:
    """Plot strategy and benchmark equity curves."""

    fig, ax = plt.subplots(figsize=(10, 5))
    results_df[["strategy_equity", "benchmark_equity"]].plot(ax=ax, linewidth=1.8)
    ax.set_title("Equity Curves")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("")
    ax.grid(True, alpha=0.3)
    ax.legend(["Strategy", "Benchmark"])
    return _save(fig, output_path)


def plot_drawdowns(results_df: pd.DataFrame, output_path: str | Path | None = None) -> plt.Figure:
    """Plot drawdowns for strategy and benchmark equity curves."""

    drawdowns = pd.DataFrame(index=results_df.index)
    for column in ("strategy_equity", "benchmark_equity"):
        equity = results_df[column]
        drawdowns[column] = equity / equity.cummax() - 1.0

    fig, ax = plt.subplots(figsize=(10, 4))
    drawdowns.plot(ax=ax, linewidth=1.5)
    ax.set_title("Drawdowns")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("")
    ax.grid(True, alpha=0.3)
    ax.legend(["Strategy", "Benchmark"])
    return _save(fig, output_path)


def plot_signal_over_price(
    close: pd.Series,
    signal: pd.Series,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot close price with the shifted tradable signal."""

    fig, ax_price = plt.subplots(figsize=(10, 5))
    close.plot(ax=ax_price, color="black", linewidth=1.4)
    ax_price.set_title("Price and Momentum Signal")
    ax_price.set_ylabel("Adjusted Close")
    ax_price.set_xlabel("")
    ax_price.grid(True, alpha=0.25)

    ax_signal = ax_price.twinx()
    signal.reindex(close.index).plot(ax=ax_signal, color="tab:green", alpha=0.35, drawstyle="steps-post")
    ax_signal.set_ylabel("Signal / Position")
    ax_signal.set_ylim(-0.05, 1.05)
    return _save(fig, output_path)


def plot_parameter_heatmap(
    summary_df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a lookback-by-cost heatmap from raw or pivoted sweep results."""

    if {"lookback", "cost_bps", "sharpe_ratio"}.issubset(summary_df.columns):
        heatmap_data = summary_df.pivot(index="lookback", columns="cost_bps", values="sharpe_ratio")
    else:
        heatmap_data = summary_df

    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(heatmap_data.values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels([str(col) for col in heatmap_data.columns])
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels([str(idx) for idx in heatmap_data.index])
    for row_idx, lookback in enumerate(heatmap_data.index):
        for col_idx, cost_bps in enumerate(heatmap_data.columns):
            value = heatmap_data.loc[lookback, cost_bps]
            if pd.notna(value):
                ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xlabel("Transaction Cost (bps)")
    ax.set_ylabel("Lookback (days)")
    ax.set_title("Parameter Robustness: Sharpe Ratio")
    fig.colorbar(image, ax=ax, label="Sharpe Ratio")
    return _save(fig, output_path)


def plot_multi_asset_exposures(
    weights_df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot stacked active ETF weights over time."""

    fig, ax = plt.subplots(figsize=(11, 5))
    weights_df.plot.area(ax=ax, linewidth=0.0, alpha=0.85)
    ax.set_title("Multi-Asset Trend-Following Exposures")
    ax.set_ylabel("Portfolio Weight")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    return _save(fig, output_path)
