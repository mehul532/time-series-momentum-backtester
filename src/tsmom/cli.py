"""Command-line interface for the Time-Series Momentum Backtester."""

from __future__ import annotations

from pathlib import Path

import typer


app = typer.Typer(help="Run time-series momentum research workflows.")


@app.command()
def run_spy(
    start: str = "2000-01-01",
    lookback: int = 252,
    cost_bps: float = 5.0,
) -> None:
    """Run the SPY 12-month momentum baseline."""

    from .data import download_prices
    from .signals import trailing_return_signal
    from .backtest import backtest_single_asset
    from .metrics import summarize_performance
    from .plotting import plot_drawdowns, plot_equity_curves, plot_signal_over_price

    prices = download_prices("SPY", start=start)
    close = prices["SPY"] if "SPY" in prices.columns else prices.iloc[:, 0]
    signal = trailing_return_signal(close, lookback=lookback)
    results = backtest_single_asset(close, signal, cost_bps=cost_bps)

    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    summarize_performance(results["strategy_return"], results["benchmark_return"]).to_csv("reports/spy_metrics.csv")
    plot_equity_curves(results, "reports/figures/spy_equity_curve.png")
    plot_drawdowns(results, "reports/figures/spy_drawdown.png")
    plot_signal_over_price(close, results["position"], "reports/figures/spy_signal.png")
    typer.echo("Saved SPY baseline outputs under reports/.")


@app.command()
def sweep(
    start: str = "2000-01-01",
    ticker: str = "SPY",
) -> None:
    """Run a lookback and transaction-cost parameter sweep."""

    from .data import download_prices
    from .plotting import plot_parameter_heatmap
    from .robustness import parameter_sweep

    prices = download_prices(ticker, start=start)
    close = prices[ticker] if ticker in prices.columns else prices.iloc[:, 0]
    results = parameter_sweep(close, lookbacks=[21, 63, 126, 189, 252], cost_bps_values=[0, 5, 10, 25])
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    results.to_csv("reports/parameter_sweep.csv", index=False)
    plot_parameter_heatmap(results, "reports/figures/parameter_heatmap.png")
    typer.echo("Saved parameter sweep outputs under reports/.")


@app.command()
def multi_asset(
    start: str = "2005-01-01",
    lookback: int = 252,
    cost_bps: float = 5.0,
) -> None:
    """Run the ETF multi-asset trend-following portfolio."""

    from .data import download_prices
    from .metrics import summarize_performance
    from .plotting import plot_equity_curves, plot_multi_asset_exposures
    from .portfolio import multi_asset_tsmom

    universe = ["SPY", "QQQ", "IWM", "TLT", "IEF", "GLD", "DBC", "EFA", "EEM", "VNQ"]
    prices = download_prices(universe, start=start)
    output = multi_asset_tsmom(prices, lookback=lookback, cost_bps=cost_bps)
    results = output["results"]
    weights = output["weights"]
    assert hasattr(results, "to_csv") and hasattr(weights, "to_csv")

    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    summarize_performance(results["strategy_return"], results["benchmark_return"]).to_csv(
        "reports/multi_asset_metrics.csv"
    )
    plot_equity_curves(results, "reports/figures/multi_asset_equity_curve.png")
    plot_multi_asset_exposures(weights, "reports/figures/multi_asset_exposures.png")
    typer.echo("Saved multi-asset outputs under reports/.")


if __name__ == "__main__":
    app()
