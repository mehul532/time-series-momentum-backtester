"""Command-line interface for the Time-Series Momentum Backtester."""

from __future__ import annotations

import argparse
from pathlib import Path


def run_spy(args: argparse.Namespace) -> None:
    """Run the SPY 12-month momentum baseline."""

    from .data import download_prices
    from .signals import trailing_return_signal
    from .backtest import backtest_single_asset
    from .metrics import summarize_performance
    from .plotting import plot_drawdowns, plot_equity_curves, plot_signal_over_price

    prices = download_prices("SPY", start=args.start)
    close = prices["SPY"] if "SPY" in prices.columns else prices.iloc[:, 0]
    signal = trailing_return_signal(close, lookback=args.lookback)
    results = backtest_single_asset(close, signal, cost_bps=args.cost_bps)

    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    summarize_performance(results["strategy_return"], results["benchmark_return"]).to_csv("reports/spy_metrics.csv")
    plot_equity_curves(results, "reports/figures/spy_equity_curve.png")
    plot_drawdowns(results, "reports/figures/spy_drawdown.png")
    plot_signal_over_price(close, results["position"], "reports/figures/spy_signal.png")
    print("Saved SPY baseline outputs under reports/.")


def sweep(args: argparse.Namespace) -> None:
    """Run a lookback and transaction-cost parameter sweep."""

    from .data import download_prices
    from .plotting import plot_parameter_heatmap
    from .robustness import parameter_sweep

    ticker = args.ticker.upper()
    prices = download_prices(ticker, start=args.start)
    close = prices[ticker] if ticker in prices.columns else prices.iloc[:, 0]
    results = parameter_sweep(close, lookbacks=[21, 63, 126, 189, 252], cost_bps_values=[0, 5, 10, 25])
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    results.to_csv("reports/parameter_sweep.csv", index=False)
    plot_parameter_heatmap(results, "reports/figures/parameter_heatmap.png")
    print("Saved parameter sweep outputs under reports/.")


def multi_asset(args: argparse.Namespace) -> None:
    """Run the ETF multi-asset trend-following portfolio."""

    from .data import download_prices
    from .metrics import summarize_performance
    from .plotting import plot_equity_curves, plot_multi_asset_exposures
    from .portfolio import multi_asset_tsmom

    universe = ["SPY", "QQQ", "IWM", "TLT", "IEF", "GLD", "DBC", "EFA", "EEM", "VNQ"]
    prices = download_prices(universe, start=args.start)
    output = multi_asset_tsmom(prices, lookback=args.lookback, cost_bps=args.cost_bps)
    results = output["results"]
    weights = output["weights"]
    assert hasattr(results, "to_csv") and hasattr(weights, "to_csv")

    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    summarize_performance(results["strategy_return"], results["benchmark_return"]).to_csv(
        "reports/multi_asset_metrics.csv"
    )
    plot_equity_curves(results, "reports/figures/multi_asset_equity_curve.png")
    plot_multi_asset_exposures(weights, "reports/figures/multi_asset_exposures.png")
    print("Saved multi-asset outputs under reports/.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run time-series momentum research workflows.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    spy_parser = subparsers.add_parser("run-spy", help="Run the SPY 12-month momentum baseline.")
    spy_parser.add_argument("--start", default="2000-01-01", help="Download start date.")
    spy_parser.add_argument("--lookback", type=int, default=252, help="Momentum lookback in trading days.")
    spy_parser.add_argument("--cost-bps", type=float, default=5.0, help="Cost in basis points per position change.")
    spy_parser.set_defaults(func=run_spy)

    sweep_parser = subparsers.add_parser("sweep", help="Run a lookback and transaction-cost sweep.")
    sweep_parser.add_argument("--start", default="2000-01-01", help="Download start date.")
    sweep_parser.add_argument("--ticker", default="SPY", help="Ticker to test.")
    sweep_parser.set_defaults(func=sweep)

    multi_parser = subparsers.add_parser("multi-asset", help="Run the ETF multi-asset trend-following portfolio.")
    multi_parser.add_argument("--start", default="2005-01-01", help="Download start date.")
    multi_parser.add_argument("--lookback", type=int, default=252, help="Momentum lookback in trading days.")
    multi_parser.add_argument("--cost-bps", type=float, default=5.0, help="Cost in basis points per rebalance turnover.")
    multi_parser.set_defaults(func=multi_asset)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
