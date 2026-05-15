"""Run an equal-weight active ETF time-series momentum portfolio."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_UNIVERSE = ["SPY", "QQQ", "IWM", "TLT", "IEF", "GLD", "DBC", "EFA", "EEM", "VNQ"]


def run(start: str, lookback: int, cost_bps: float, tickers: list[str]):
    from tsmom.data import download_prices
    from tsmom.metrics import summarize_performance
    from tsmom.plotting import plot_equity_curves, plot_multi_asset_exposures
    from tsmom.portfolio import multi_asset_tsmom

    prices = download_prices(tickers, start=start)
    output = multi_asset_tsmom(prices, lookback=lookback, cost_bps=cost_bps)
    results = output["results"]
    weights = output["weights"]

    reports_dir = ROOT / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    summarize_performance(results["strategy_return"], results["benchmark_return"]).to_csv(
        reports_dir / "multi_asset_metrics.csv"
    )
    plot_equity_curves(results, figures_dir / "multi_asset_equity_curve.png")
    plot_multi_asset_exposures(weights, figures_dir / "multi_asset_exposures.png")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multi-asset ETF trend-following portfolio.")
    parser.add_argument("--start", default="2005-01-01", help="Download start date.")
    parser.add_argument("--lookback", type=int, default=252, help="Momentum lookback in trading days.")
    parser.add_argument("--cost-bps", type=float, default=5.0, help="Cost in basis points per rebalance turnover.")
    parser.add_argument(
        "--tickers",
        default=",".join(DEFAULT_UNIVERSE),
        help="Comma-separated ETF universe.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = [ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()]
    run(start=args.start, lookback=args.lookback, cost_bps=args.cost_bps, tickers=tickers)
    print("Saved multi-asset metrics and figures under reports/.")


if __name__ == "__main__":
    main()
