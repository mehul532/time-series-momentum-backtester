"""Run lookback and transaction-cost robustness tests for SPY."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_LOOKBACKS = [21, 63, 126, 189, 252]
DEFAULT_COSTS = [0, 5, 10, 25]


def run(start: str, ticker: str, lookbacks: list[int], cost_bps_values: list[float]):
    from tsmom.data import download_prices
    from tsmom.plotting import plot_parameter_heatmap
    from tsmom.robustness import parameter_sweep

    prices = download_prices(ticker, start=start)
    close = prices[ticker] if ticker in prices.columns else prices.iloc[:, 0]
    results = parameter_sweep(close, lookbacks=lookbacks, cost_bps_values=cost_bps_values)

    reports_dir = ROOT / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(reports_dir / "parameter_sweep.csv", index=False)
    plot_parameter_heatmap(results, figures_dir / "parameter_heatmap.png")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a momentum lookback/cost parameter sweep.")
    parser.add_argument("--ticker", default="SPY", help="Ticker to test.")
    parser.add_argument("--start", default="2000-01-01", help="Download start date.")
    parser.add_argument(
        "--lookbacks",
        default="21,63,126,189,252",
        help="Comma-separated lookbacks in trading days.",
    )
    parser.add_argument(
        "--cost-bps",
        default="0,5,10,25",
        help="Comma-separated transaction costs in basis points.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lookbacks = [int(value) for value in args.lookbacks.split(",") if value]
    cost_bps_values = [float(value) for value in args.cost_bps.split(",") if value]
    run(
        start=args.start,
        ticker=args.ticker.upper(),
        lookbacks=lookbacks,
        cost_bps_values=cost_bps_values,
    )
    print("Saved parameter sweep grid and heatmap under reports/.")


if __name__ == "__main__":
    main()
