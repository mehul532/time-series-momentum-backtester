"""Run the SPY 12-month time-series momentum baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

def _format_value(value: float) -> str:
    if abs(value) < 10:
        return f"{value:.2%}"
    return f"{value:.2f}"


def _summary_to_markdown(summary) -> str:
    rows = ["| Metric | Strategy | Buy-and-Hold |", "| --- | ---: | ---: |"]
    for metric in summary.index:
        strategy = _format_value(float(summary.loc[metric, "strategy"]))
        benchmark = _format_value(float(summary.loc[metric, "benchmark"]))
        rows.append(f"| {metric.replace('_', ' ').title()} | {strategy} | {benchmark} |")
    return "\n".join(rows)


def update_report_summary(summary, start: str, lookback: int, cost_bps: float) -> None:
    """Write a concise baseline report summary."""

    report_path = ROOT / "reports" / "summary.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# Time-Series Momentum Backtester Summary",
                "",
                "## SPY Baseline",
                "",
                f"- Universe: SPY",
                f"- Start date: {start}",
                f"- Lookback: {lookback} trading days",
                f"- Transaction cost: {cost_bps} bps per position change",
                "- Signal alignment: raw momentum signals are shifted by one trading day before trading.",
                "",
                _summary_to_markdown(summary),
                "",
                "Figures are saved under `reports/figures/`.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run(start: str, lookback: int, cost_bps: float):
    from tsmom.backtest import backtest_single_asset
    from tsmom.data import download_prices
    from tsmom.metrics import summarize_performance
    from tsmom.plotting import plot_drawdowns, plot_equity_curves, plot_signal_over_price
    from tsmom.signals import trailing_return_signal

    prices = download_prices("SPY", start=start)
    close = prices["SPY"] if "SPY" in prices.columns else prices.iloc[:, 0]
    signal = trailing_return_signal(close, lookback=lookback)
    results = backtest_single_asset(close, signal, cost_bps=cost_bps)
    summary = summarize_performance(results["strategy_return"], results["benchmark_return"])

    reports_dir = ROOT / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(reports_dir / "spy_metrics.csv")
    plot_equity_curves(results, figures_dir / "spy_equity_curve.png")
    plot_drawdowns(results, figures_dir / "spy_drawdown.png")
    plot_signal_over_price(close, results["position"], figures_dir / "spy_signal.png")
    update_report_summary(summary, start=start, lookback=lookback, cost_bps=cost_bps)
    return results, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SPY 12-month time-series momentum baseline.")
    parser.add_argument("--start", default="2000-01-01", help="Download start date.")
    parser.add_argument("--lookback", type=int, default=252, help="Momentum lookback in trading days.")
    parser.add_argument("--cost-bps", type=float, default=5.0, help="Cost in basis points per position change.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(start=args.start, lookback=args.lookback, cost_bps=args.cost_bps)
    print("Saved SPY baseline metrics and figures under reports/.")


if __name__ == "__main__":
    main()
