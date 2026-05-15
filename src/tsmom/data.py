"""Data access helpers for adjusted-close price histories."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


CACHE_DIR = Path("data/raw")


def _normalize_tickers(tickers: str | Iterable[str]) -> list[str]:
    if isinstance(tickers, str):
        return [tickers.upper()]
    return [ticker.upper() for ticker in tickers]


def _cache_path(tickers: list[str], start: str, end: str | None, auto_adjust: bool) -> Path:
    ticker_key = "-".join(tickers)
    end_key = end or "latest"
    adjust_key = "autoadj" if auto_adjust else "raw"
    filename = f"{ticker_key}_{start}_{end_key}_{adjust_key}.csv".replace("/", "-")
    return CACHE_DIR / filename


def download_prices(
    tickers: str | Iterable[str],
    start: str,
    end: str | None = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download adjusted close prices with a local CSV cache.

    Parameters
    ----------
    tickers:
        One ticker or an iterable of tickers accepted by Yahoo Finance.
    start, end:
        Date strings passed through to ``yfinance.download``.
    auto_adjust:
        Passed to yfinance. When true, Yahoo's ``Close`` column is already
        adjusted for dividends and splits.
    """

    ticker_list = _normalize_tickers(tickers)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path(ticker_list, start, end, auto_adjust)
    if cache_path.exists():
        return load_prices(cache_path)

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("Install yfinance to download market data.") from exc

    ticker_arg: str | list[str] = ticker_list[0] if len(ticker_list) == 1 else ticker_list
    raw = yf.download(
        ticker_arg,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        raise ValueError(f"No price data returned for {ticker_list}.")

    close = get_adjusted_close(raw)
    if isinstance(close, pd.Series):
        close = close.to_frame(ticker_list[0])
    close = close.sort_index().dropna(how="all")
    close.index = pd.to_datetime(close.index)
    close.index.name = "date"
    save_prices(close, cache_path)
    return close


def load_prices(path: str | Path) -> pd.DataFrame:
    """Load cached close prices from CSV."""

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df.sort_index()


def save_prices(df: pd.DataFrame | pd.Series, path: str | Path) -> None:
    """Save close prices to CSV, creating parent directories when needed."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = df.to_frame() if isinstance(df, pd.Series) else df
    data.to_csv(output_path, index_label="date")


def get_adjusted_close(data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Extract adjusted close prices from common yfinance output shapes.

    For ``auto_adjust=True`` downloads, yfinance provides adjusted prices in
    the ``Close`` field. For raw downloads, this function prefers
    ``Adj Close`` and falls back to ``Close``.
    """

    if isinstance(data, pd.Series):
        return data

    if isinstance(data.columns, pd.MultiIndex):
        for field in ("Adj Close", "Close"):
            for level in range(data.columns.nlevels):
                if field in data.columns.get_level_values(level):
                    close = data.xs(field, level=level, axis=1)
                    return close.sort_index()
        raise KeyError("Could not find Adj Close or Close in multi-index price data.")

    for field in ("Adj Close", "Close"):
        if field in data.columns:
            close = data[field]
            if isinstance(close, pd.Series):
                close.name = data.attrs.get("ticker", close.name)
            return close.sort_index()

    numeric = data.select_dtypes("number")
    if not numeric.empty:
        return numeric.sort_index()
    raise KeyError("Could not identify adjusted close prices.")
