"""Signal construction utilities.

Trading functions in this project shift signals before applying returns. If
you use these signal outputs directly, call :func:`shift_signal` first so the
position for today's return is based only on information known before today.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd


PandasObject = pd.Series | pd.DataFrame


def trailing_return_signal(
    close: PandasObject,
    lookback: int = 252,
    threshold: float = 0.0,
) -> PandasObject:
    """Return a binary time-series momentum signal from past returns.

    The default rule is ``1`` when ``close / close.shift(lookback) - 1`` is
    greater than ``threshold`` and ``0`` otherwise. The returned signal is not a
    tradable position yet; it must be shifted by at least one trading day before
    it is multiplied by same-day asset returns.
    """

    if lookback <= 0:
        raise ValueError("lookback must be positive.")
    momentum = close / close.shift(lookback) - 1.0
    return (momentum > threshold).astype(int)


def moving_average_signal(
    close: PandasObject,
    short_window: int = 50,
    long_window: int = 200,
) -> PandasObject:
    """Return ``1`` when a short moving average is above a long moving average.

    As with trailing-return signals, shift this output before trading to avoid
    using today's close to trade today's return.
    """

    if short_window <= 0 or long_window <= 0:
        raise ValueError("moving-average windows must be positive.")
    if short_window >= long_window:
        raise ValueError("short_window must be smaller than long_window.")

    short_ma = close.rolling(short_window, min_periods=short_window).mean()
    long_ma = close.rolling(long_window, min_periods=long_window).mean()
    valid = short_ma.notna() & long_ma.notna()
    return ((short_ma > long_ma) & valid).astype(int)


def combine_signals(
    signals: Iterable[PandasObject] | Mapping[str, PandasObject],
    method: str = "majority",
) -> PandasObject:
    """Combine binary signals using majority, any, unanimous, or average rules."""

    signal_list = list(signals.values()) if isinstance(signals, Mapping) else list(signals)
    if not signal_list:
        raise ValueError("At least one signal is required.")

    total: PandasObject | None = None
    counts: PandasObject | None = None
    for signal in signal_list:
        numeric = signal.astype(float)
        present = signal.notna().astype(float)
        total = numeric if total is None else total.add(numeric, fill_value=0.0)
        counts = present if counts is None else counts.add(present, fill_value=0.0)

    assert total is not None and counts is not None
    score = total / counts.replace(0.0, np.nan)

    if method == "average":
        return score.fillna(0.0)
    if method == "majority":
        return (score > 0.5).fillna(False).astype(int)
    if method == "any":
        return (total > 0).fillna(False).astype(int)
    if method == "unanimous":
        return ((counts > 0) & (total == counts)).fillna(False).astype(int)
    raise ValueError("method must be one of: majority, any, unanimous, average.")


def shift_signal(signal: PandasObject, lag: int = 1) -> PandasObject:
    """Shift a raw signal into a tradable position.

    A lag of one means the signal observed after today's close is first used for
    the next trading day's return. Missing leading positions are filled with
    zero, representing cash exposure.
    """

    if lag < 0:
        raise ValueError("lag must be non-negative.")
    shifted = signal.shift(lag).fillna(0)
    if isinstance(signal, pd.DataFrame):
        return shifted.astype(signal.dtypes.to_dict(), errors="ignore")
    return shifted.astype(signal.dtype, errors="ignore")
