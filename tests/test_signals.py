import pandas as pd

from tsmom.signals import combine_signals, shift_signal, trailing_return_signal


def test_trailing_return_signal_uses_past_lookback_return():
    close = pd.Series([100.0, 110.0, 90.0, 120.0])

    signal = trailing_return_signal(close, lookback=2)

    expected = pd.Series([0, 0, 0, 1])
    pd.testing.assert_series_equal(signal, expected)


def test_shift_signal_moves_signal_forward_one_day():
    raw = pd.Series([1, 0, 1])

    shifted = shift_signal(raw)

    expected = pd.Series([0, 1, 0])
    pd.testing.assert_series_equal(shifted, expected)


def test_combine_signals_majority_rule():
    first = pd.Series([1, 1, 0])
    second = pd.Series([1, 0, 0])
    third = pd.Series([0, 1, 0])

    combined = combine_signals([first, second, third], method="majority")

    expected = pd.Series([1, 1, 0])
    pd.testing.assert_series_equal(combined, expected)
