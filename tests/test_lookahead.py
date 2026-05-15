import pandas as pd

from tsmom.backtest import align_signal_and_returns, backtest_single_asset


def test_alignment_uses_prior_day_signal_as_position():
    close = pd.Series([100.0, 105.0, 110.0], index=pd.date_range("2021-01-01", periods=3))
    signal = pd.Series([1, 0, 1], index=close.index)

    aligned = align_signal_and_returns(close, signal)

    expected_position = pd.Series([0.0, 1.0, 0.0], index=close.index)
    pd.testing.assert_series_equal(aligned["position"], expected_position, check_names=False)


def test_strategy_does_not_use_profitable_same_day_signal_by_default():
    close = pd.Series([100.0, 120.0, 80.0], index=pd.date_range("2021-01-01", periods=3))
    same_day_signal = pd.Series([0, 1, 0], index=close.index)

    result = backtest_single_asset(close, same_day_signal)

    assert result.loc[close.index[1], "strategy_return"] == 0.0
