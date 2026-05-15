import pandas as pd
import pytest
import numpy as np

from tsmom.backtest import backtest_single_asset, compute_strategy_returns
from tsmom.portfolio import multi_asset_tsmom


def test_backtest_shifts_signal_before_applying_returns():
    close = pd.Series([100.0, 110.0, 99.0], index=pd.date_range("2020-01-01", periods=3))
    signal = pd.Series([0, 1, 0], index=close.index)

    result = backtest_single_asset(close, signal)

    assert result.loc[close.index[1], "position"] == 0
    assert result.loc[close.index[1], "strategy_return"] == 0
    assert result.loc[close.index[2], "position"] == 1
    assert result.loc[close.index[2], "strategy_return"] == pytest.approx(-0.10)


def test_transaction_costs_apply_only_when_position_changes():
    returns = pd.Series([0.0, 0.10, 0.10, 0.0])
    position = pd.Series([0.0, 1.0, 1.0, 0.0])

    strategy_returns = compute_strategy_returns(
        returns,
        position,
        cost_bps=100,
        signal_is_position=True,
    )

    expected = pd.Series([0.0, 0.09, 0.10, -0.01], name="strategy_return")
    pd.testing.assert_series_equal(strategy_returns, expected)


def test_multi_asset_active_weights_sum_to_one_when_assets_are_active():
    idx = pd.date_range("2020-01-01", periods=6)
    close = pd.DataFrame(
        {
            "AAA": [100, 101, 102, 103, 104, 105],
            "BBB": [100, 99, 98, 99, 100, 101],
        },
        index=idx,
        dtype=float,
    )

    output = multi_asset_tsmom(close, lookback=2, cost_bps=0)
    weights = output["weights"]
    active_rows = weights.sum(axis=1) > 0

    assert np.allclose(weights.loc[active_rows].sum(axis=1), 1.0)
