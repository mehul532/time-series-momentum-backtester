import pandas as pd
import pytest

from tsmom.metrics import cagr, max_drawdown, sharpe_ratio


def test_max_drawdown_calculation():
    equity = pd.Series([1.0, 1.2, 0.9, 1.5])

    assert max_drawdown(equity) == pytest.approx(-0.25)


def test_cagr_for_constant_daily_return_is_reasonable():
    returns = pd.Series([0.01] * 252)

    assert cagr(returns) == pytest.approx((1.01**252) - 1.0)


def test_sharpe_ratio_positive_for_positive_mean_returns():
    returns = pd.Series([0.01, -0.005, 0.006, 0.002] * 100)

    assert sharpe_ratio(returns) > 0
