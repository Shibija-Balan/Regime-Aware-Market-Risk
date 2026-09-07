import numpy as np
import pandas as pd
from market_risk.portfolio import component_volatility_contributions, portfolio_returns
from market_risk.stress_testing import reverse_stress_test, scenario_portfolio_return


def test_portfolio_returns_equal_weight():
    returns = pd.DataFrame({"A": [0.01, 0.02], "B": [0.03, -0.02]})
    result = portfolio_returns(returns, {"A": 0.5, "B": 0.5})
    np.testing.assert_allclose(result.to_numpy(), [0.02, 0.00])


def test_component_volatility_percentages_sum_to_one():
    covariance = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=["A", "B"], columns=["A", "B"])
    out = component_volatility_contributions(covariance, {"A": 0.5, "B": 0.5})
    assert np.isclose(out["pct_of_portfolio_volatility"].sum(), 1.0)


def test_scenario_portfolio_return():
    assert np.isclose(scenario_portfolio_return({"A": -0.10, "B": 0.04}, {"A": 0.5, "B": 0.5}), -0.03)


def test_reverse_stress_reaches_target_loss():
    covariance = pd.DataFrame([[0.0004, 0.0001], [0.0001, 0.0009]], index=["A", "B"], columns=["A", "B"])
    weights = {"A": 0.5, "B": 0.5}
    shocks = reverse_stress_test(covariance, weights, target_loss=0.05)
    achieved_loss = -float(pd.Series(weights) @ shocks)
    assert achieved_loss >= 0.05 - 1e-8
