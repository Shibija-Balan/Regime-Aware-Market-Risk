import numpy as np
import pandas as pd
from market_risk.backtesting import kupiec_pof_test, var_exceptions
from market_risk.risk_metrics import historical_expected_shortfall, historical_var, parametric_var


def test_historical_var_and_es_are_positive_losses():
    returns = pd.Series([-0.10, -0.05, -0.01, 0.00, 0.02, 0.03, 0.04, 0.05])
    var_75 = historical_var(returns, 0.75)
    es_75 = historical_expected_shortfall(returns, 0.75)
    assert var_75 > 0
    assert es_75 >= var_75


def test_parametric_var_increases_with_confidence():
    returns = pd.Series(np.linspace(-0.03, 0.03, 101))
    assert parametric_var(returns, 0.99) > parametric_var(returns, 0.95)


def test_var_exception_definition():
    returns = pd.Series([-0.01, -0.03, 0.01])
    var = pd.Series([0.02, 0.02, 0.02])
    assert var_exceptions(returns, var).tolist() == [False, True, False]


def test_kupiec_output_is_well_formed():
    result = kupiec_pof_test(pd.Series([False] * 98 + [True] * 2), 0.99)
    assert result["observations"] == 100.0
    assert result["exceptions"] == 2.0
    assert 0.0 <= result["p_value"] <= 1.0
