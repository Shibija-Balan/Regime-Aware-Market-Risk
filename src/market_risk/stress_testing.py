"""Scenario and reverse stress testing."""

from collections.abc import Mapping, Sequence
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .portfolio import validate_weights


def scenario_portfolio_return(shocks: Mapping[str, float], weights: Mapping[str, float]) -> float:
    s = pd.Series(shocks, dtype=float)
    w = validate_weights(weights, s.index)
    return float(w @ s)


def historical_worst_window(port_returns: pd.Series, horizon: int = 5) -> dict[str, object]:
    s = pd.Series(port_returns, dtype=float).dropna()
    if horizon < 1 or len(s) < horizon:
        raise ValueError("Invalid horizon for available return history.")
    compounded = (1.0 + s).rolling(horizon).apply(np.prod, raw=True) - 1.0
    end_date = compounded.idxmin()
    end_loc = s.index.get_loc(end_date)
    start_date = s.index[end_loc-horizon+1]
    return {
        "start_date": start_date,
        "end_date": end_date,
        "horizon_days": horizon,
        "compounded_return": float(compounded.loc[end_date]),
    }


def reverse_stress_test(
    covariance: pd.DataFrame,
    weights: Mapping[str, float],
    target_loss: float = 0.05,
    bounds: Sequence[tuple[float, float]] | None = None,
) -> pd.Series:
    """Find the smallest covariance-scaled joint shock that reaches a loss threshold.

    When the unconstrained optimum lies inside the shock bounds, the solution has
    a closed form. Otherwise a bounded SLSQP problem is solved from a feasible
    starting point. This avoids unnecessary numerical optimisation in the common
    case and makes the result reproducible across SciPy versions.
    """
    if covariance.empty or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("Covariance matrix must be non-empty and square.")
    if list(covariance.index) != list(covariance.columns):
        raise ValueError("Covariance matrix index and columns must match.")
    if not 0.0 < target_loss < 1.0:
        raise ValueError("Target loss must be between 0 and 1.")

    w = validate_weights(weights, covariance.columns)
    sigma = covariance.to_numpy(dtype=float)
    wv = w.to_numpy(dtype=float)
    portfolio_variance = float(wv @ sigma @ wv)
    if not np.isfinite(portfolio_variance) or portfolio_variance <= 0:
        raise ValueError("Portfolio variance must be positive and finite.")

    # Minimum Mahalanobis-distance shock subject to w'x = -target_loss.
    analytical = -(target_loss / portfolio_variance) * (sigma @ wv)

    if bounds is None:
        bounds = [(-0.30, 0.30)] * len(w)
    bounds_list = list(bounds)
    if len(bounds_list) != len(w):
        raise ValueError("One shock bound is required for each asset.")

    if all(lo <= x <= hi for x, (lo, hi) in zip(analytical, bounds_list)):
        return pd.Series(analytical, index=covariance.columns, name="reverse_stress_shock")

    precision = np.linalg.pinv(sigma)

    def objective(x: np.ndarray) -> float:
        return float(0.5 * x @ precision @ x)

    def breach_constraint(x: np.ndarray) -> float:
        # scipy inequality constraints are feasible when the value is >= 0.
        return float(-target_loss - wv @ x)

    x0 = np.array(
        [np.clip(x, lo, hi) for x, (lo, hi) in zip(analytical, bounds_list)],
        dtype=float,
    )
    if breach_constraint(x0) < 0:
        lower = np.array([lo for lo, _ in bounds_list], dtype=float)
        if breach_constraint(lower) < 0:
            raise ValueError("Target loss cannot be reached within the supplied shock bounds.")
        x0 = lower

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds_list,
        constraints={"type": "ineq", "fun": breach_constraint},
        options={"ftol": 1e-10, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(f"Reverse stress optimisation failed: {result.message}")

    shocks = pd.Series(result.x, index=covariance.columns, name="reverse_stress_shock")
    if -float(w @ shocks) + 1e-8 < target_loss:
        raise RuntimeError("Optimisation returned a shock below the target loss.")
    return shocks
