"""Portfolio construction and volatility attribution."""

from collections.abc import Mapping
import numpy as np
import pandas as pd


def validate_weights(weights: Mapping[str, float], columns: pd.Index) -> pd.Series:
    if not weights:
        raise ValueError("Portfolio weights are required.")
    w = pd.Series(weights, dtype=float)
    missing = [c for c in columns if c not in w.index]
    extra = [t for t in w.index if t not in columns]
    if missing or extra:
        raise ValueError(f"Weight/return mismatch. Missing={missing}, extra={extra}")
    if not np.isfinite(w).all():
        raise ValueError("Weights must be finite.")
    if not np.isclose(w.sum(), 1.0, atol=1e-10):
        raise ValueError("Portfolio weights must sum to 1.")
    return w.reindex(columns)


def portfolio_returns(asset_returns: pd.DataFrame, weights: Mapping[str, float]) -> pd.Series:
    if asset_returns.empty:
        raise ValueError("Asset returns are empty.")
    w = validate_weights(weights, asset_returns.columns)
    result = asset_returns.mul(w, axis=1).sum(axis=1)
    result.name = "portfolio_return"
    return result


def component_volatility_contributions(covariance: pd.DataFrame, weights: Mapping[str, float]) -> pd.DataFrame:
    if covariance.empty or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("Covariance matrix must be non-empty and square.")
    if list(covariance.index) != list(covariance.columns):
        raise ValueError("Covariance matrix index and columns must match.")
    w = validate_weights(weights, covariance.columns)
    sigma = covariance.to_numpy(dtype=float)
    wv = w.to_numpy(dtype=float)
    variance = float(wv @ sigma @ wv)
    if variance <= 0:
        raise ValueError("Portfolio variance must be positive.")
    vol = float(np.sqrt(variance))
    marginal = sigma @ wv / vol
    component = wv * marginal
    return pd.DataFrame(
        {
            "weight": wv,
            "component_volatility": component,
            "pct_of_portfolio_volatility": component / vol,
        },
        index=covariance.columns,
    )
