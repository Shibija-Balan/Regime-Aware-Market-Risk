"""Core market risk measures."""

import numpy as np
import pandas as pd
from scipy.stats import norm


def _clean_returns(returns: pd.Series) -> pd.Series:
    s = pd.Series(returns, dtype=float).dropna()
    if s.empty:
        raise ValueError("Return series is empty.")
    if not np.isfinite(s).all():
        raise ValueError("Return series contains non-finite values.")
    return s


def _check_confidence(confidence: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError("Confidence must lie strictly between 0 and 1.")


def historical_var(returns: pd.Series, confidence: float = 0.99) -> float:
    _check_confidence(confidence)
    s = _clean_returns(returns)
    return max(0.0, -float(s.quantile(1.0 - confidence)))


def historical_expected_shortfall(returns: pd.Series, confidence: float = 0.99) -> float:
    _check_confidence(confidence)
    s = _clean_returns(returns)
    threshold = float(s.quantile(1.0 - confidence))
    tail = s[s <= threshold]
    if tail.empty:
        raise ValueError("No observations fall in the requested tail.")
    return max(0.0, -float(tail.mean()))


def parametric_var(returns: pd.Series, confidence: float = 0.99) -> float:
    _check_confidence(confidence)
    s = _clean_returns(returns)
    sigma = float(s.std(ddof=1))
    if sigma <= 0:
        return max(0.0, -float(s.mean()))
    quantile = float(s.mean()) + float(norm.ppf(1.0 - confidence)) * sigma
    return max(0.0, -quantile)


def ewma_variance(returns: pd.Series, lam: float = 0.94) -> float:
    s = _clean_returns(returns)
    if not 0.0 < lam < 1.0:
        raise ValueError("EWMA lambda must lie strictly between 0 and 1.")
    if len(s) < 2:
        raise ValueError("At least two returns are required.")
    init_n = min(20, len(s))
    variance = float(s.iloc[:init_n].var(ddof=1))
    if not np.isfinite(variance) or variance < 0:
        raise ValueError("Unable to initialise EWMA variance.")
    for r in s.iloc[init_n:]:
        variance = lam * variance + (1.0 - lam) * float(r) ** 2
    return float(variance)


def ewma_var(returns: pd.Series, confidence: float = 0.99, lam: float = 0.94) -> float:
    _check_confidence(confidence)
    return max(0.0, float(norm.ppf(confidence)) * np.sqrt(ewma_variance(returns, lam)))
