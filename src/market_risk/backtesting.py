"""Walk-forward VaR backtesting."""

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from .risk_metrics import historical_var


def rolling_historical_var(returns: pd.Series, window: int = 250, confidence: float = 0.99) -> pd.Series:
    s = pd.Series(returns, dtype=float).dropna()
    if window < 20 or len(s) <= window:
        raise ValueError("Return history must be longer than a window of at least 20 observations.")
    forecasts = pd.Series(index=s.index, dtype=float, name="historical_var")
    for i in range(window, len(s)):
        forecasts.iloc[i] = historical_var(s.iloc[i-window:i], confidence)
    return forecasts


def rolling_ewma_var(returns: pd.Series, confidence: float = 0.99, lam: float = 0.94, min_periods: int = 60) -> pd.Series:
    s = pd.Series(returns, dtype=float).dropna()
    if not 0.0 < confidence < 1.0:
        raise ValueError("Confidence must lie strictly between 0 and 1.")
    if not 0.0 < lam < 1.0:
        raise ValueError("EWMA lambda must lie strictly between 0 and 1.")
    if min_periods < 20 or len(s) <= min_periods:
        raise ValueError("Insufficient observations for EWMA backtesting.")
    forecasts = pd.Series(index=s.index, dtype=float, name="ewma_var")
    variance = float(s.iloc[:min_periods].var(ddof=1))
    z = float(norm.ppf(confidence))
    for i in range(min_periods, len(s)):
        r_prev = float(s.iloc[i-1])
        variance = lam * variance + (1.0 - lam) * r_prev**2
        forecasts.iloc[i] = z * np.sqrt(max(variance, 0.0))
    return forecasts


def var_exceptions(returns: pd.Series, var_forecasts: pd.Series) -> pd.Series:
    aligned = pd.concat([pd.Series(returns, dtype=float).rename("return"), var_forecasts.rename("var")], axis=1).dropna()
    out = aligned["return"] < -aligned["var"]
    out.name = "exception"
    return out


def kupiec_pof_test(exceptions: pd.Series, confidence: float = 0.99) -> dict[str, float]:
    if not 0.0 < confidence < 1.0:
        raise ValueError("Confidence must lie strictly between 0 and 1.")
    obs = pd.Series(exceptions).dropna().astype(bool)
    n = int(len(obs))
    if n == 0:
        raise ValueError("No backtest observations supplied.")
    x = int(obs.sum())
    p = 1.0 - confidence
    phat = x / n
    eps = np.finfo(float).eps
    p0 = float(np.clip(p, eps, 1.0-eps))
    p1 = float(np.clip(phat, eps, 1.0-eps))
    ll0 = (n-x)*np.log(1-p0) + x*np.log(p0)
    ll1 = (n-x)*np.log(1-p1) + x*np.log(p1)
    lr = float(-2.0*(ll0-ll1))
    return {
        "observations": float(n),
        "exceptions": float(x),
        "expected_exception_rate": float(p),
        "observed_exception_rate": float(phat),
        "lr_stat": lr,
        "p_value": float(chi2.sf(lr, 1)),
    }
