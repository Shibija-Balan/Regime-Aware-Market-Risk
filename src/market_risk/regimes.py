"""Rolling indicators for cross-asset regime analysis."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def average_pairwise_correlation(window_returns: pd.DataFrame) -> float:
    corr = window_returns.corr().to_numpy(dtype=float)
    n = corr.shape[0]
    if n < 2:
        raise ValueError("At least two assets are required.")
    values = corr[~np.eye(n, dtype=bool)]
    return float(np.nanmean(values))


def first_pc_variance_share(window_returns: pd.DataFrame) -> float:
    clean = window_returns.dropna(how="any")
    if clean.shape[0] < 2 or clean.shape[1] < 2:
        raise ValueError("Insufficient observations for PCA.")
    scaled = StandardScaler().fit_transform(clean)
    pca = PCA(n_components=1).fit(scaled)
    return float(pca.explained_variance_ratio_[0])


def rolling_regime_features(asset_returns: pd.DataFrame, port_returns: pd.Series, window: int = 60, annualisation: int = 252) -> pd.DataFrame:
    if window < 20:
        raise ValueError("Regime window should contain at least 20 observations.")
    if not asset_returns.index.equals(port_returns.index):
        raise ValueError("Asset and portfolio return indices must match.")
    rows = []
    for end in range(window, len(asset_returns) + 1):
        aw = asset_returns.iloc[end-window:end]
        pw = port_returns.iloc[end-window:end]
        rows.append(
            {
                "date": asset_returns.index[end-1],
                "portfolio_volatility": float(pw.std(ddof=1) * np.sqrt(annualisation)),
                "average_correlation": average_pairwise_correlation(aw),
                "pc1_variance_share": first_pc_variance_share(aw),
            }
        )
    return pd.DataFrame(rows).set_index("date")


def expanding_percentile_score(features: pd.DataFrame, min_periods: int = 60) -> pd.Series:
    required = ["portfolio_volatility", "average_correlation", "pc1_variance_share"]
    missing = [c for c in required if c not in features.columns]
    if missing:
        raise ValueError(f"Missing regime feature columns: {missing}")
    score = pd.Series(index=features.index, dtype=float, name="regime_stress_score")
    for i in range(len(features)):
        history = features.iloc[:i+1]
        if len(history) < min_periods:
            continue
        current = history.iloc[-1]
        pct = [float((history[c] <= current[c]).mean()) for c in required]
        score.iloc[i] = float(np.mean(pct))
    return score


def label_regimes(score: pd.Series) -> pd.Series:
    clean = score.dropna()
    labels = pd.Series(index=score.index, dtype="object", name="regime")
    if clean.empty:
        return labels
    low, high = clean.quantile([1/3, 2/3])
    labels.loc[score <= low] = "calm"
    labels.loc[(score > low) & (score <= high)] = "transitional"
    labels.loc[score > high] = "stressed"
    return labels
