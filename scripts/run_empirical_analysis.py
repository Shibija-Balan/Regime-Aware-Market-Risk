"""Run the reproducible empirical analysis and write report outputs."""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from market_risk.backtesting import (
    kupiec_pof_test,
    rolling_ewma_var,
    rolling_historical_var,
    var_exceptions,
)
from market_risk.config import (
    DEFAULT_EWMA_LAMBDA,
    DEFAULT_REGIME_WINDOW,
    DEFAULT_START_DATE,
    DEFAULT_VAR_WINDOW,
    DEFAULT_WEIGHTS,
)
from market_risk.data import download_adjusted_close, simple_returns
from market_risk.portfolio import component_volatility_contributions, portfolio_returns
from market_risk.regimes import expanding_percentile_score, label_regimes, rolling_regime_features
from market_risk.risk_metrics import historical_expected_shortfall, historical_var, parametric_var
from market_risk.stress_testing import historical_worst_window, reverse_stress_test, scenario_portfolio_return


OUT = Path("outputs")
FIG = OUT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)


def annualised_summary(port: pd.Series) -> dict[str, float]:
    ann_return = (1 + port).prod() ** (252 / len(port)) - 1
    ann_vol = port.std(ddof=1) * np.sqrt(252)
    wealth = (1 + port).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    return {
        "annualised_return": float(ann_return),
        "annualised_volatility": float(ann_vol),
        "sharpe_zero_rf": float(sharpe),
        "max_drawdown": float(drawdown.min()),
    }


def save_plot(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    prices = download_adjusted_close(list(DEFAULT_WEIGHTS), DEFAULT_START_DATE)
    returns = simple_returns(prices)
    port = portfolio_returns(returns, DEFAULT_WEIGHTS)

    summary: dict[str, object] = {
        "sample_start": str(returns.index.min().date()),
        "sample_end": str(returns.index.max().date()),
        "observations": int(len(returns)),
        "weights": dict(DEFAULT_WEIGHTS),
        "portfolio": annualised_summary(port),
        "risk_metrics": {
            "historical_var_95": historical_var(port, 0.95),
            "historical_var_99": historical_var(port, 0.99),
            "historical_es_95": historical_expected_shortfall(port, 0.95),
            "historical_es_99": historical_expected_shortfall(port, 0.99),
            "parametric_var_95": parametric_var(port, 0.95),
            "parametric_var_99": parametric_var(port, 0.99),
        },
    }

    hist_var = rolling_historical_var(port, DEFAULT_VAR_WINDOW, 0.99)
    ewma_var = rolling_ewma_var(port, 0.99, DEFAULT_EWMA_LAMBDA, 60)
    hist_exc = var_exceptions(port, hist_var)
    ewma_exc = var_exceptions(port, ewma_var)
    summary["backtests"] = {
        "historical_99": kupiec_pof_test(hist_exc, 0.99),
        "ewma_99": kupiec_pof_test(ewma_exc, 0.99),
    }

    features = rolling_regime_features(returns, port, DEFAULT_REGIME_WINDOW)
    stress_score = expanding_percentile_score(features, 60)
    regimes = label_regimes(stress_score)
    regime_frame = features.join(stress_score).join(regimes)

    aligned_hist = pd.DataFrame({"return": port, "var": hist_var}).dropna()
    aligned_hist["exception"] = aligned_hist["return"] < -aligned_hist["var"]
    aligned_hist = aligned_hist.join(regime_frame[["regime", "regime_stress_score"]], how="left")
    breach_by_regime = aligned_hist.groupby("regime", dropna=False)["exception"].agg(["count", "sum", "mean"])
    breach_by_regime.to_csv(OUT / "historical_var_breaches_by_regime.csv")

    summary["regime"] = {
        "stress_score_mean": float(stress_score.mean()),
        "stress_score_95pct": float(stress_score.quantile(0.95)),
        "breaches_in_stressed_regime": int(
            aligned_hist.loc[aligned_hist["regime"] == "stressed", "exception"].sum()
        ),
        "total_historical_var_breaches": int(aligned_hist["exception"].sum()),
    }

    cov = returns.cov()
    contrib = component_volatility_contributions(cov, DEFAULT_WEIGHTS)
    contrib.to_csv(OUT / "volatility_contributions.csv")

    worst_1 = historical_worst_window(port, 1)
    worst_5 = historical_worst_window(port, 5)
    worst_20 = historical_worst_window(port, 20)
    summary["historical_worst_windows"] = {
        "1_day": {k: str(v) if hasattr(v, "date") else v for k, v in worst_1.items()},
        "5_day": {k: str(v) if hasattr(v, "date") else v for k, v in worst_5.items()},
        "20_day": {k: str(v) if hasattr(v, "date") else v for k, v in worst_20.items()},
    }

    scenarios = {
        "risk_off": {"SPY": -0.12, "TLT": 0.04, "LQD": -0.06, "GLD": 0.03, "FXB": -0.04},
        "inflation_rates_shock": {"SPY": -0.08, "TLT": -0.10, "LQD": -0.05, "GLD": 0.02, "FXB": -0.03},
        "sterling_shock": {"SPY": -0.03, "TLT": 0.01, "LQD": -0.01, "GLD": 0.02, "FXB": -0.10},
    }
    summary["scenarios"] = {
        name: scenario_portfolio_return(shocks, DEFAULT_WEIGHTS) for name, shocks in scenarios.items()
    }

    reverse = reverse_stress_test(cov, DEFAULT_WEIGHTS, target_loss=0.05)
    reverse.to_csv(OUT / "reverse_stress_5pct.csv", header=True)
    summary["reverse_stress_5pct"] = reverse.to_dict()

    regime_frame.to_csv(OUT / "regime_features.csv")
    pd.DataFrame({"portfolio_return": port, "historical_var_99": hist_var, "ewma_var_99": ewma_var}).to_csv(
        OUT / "var_backtest_series.csv"
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ((1 + port).cumprod()).plot(ax=ax)
    ax.set_title("Equal-weight cross-asset portfolio growth")
    ax.set_ylabel("Growth of $1")
    save_plot(fig, "portfolio_growth.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    regime_frame["regime_stress_score"].plot(ax=ax)
    ax.set_title("Regime stress score")
    ax.set_ylabel("Score")
    save_plot(fig, "regime_stress_score.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    aligned_hist["return"].plot(ax=ax, linewidth=0.8, label="Return")
    (-aligned_hist["var"]).plot(ax=ax, linewidth=1.0, label="-99% historical VaR")
    breaches = aligned_hist[aligned_hist["exception"]]
    ax.scatter(breaches.index, breaches["return"], s=20, label="VaR exception")
    ax.legend()
    ax.set_title("99% historical VaR backtest")
    save_plot(fig, "historical_var_backtest.png")

    with (OUT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# Empirical analysis results",
        "",
        f"Sample: {summary['sample_start']} to {summary['sample_end']} ({summary['observations']} daily observations)",
        "",
        "## Portfolio",
        "",
    ]
    for k, v in summary["portfolio"].items():
        lines.append(f"- {k.replace('_', ' ').title()}: {v:.4f}")
    lines += ["", "## Risk metrics", ""]
    for k, v in summary["risk_metrics"].items():
        lines.append(f"- {k}: {v:.4%}")
    lines += ["", "## 99% VaR backtests", ""]
    for model, result in summary["backtests"].items():
        lines.append(
            f"- {model}: {int(result['exceptions'])} exceptions / {int(result['observations'])} observations "
            f"({result['observed_exception_rate']:.2%}); Kupiec p-value {result['p_value']:.4f}"
        )
    lines += ["", "## Regime result", ""]
    total_b = summary["regime"]["total_historical_var_breaches"]
    stressed_b = summary["regime"]["breaches_in_stressed_regime"]
    share = stressed_b / total_b if total_b else np.nan
    lines.append(f"- Historical-VaR breaches occurring in the stressed regime: {stressed_b}/{total_b} ({share:.1%})")
    lines += ["", "## Hypothetical scenarios", ""]
    for k, v in summary["scenarios"].items():
        lines.append(f"- {k}: {v:.2%}")
    lines += ["", "## Notes", "", "These are empirical outputs from public ETF data and are not production risk estimates."]
    (OUT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
