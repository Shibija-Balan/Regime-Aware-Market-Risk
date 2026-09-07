# Regime-Aware Market Risk

A cross-asset market risk project examining how portfolio risk changes when diversification weakens during stressed markets.

The project is built around a simple question:

> Do standard risk measures become less reliable when volatility and cross-asset dependence rise at the same time?

Rather than treating Value at Risk as the end result, the analysis combines conventional risk metrics with rolling regime indicators, backtesting and reverse stress testing.

## Portfolio

The default portfolio is intentionally simple and equally weighted across five liquid US-traded instruments so that all assets share the same trading calendar:

| Ticker | Exposure | Weight |
| --- | --- | ---: |
| SPY | US equities | 20% |
| TLT | Long-duration US Treasuries / rates | 20% |
| LQD | Investment-grade corporate credit | 20% |
| GLD | Gold / commodities | 20% |
| FXB | British pound / FX proxy | 20% |

The equal-weight choice is deliberate. The aim is to study changes in risk and dependence, not to claim an optimal portfolio allocation.

## Methods

The current implementation includes:

- historical and normal-theory VaR;
- historical Expected Shortfall;
- EWMA volatility-based VaR;
- rolling portfolio volatility;
- rolling average cross-asset correlation;
- rolling PCA concentration, measured by the variance share of the first principal component;
- a look-ahead-safe stress score built from expanding percentile ranks;
- walk-forward VaR backtesting;
- Kupiec proportion-of-failures testing;
- component contributions to portfolio volatility;
- historical worst-window analysis;
- hypothetical scenario testing; and
- reverse stress testing using covariance-scaled optimisation.

## Why the regime layer matters

A portfolio can look diversified by capital weight while becoming concentrated in risk terms. During stressed periods, volatility often rises at the same time as correlations and common-factor exposure. This project tracks three descriptive indicators:

1. **Portfolio volatility**: how unstable the combined portfolio is.
2. **Average correlation**: whether assets are moving together more strongly.
3. **PC1 variance share**: how much of standardised cross-asset movement is explained by one common principal component.

A higher value across all three suggests that diversification is providing less protection than the portfolio weights alone imply.

The regime score is descriptive rather than predictive. It is designed to help compare risk behaviour across market states, not to claim that market regimes can be forecast reliably.

## Backtesting design

VaR forecasts are generated on a one-step-ahead basis using only information available before the forecast date.

For historical VaR, each forecast uses a trailing window of prior observations. The EWMA model updates variance recursively using the previous realised return. A VaR exception occurs when the realised return falls below the negative VaR forecast.

The Kupiec test is then used to assess whether the observed exception frequency is consistent with the stated confidence level.

## Reverse stress testing

Traditional stress testing starts with a market scenario and measures the portfolio loss. Reverse stress testing starts with a loss threshold and asks which joint market shock could produce it.

The implementation minimises the Mahalanobis distance of the shock vector from zero, subject to the portfolio breaching a chosen loss threshold. The covariance matrix therefore penalises shocks that are large relative to the historical dependence structure.

This is an illustrative stress construction, not a forecast of the probability of that scenario occurring.

## Repository structure

```text
Regime-Aware-Market-Risk/
├── README.md
├── pyproject.toml
├── requirements.txt
├── notebooks/
│   └── 01_market_risk_analysis.ipynb
├── src/
│   └── market_risk/
│       ├── config.py
│       ├── data.py
│       ├── portfolio.py
│       ├── risk_metrics.py
│       ├── regimes.py
│       ├── backtesting.py
│       └── stress_testing.py
└── tests/
    ├── test_portfolio.py
    └── test_risk_metrics.py
```

## Running the project

Create an environment and install the requirements:

```bash
pip install -r requirements.txt
```

Then run the tests:

```bash
pytest -q
```

Open the analysis notebook with:

```bash
jupyter notebook notebooks/01_market_risk_analysis.ipynb
```

## Limitations

This project uses public ETF data as liquid proxies for broad risk exposures. Those proxies are not equivalent to the positions or risk systems used by a bank. Historical VaR assumes the recent empirical distribution remains informative; parametric VaR imposes a distributional approximation; EWMA is sensitive to its decay parameter; and reverse stress scenarios depend on the chosen covariance estimate and bounds.

The analysis also excludes transaction costs, liquidity effects, derivatives convexity, intraday risk, funding constraints and portfolio rebalancing. Results should therefore be interpreted as an educational market-risk study rather than a production risk model.
