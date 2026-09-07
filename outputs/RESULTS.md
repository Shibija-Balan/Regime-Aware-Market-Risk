# Empirical analysis results

Sample: 2015-01-05 to 2026-09-04 (2935 daily observations)

## Portfolio

- Annualised Return: 0.0579
- Annualised Volatility: 0.0793
- Sharpe Zero Rf: 0.7294
- Max Drawdown: -0.2187

## Risk metrics

- historical_var_95: 0.7794%
- historical_var_99: 1.2792%
- historical_es_95: 1.1319%
- historical_es_99: 1.7937%
- parametric_var_95: 0.7984%
- parametric_var_99: 1.1390%

## 99% VaR backtests

- historical_99: 39 exceptions / 2685 observations (1.45%); Kupiec p-value 0.0273
- ewma_99: 52 exceptions / 2875 observations (1.81%); Kupiec p-value 0.0001

## Regime result

- Historical-VaR breaches occurring in the stressed regime: 16/39 (41.0%)

## Hypothetical scenarios

- risk_off: -3.00%
- inflation_rates_shock: -4.80%
- sterling_shock: -2.20%

## Notes

These are empirical outputs from public ETF data and are not production risk estimates.