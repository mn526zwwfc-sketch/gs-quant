---
name: gs-quant-research-strategies
description: "Estrategias sistemáticas, construcción de carteras y backtesting con gs_quant.research (sin sesión de Marquee): momentum de series temporales y transversal, reversión, baja volatilidad, escalado por volatilidad, paridad de riesgo, HRP, Markowitz, Black-Litterman, Kelly, backtest walk-forward neto de costos, Sharpe deflactado y regresiones de factores Fama-French. Usar cuando haya que diseñar, probar o evaluar una estrategia o una asignación de activos."
---

# gs_quant.research — Estrategias, carteras y backtesting

Funciones locales; **no requieren `GsSession`**. Cada función cita en su docstring el artículo en el que se basa.

## 0. Instalación

```bash
python -c "import gs_quant.research" 2>/dev/null || \
  pip install "git+https://github.com/mn526zwwfc-sketch/gs-quant@claude/financial-data-management-dqpqs6"
```

Demo completo con datos sintéticos o un CSV propio (fechas en la 1.ª columna, un activo por columna):

```bash
python -m gs_quant.research.demo
python -m gs_quant.research.demo --csv precios.csv --cost-bps 10
```

## 1. Flujo recomendado

```python
import pandas as pd
from gs_quant.research import (
    clean_prices, compute_returns, ledoit_wolf_covariance, risk_parity,
    rolling_allocation, backtest, time_series_momentum, volatility_scaling,
    performance_summary, sharpe_ratio, deflated_sharpe_ratio,
)

prices = clean_prices(pd.read_csv('precios.csv', index_col=0, parse_dates=True))
returns = compute_returns(prices)

# A) Asignación walk-forward: cada mes se estima SOLO con los 252 días anteriores
weights = rolling_allocation(returns, lambda r: risk_parity(ledoit_wolf_covariance(r)),
                             lookback=252, rebalance='M')
rp = backtest(returns, weights, transaction_cost_bps=10)

# B) Señal: momentum de series temporales con volatilidad objetivo del 10 %
tsmom = volatility_scaling(time_series_momentum(prices), prices, target_volatility=0.10)
trend = backtest(returns, tsmom, transaction_cost_bps=10, rebalance='M')

print(rp.summary())       # incluye annual_turnover y annual_cost_drag

# C) Comparar netos de costos y corregir por pruebas múltiples
net = pd.DataFrame({'risk_parity': rp.returns, 'tsmom': trend.returns}).dropna()
print(performance_summary(net))
trials = [sharpe_ratio(net[c]) for c in net]          # TODAS las variantes probadas
print('Deflated Sharpe:', deflated_sharpe_ratio(net['tsmom'], trials))   # exigir > 0.95
```

`backtest` aplica los pesos con `execution_lag=1` (se opera el día siguiente a la señal), deja derivar los pesos
entre rebalanceos y cobra costos sobre la rotación real. Devuelve `BacktestResult` con `returns` (netos),
`gross_returns`, `weights`, `turnover`, `costs` y `.summary()`.

## 2. Señales — `signals.py`

Todas reciben `prices` (fechas × activos) y devuelven pesos/posiciones (fechas × activos) listos para `backtest`.

| Función | Firma | Fundamento |
|---|---|---|
| `time_series_momentum` | `(prices, lookback=252, volatility_target=0.4, center_of_mass=60, periods_per_year=261)` | Moskowitz, Ooi & Pedersen (2012) |
| `cross_sectional_momentum` | `(prices, lookback=252, skip=21, quantile=0.2, long_only=False)` | Jegadeesh & Titman (1993), 12-1 meses |
| `short_term_reversal` | `(prices, lookback=21)` | Jegadeesh (1990); Lehmann (1990) |
| `moving_average_crossover` | `(prices, fast=50, slow=200, long_short=False)` | Brock, Lakonishok & LeBaron (1992) |
| `low_volatility` | `(prices, lookback=252, quantile=0.2, long_only=True)` | Ang et al. (2006); Frazzini & Pedersen (2014) |
| `volatility_scaling` | `(weights, prices, target_volatility=0.1, lookback=63, max_leverage=2.0, periods_per_year=252)` | Moreira & Muir (2017) |
| `combine_signals` | `(signals: dict[str, DataFrame], weights=None)` | Asness, Moskowitz & Pedersen (2013) |
| `ex_ante_volatility` | `(prices, center_of_mass=60, periods_per_year=261)` | |

```python
from gs_quant.research import combine_signals, cross_sectional_momentum, low_volatility
combo = combine_signals({'mom': cross_sectional_momentum(prices),
                         'lowvol': low_volatility(prices, long_only=False)})
```

## 3. Carteras — `portfolio.py`

Reciben una covarianza (usar `ledoit_wolf_covariance`) y devuelven una `Series` de pesos.

| Función | Firma | Cuándo |
|---|---|---|
| `equal_weight` | `(assets)` | Referencia difícil de batir (DeMiguel et al., 2009) |
| `inverse_volatility` | `(cov)` | |
| `risk_parity` | `(cov, risk_budget=None, ...)` | Sin pronósticos de rendimiento |
| `hierarchical_risk_parity` | `(cov, linkage_method='single')` | Muchos activos, covarianza inestable |
| `minimum_variance` | `(cov, long_only=True, max_weight=None)` | |
| `mean_variance` | `(expected_returns, cov, risk_aversion=2.5, long_only=True, max_weight=None)` | Solo con pronósticos fiables |
| `maximum_sharpe` | `(expected_returns, cov, risk_free=0.0, long_only=True, max_weight=None)` | Cartera tangente |
| `efficient_frontier` | `(expected_returns, cov, n_points=25, long_only=True, max_weight=None)` → DataFrame | |
| `black_litterman` | `(cov, market_weights, risk_aversion=2.5, views=None, view_returns=None, view_uncertainty=None, tau=0.05)` → `BlackLittermanResult` | Combinar equilibrio + opiniones |
| `kelly_weights` | `(expected_returns, cov, risk_free=0.0, fraction=0.5)` | Medio Kelly por defecto |
| `kelly_fraction` | `(win_probability, win_loss_ratio)` | Apuesta binaria |

```python
from gs_quant.research import black_litterman, maximum_sharpe
P = pd.DataFrame([[1, -1, 0, 0]], columns=cov.columns)       # opinión: activo 1 supera al 2...
bl = black_litterman(cov, market_weights, views=P, view_returns=[0.02])   # ...en un 2 %
w = maximum_sharpe(bl.posterior_returns, cov, max_weight=0.4)
```

## 4. Desempeño — `performance.py`

`sharpe_ratio(returns, risk_free=0.0, periods_per_year=None)`, `sortino_ratio`, `calmar_ratio`, `omega_ratio`,
`information_ratio(returns, benchmark)`, `max_drawdown`, `drawdown`, `annualized_return`, `annualized_volatility`.
La frecuencia se infiere del índice si `periods_per_year=None`.

- `performance_summary(returns, risk_free=0.0, benchmark=None)` — tabla con todo lo anterior + VaR/ES y PSR.
- `probabilistic_sharpe_ratio(returns, benchmark_sharpe=0.0)` — probabilidad de que el Sharpe real supere al de referencia.
- `minimum_track_record_length(returns, benchmark_sharpe=0.0, confidence=0.95)` — historia necesaria.
- `deflated_sharpe_ratio(returns, trial_sharpe_ratios)` — corrige por el número de variantes probadas.

## 5. Validación y factores

- `rolling_allocation(returns, allocator, lookback=252, rebalance='M', min_observations=None)` — walk-forward.
- `walk_forward_splits(n, train_size, test_size, expanding=False)` y
  `purged_kfold_splits(n, n_splits=5, label_horizon=0, embargo=0.01)` — índices para validar modelos de ML
  sin fuga de información (López de Prado, 2018).
- `factor_regression(returns, factors, risk_free=None, hac_lags=None)` → `alpha`, `alpha_annualized`, `betas`,
  `t_stats`, `p_values`, `r_squared` con errores Newey-West. Los factores (Mkt-RF, SMB, HML, MOM, RMW, CMA) se
  descargan de la *Kenneth R. French Data Library*; deben estar en decimales, no en porcentaje.
- `fama_macbeth(returns, characteristics: dict[str, DataFrame], lag=1, hac_lags=0)` → primas por característica.

## 6. Reglas que hay que respetar

1. Siempre neto de costos: revisar `annual_turnover` y `annual_cost_drag`.
2. Estimar solo con datos pasados (`rolling_allocation`, `execution_lag>=1`); no optimizar con toda la muestra.
3. Desconfiar de los rendimientos esperados: sin pronósticos fiables, preferir 1/N, paridad de riesgo o HRP.
4. Tras probar varias configuraciones, informar el Sharpe deflactado (> 0,95) o un estadístico t > 3.
5. Un backtest es una simulación histórica; no garantiza resultados futuros. Material educativo, no asesoramiento.

Para preparar los datos usar `gs-quant-research-data-risk`; para valoración y la estrategia de empresas de
crecimiento, `gs-quant-research-valuation`.
