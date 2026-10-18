---
name: gs-quant-research-valuation
description: "Valoración de valores con gs_quant.research (sin sesión de Marquee): bonos (precio, TIR, duración, convexidad, DV01), opciones Black-Scholes con griegas y volatilidad implícita, DCF, Gordon, CAPM, DCF de empresas de crecimiento, DCF inverso, ingreso residual, fundamentales de la SEC (ROIC, F-score de Piotroski, G-score de Mohanram) y la estrategia de largo plazo de crecimiento de calidad a precio razonable. Usar para valorar un bono, una opción o una empresa, o para seleccionar acciones de crecimiento."
---

# gs_quant.research — Valoración y empresas de crecimiento

Funciones locales; **no requieren `GsSession`** (solo `to_position_set` sirve para pasar a Marquee).
Cada función cita en su docstring el artículo en el que se basa.

## 0. Instalación

```bash
python -c "import gs_quant.research" 2>/dev/null || \
  pip install "git+https://github.com/mn526zwwfc-sketch/gs-quant@claude/financial-data-management-dqpqs6"
```

## 1. Bonos, opciones y modelos clásicos — `valuation.py`

Tasas y rendimientos siempre en **decimales** (5 % = `0.05`); plazos en **años**.

```python
from gs_quant.research import (bond_analytics, bond_yield, black_scholes, implied_volatility, OptionType,
                               discounted_cash_flow, gordon_growth_value, capm_expected_return)

b = bond_analytics(yield_to_maturity=0.05, coupon_rate=0.04, maturity=10)   # face=100, frequency=2
b.price, b.macaulay_duration, b.modified_duration, b.convexity, b.dv01
bond_yield(price=95, coupon_rate=0.04, maturity=10)                          # TIR

o = black_scholes(spot=100, strike=105, expiry=0.5, rate=0.03, volatility=0.20,
                  dividend_yield=0.0, option_type=OptionType.PUT)
o.price, o.delta, o.gamma, o.vega, o.theta, o.rho
implied_volatility(o.price, 100, 105, 0.5, 0.03, option_type=OptionType.PUT)   # → 0.20

discounted_cash_flow([10, 11, 12], discount_rate=0.09, terminal_growth=0.02)
gordon_growth_value(next_dividend=2, required_return=0.08, growth=0.03)
capm_expected_return(beta=1.2, risk_free=0.04, market_return=0.09)
```

Unidades de las griegas: `vega` y `rho` por **1,00** de cambio (dividir entre 100 para 1 punto porcentual);
`theta` por **año** (dividir entre 365 para un día). `bond_cash_flows(coupon_rate, maturity)` devuelve los
tiempos y flujos.

## 2. DCF de empresas de crecimiento — `growth_valuation.py`

```python
from gs_quant.research import growth_fade_path, growth_dcf, reverse_dcf_growth, residual_income_value

g = growth_fade_path(initial_growth=0.25, terminal_growth=0.025, horizon=10, hold_years=2)
d = growth_dcf(revenue=1000, growth=g, operating_margin=0.20, wacc=0.09, terminal_growth=0.025,
               tax_rate=0.21, sales_to_capital=1.5, net_debt=200, shares=100)
d.value_per_share, d.pv_terminal / d.enterprise_value, d.projections   # proyecciones año a año

# ¿Qué crecimiento descuenta el precio actual?
reverse_dcf_growth(price=52.8, revenue=1000, operating_margin=0.20, wacc=0.09, terminal_growth=0.025,
                   net_debt=200, shares=100)

residual_income_value(book_value_per_share=20, roe=0.18, cost_of_equity=0.09)   # contraste Ohlson (1995)
```

- El crecimiento cuesta: cada unidad de ingreso nuevo exige inversión según `sales_to_capital`.
- Valor terminal con la fórmula *value driver*: con `terminal_roic=None` el ROIC nuevo = WACC, así que el
  crecimiento a perpetuidad **no crea valor** (supuesto conservador de Koller, Goedhart & Wessels).
- Otras piezas: `cost_of_capital(beta, risk_free, equity_risk_premium, market_equity, debt,
  pre_tax_cost_of_debt, tax_rate)` → `(cost_of_equity, wacc)`; `blume_adjusted_beta(beta)` = 0,67·β + 0,33;
  `estimate_initial_growth(revenue_growth, revenue_cagr_3y, terminal_growth, persistence=0.6)`.

## 3. Fundamentales reales — `fundamentals.py`

```python
from gs_quant.research import fetch_sec_fundamentals, prepare_fundamentals, compute_fundamental_metrics

raw = fetch_sec_fundamentals(['AAPL', 'MSFT', 'NVDA'], user_agent='Nombre Apellido correo@ejemplo.com')
metrics = compute_fundamental_metrics(prepare_fundamentals(raw))
# columnas añadidas: roic, operating_margin_3y, gross_profitability, accruals, asset_growth, f_score, ...
```

- La SEC **exige** un `user_agent` con nombre y correo reales; respetar `pause` (0,15 s) entre peticiones.
- Solo cifras US GAAP de informes anuales (10-K, 20-F, 40-F): las empresas que reportan en IFRS se omiten.
- Con datos propios, `prepare_fundamentals` espera una fila por empresa y año fiscal con `ticker`,
  `period_end`, opcional `filed`, y como mínimo `revenue`, `operating_income`, `net_income`, `total_assets`,
  `total_equity`, `shares_outstanding`. Si falta `filed` asume 90 días de retraso de publicación. Ante
  reexpresiones conserva la **primera** cifra publicada (evita sesgo de anticipación).
- `mohanram_g_score(cross_section, groups=None)` (0–8, para acciones de crecimiento) y
  `point_in_time_metrics(metrics, dates, tickers=None, extra_lag_days=0)`.

Valorar una empresa con su última fila de métricas:

```python
from gs_quant.research import value_company, ValuationAssumptions
v = value_company(metrics[metrics.ticker == 'MSFT'].iloc[-1], price=420.0, beta=1.1,
                  assumptions=ValuationAssumptions(risk_free=0.04, equity_risk_premium=0.05))
v.expected_value, v.margin_of_safety, v.implied_growth, v.scenario_values   # bear / base / bull
v.to_series()
```

## 4. Estrategia de largo plazo — `growth_strategy.py`

Crecimiento de calidad a precio razonable, rebalanceo trimestral y solo con datos ya publicados.

```python
import pandas as pd
from gs_quant.research import fetch_sec_fundamentals, GrowthStrategyConfig, LongTermGrowthStrategy

fundamentals = fetch_sec_fundamentals(tickers, user_agent='Nombre Apellido correo@ejemplo.com')
prices = pd.read_csv('precios.csv', index_col=0, parse_dates=True)   # diarios, ajustados, fechas x tickers

strategy = LongTermGrowthStrategy(fundamentals, prices,
                                  GrowthStrategyConfig(n_holdings=5, max_weight=0.25),
                                  sectors=None, market_prices=None)
result = strategy.run()                         # opcional: start='2015-01-01', end=...
result.holdings()                               # cartera actual: weight, composite_score, margin_of_safety, ...
strategy.backtest(result, transaction_cost_bps=20).summary()
position_set = strategy.to_position_set(result) # PositionSet de gs_quant.markets (resolve/price con Marquee)
```

Demo sin datos externos: `python -m gs_quant.research.growth_demo` (120 empresas sintéticas), o con datos
propios `--fundamentals f.csv --prices p.csv --sectors s.csv`.

`GrowthStrategyConfig` (valores por defecto): `n_holdings=25`, `buffer=1.5` (histéresis), `rebalance='Q'`,
`min_history_years=3`, `growth_quantile=0.5`, `min_revenue_growth=0.05`, `min_f_score=5`,
`require_value_creation=True` (ROIC > WACC), `min_margin_of_safety=0.0`, `hold_margin_of_safety=-0.1`,
`max_per_sector=None`, `score_weights={'growth': 0.25, 'quality': 0.30, 'valuation': 0.30, 'momentum': 0.15}`,
`weighting='risk_parity'` (`'equal'`, `'inverse_volatility'`, `'hrp'`), `max_weight=0.10`,
`valuation=ValuationAssumptions()`. Si hay pocas oportunidades baratas, el resto queda en **efectivo**.

## 5. Reglas que hay que respetar

1. Explicar los supuestos (WACC, crecimiento inicial, margen, valor terminal) y su peso: `terminal_value_share`.
2. El modelo supone beneficio operativo positivo; para empresas con pérdidas pedir un margen objetivo explícito.
3. Usar solo empresas que cotizan hoy introduce **sesgo de supervivencia** en el backtest: advertirlo.
4. Los resultados del demo usan datos sintéticos: validan la mecánica, no la rentabilidad real.
5. Material educativo; no constituye asesoramiento de inversión.

Para preparar precios usar `gs-quant-research-data-risk`; para señales, carteras y evaluación de backtests,
`gs-quant-research-strategies`.
