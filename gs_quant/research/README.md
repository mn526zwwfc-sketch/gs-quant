# `gs_quant.research` — Finanzas cuantitativas basadas en investigación académica

Kit de herramientas para **gestión de datos**, **valoración de valores (securities)** y **estrategias
financieras**, implementado a partir de artículos revisados por pares y libros de texto universitarios.
Cada función documenta en su *docstring* el estudio en el que se basa.

- Solo depende de `numpy`, `pandas`, `scipy` y `statsmodels` (ya incluidos en `gs_quant`).
- **No requiere sesión de Marquee**: funciona con cualquier serie de precios (CSV, Bloomberg, Yahoo, etc.).
- 97 tests unitarios verifican las fórmulas contra valores de referencia, implementaciones independientes
  y diferencias finitas.

> ⚠️ Material educativo y de investigación. No constituye asesoramiento de inversión. Un backtest es una
> simulación histórica: los resultados pasados no garantizan resultados futuros.

## Inicio rápido

```bash
pip install -e .
python -m gs_quant.research.demo                 # datos sintéticos
python -m gs_quant.research.demo --csv precios.csv --cost-bps 10
```

El CSV debe tener las fechas en la primera columna y una columna de precios por activo.

```python
import pandas as pd
from gs_quant.research import (
    clean_prices, validate_prices, compute_returns, hampel_filter,
    ledoit_wolf_covariance, risk_parity, hierarchical_risk_parity,
    rolling_allocation, backtest, time_series_momentum, volatility_scaling,
    performance_summary, deflated_sharpe_ratio, sharpe_ratio,
)

prices = pd.read_csv('precios.csv', index_col=0, parse_dates=True)

# 1. Gestión de datos: auditar, limpiar sin sesgo de anticipación y filtrar outliers de forma robusta
print(validate_prices(prices).to_frame())
prices = clean_prices(prices, max_fill=5)
returns = hampel_filter(compute_returns(prices), window=63, n_sigmas=8)

# 2. Asignación walk-forward (fuera de muestra) con covarianza Ledoit-Wolf
weights = rolling_allocation(returns, lambda r: risk_parity(ledoit_wolf_covariance(r)),
                             lookback=252, rebalance='M')
rp = backtest(returns, weights, transaction_cost_bps=10)

# 3. Estrategia: momentum de series temporales con volatilidad objetivo del 10 %
tsmom = volatility_scaling(time_series_momentum(prices), prices, target_volatility=0.10)
trend = backtest(returns, tsmom, transaction_cost_bps=10, rebalance='M')

# 4. Evaluación neta de costos y corrección por pruebas múltiples
net = pd.DataFrame({'risk_parity': rp.returns, 'tsmom': trend.returns})
print(performance_summary(net))
trials = [sharpe_ratio(net[c]) for c in net]
print('Deflated Sharpe:', deflated_sharpe_ratio(net['tsmom'], trials))
```

## Módulos

### 1. Gestión de datos — `data_management.py`

| Función | Qué hace | Fundamento académico |
|---|---|---|
| `validate_prices` | Detecta faltantes, precios ≤ 0, precios estancados, fechas duplicadas/desordenadas y saltos anómalos | Lo & MacKinlay (1990); Getmansky, Lo & Makarov (2004) |
| `clean_prices` | Ordena, deduplica, elimina precios no positivos y rellena **solo hacia adelante** (nunca *backfill*) | López de Prado (2018) |
| `detect_outliers`, `hampel_filter` | Filtro de Hampel con MAD (robusto), ventana causal por defecto | Hampel (1974); Rousseeuw & Croux (1993) |
| `winsorize` | Recorte por cuantiles, temporal o transversal | Tukey (1962); Fama & French (1992) |
| `cross_sectional_zscore`, `cross_sectional_rank` | Normalización transversal de señales | Grinold & Kahn (2000); Asness, Moskowitz & Pedersen (2013) |
| `point_in_time` | Alinea fundamentales en la fecha en que **se publicaron** | Fama & French (1993) |
| `compute_returns`, `aggregate_returns`, `align`, `lag_signal` | Rendimientos simples/log, cambio de frecuencia, alineación, rezago de señales | — |

### 2. Riesgo — `risk.py`

| Función | Qué hace | Fundamento académico |
|---|---|---|
| `ledoit_wolf_covariance` | Covarianza con *shrinkage* óptimo; siempre bien condicionada | Ledoit & Wolf (2004) |
| `ewma_covariance` | Covarianza exponencial (λ = 0,94) | J.P. Morgan RiskMetrics (1996) |
| `denoise_correlation`, `denoised_covariance` | Elimina ruido con la ley de Marchenko-Pastur | Laloux et al. (1999); López de Prado (2020) |
| `value_at_risk` | VaR histórico, gaussiano y Cornish-Fisher (asimetría y curtosis) | Jorion (2006); Zangari (1996); Favre & Galeano (2002) |
| `expected_shortfall` | CVaR / ES, medida de riesgo coherente | Artzner et al. (1999); Acerbi & Tasche (2002) |
| `risk_contributions` | Contribución de cada activo al riesgo (asignación de Euler) | Maillard, Roncalli & Teiletche (2010) |

### 3. Evaluación del desempeño — `performance.py`

| Función | Fundamento académico |
|---|---|
| `sharpe_ratio`, `sortino_ratio`, `calmar_ratio`, `omega_ratio`, `information_ratio`, `max_drawdown` | Sharpe (1966, 1994); Lo (2002); Sortino & Price (1994); Young (1991); Keating & Shadwick (2002); Grinold & Kahn (2000) |
| `probabilistic_sharpe_ratio`, `minimum_track_record_length` | Bailey & López de Prado (2012) |
| `deflated_sharpe_ratio`, `expected_maximum_sharpe` | Bailey & López de Prado (2014) |
| `performance_summary` | Tabla con todas las métricas anteriores + VaR/ES |

### 4. Valoración de valores — `valuation.py`

| Función | Fundamento académico |
|---|---|
| `bond_price`, `bond_yield`, `bond_analytics` (duración Macaulay/modificada, convexidad, DV01) | Macaulay (1938); Fabozzi (2012) |
| `black_scholes` (precio y griegas con dividendo continuo), `implied_volatility` | Black & Scholes (1973); Merton (1973); Hull (2018) |
| `gordon_growth_value`, `discounted_cash_flow` | Gordon (1962) |
| `capm_expected_return` | Sharpe (1964); Lintner (1965) |

### 5. Modelos de factores — `factors.py`

| Función | Fundamento académico |
|---|---|
| `factor_regression` (CAPM, FF3, Carhart, FF5) con errores estándar Newey-West | Jensen (1968); Fama & French (1993, 2015); Carhart (1997); Newey & West (1987, 1994) |
| `fama_macbeth` (primas de riesgo de características) | Fama & MacBeth (1973) |

Los rendimientos de los factores se obtienen de la *Kenneth R. French Data Library* (Dartmouth).

### 6. Construcción de carteras — `portfolio.py`

| Función | Fundamento académico |
|---|---|
| `equal_weight` (1/N) | DeMiguel, Garlappi & Uppal (2009) |
| `inverse_volatility`, `risk_parity` (presupuestos de riesgo) | Qian (2005); Maillard, Roncalli & Teiletche (2010); Spinu (2013); Griveau-Billion, Richard & Roncalli (2013) |
| `hierarchical_risk_parity` | López de Prado (2016) |
| `minimum_variance` | Clarke, de Silva & Thorley (2006); Jagannathan & Ma (2003) |
| `mean_variance`, `efficient_frontier` | Markowitz (1952) |
| `maximum_sharpe` (cartera tangente) | Tobin (1958); Sharpe (1964) |
| `black_litterman` | Black & Litterman (1992); He & Litterman (1999) |
| `kelly_weights`, `kelly_fraction` | Kelly (1956); Thorp (2006); MacLean, Thorp & Ziemba (2011) |

### 7. Estrategias sistemáticas — `signals.py`

| Función | Fundamento académico |
|---|---|
| `time_series_momentum` | Moskowitz, Ooi & Pedersen (2012); Hurst, Ooi & Pedersen (2017) |
| `cross_sectional_momentum` (12-1 meses) | Jegadeesh & Titman (1993); Carhart (1997); Daniel & Moskowitz (2016) |
| `short_term_reversal` | Jegadeesh (1990); Lehmann (1990); Lo & MacKinlay (1990) |
| `moving_average_crossover` | Brock, Lakonishok & LeBaron (1992); Faber (2007) |
| `low_volatility` | Ang, Hodrick, Xing & Zhang (2006); Blitz & van Vliet (2007); Baker, Bradley & Wurgler (2011); Frazzini & Pedersen (2014) |
| `volatility_scaling` | Moreira & Muir (2017); Barroso & Santa-Clara (2015); Harvey et al. (2018) |
| `combine_signals` | Asness, Moskowitz & Pedersen (2013) |

### 8. Backtesting y validación — `backtest.py`

| Función | Fundamento académico |
|---|---|
| `backtest` (ejecución retrasada, deriva de pesos, costos sobre la rotación real) | Novy-Marx & Velikov (2016) |
| `rolling_allocation` (walk-forward, estrictamente fuera de muestra) | Bailey, Borwein, López de Prado & Zhu (2014) |
| `walk_forward_splits`, `purged_kfold_splits` (purga y embargo) | López de Prado (2018, cap. 7) |

## Mejores prácticas respaldadas por la literatura

1. **Datos *point-in-time*.** Usar cada dato solo desde la fecha en que se publicó (`point_in_time`), rellenar
   solo hacia adelante y retrasar la ejecución al menos un periodo (`backtest(execution_lag=1)`). Incluir
   activos deslistados en el universo para evitar el sesgo de supervivencia: ninguna función puede corregirlo
   si los datos de origen no los contienen.
2. **Estadística robusta.** Detectar errores con la mediana y la MAD, no con la media y la desviación estándar
   (Hampel, 1974); winsorizar de forma transversal.
3. **No usar la covarianza muestral en bruto.** Con muchos activos, el optimizador maximiza el error de
   estimación (Michaud, 1989). Usar *shrinkage* (Ledoit & Wolf, 2004) o limpieza espectral (Laloux et al., 1999).
4. **Desconfiar de los rendimientos esperados.** Los errores en las medias pesan ~10 veces más que los de las
   varianzas (Chopra & Ziemba, 1993). Sin pronósticos fiables, preferir 1/N, paridad de riesgo o HRP; con
   opiniones, usar Black-Litterman y restricciones de peso.
5. **Descontar costos de transacción.** Muchas anomalías publicadas desaparecen tras costos
   (Novy-Marx & Velikov, 2016). Revisar `annual_turnover` y `annual_cost_drag` en `BacktestResult.summary()`.
6. **Validar fuera de muestra.** Reestimar en ventanas móviles (`rolling_allocation`) y usar validación cruzada
   purgada cuando las etiquetas se solapan (López de Prado, 2018).
7. **Corregir por pruebas múltiples.** Tras probar muchas configuraciones, exigir un *Deflated Sharpe Ratio*
   > 0,95 (Bailey & López de Prado, 2014) o un estadístico t > 3 (Harvey, Liu & Zhu, 2016).
8. **Gestionar la volatilidad.** Escalar la exposición a una volatilidad objetivo mejora el Sharpe y reduce
   las colas (Moreira & Muir, 2017), sobre todo en momentum (Barroso & Santa-Clara, 2015).
9. **Kelly fraccional.** El Kelly completo es óptimo solo con parámetros conocidos; en la práctica se usa
   medio Kelly o menos (MacLean, Thorp & Ziemba, 2011).

## Bibliografía

- Acerbi, C. & Tasche, D. (2002). On the coherence of expected shortfall. *Journal of Banking & Finance*, 26(7), 1487–1503.
- Ang, A., Hodrick, R., Xing, Y. & Zhang, X. (2006). The cross-section of volatility and expected returns. *Journal of Finance*, 61(1), 259–299.
- Artzner, P., Delbaen, F., Eber, J.-M. & Heath, D. (1999). Coherent measures of risk. *Mathematical Finance*, 9(3), 203–228.
- Asness, C., Moskowitz, T. & Pedersen, L. (2013). Value and momentum everywhere. *Journal of Finance*, 68(3), 929–985.
- Bailey, D., Borwein, J., López de Prado, M. & Zhu, Q. (2014). Pseudo-mathematics and financial charlatanism. *Notices of the AMS*, 61(5), 458–471.
- Bailey, D. & López de Prado, M. (2012). The Sharpe ratio efficient frontier. *Journal of Risk*, 15(2), 3–44.
- Bailey, D. & López de Prado, M. (2014). The deflated Sharpe ratio. *Journal of Portfolio Management*, 40(5), 94–107.
- Baker, M., Bradley, B. & Wurgler, J. (2011). Benchmarks as limits to arbitrage: understanding the low-volatility anomaly. *Financial Analysts Journal*, 67(1), 40–54.
- Barroso, P. & Santa-Clara, P. (2015). Momentum has its moments. *Journal of Financial Economics*, 116(1), 111–120.
- Black, F. & Litterman, R. (1992). Global portfolio optimization. *Financial Analysts Journal*, 48(5), 28–43.
- Black, F. & Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of Political Economy*, 81(3), 637–654.
- Blitz, D. & van Vliet, P. (2007). The volatility effect. *Journal of Portfolio Management*, 34(1), 102–113.
- Brock, W., Lakonishok, J. & LeBaron, B. (1992). Simple technical trading rules and the stochastic properties of stock returns. *Journal of Finance*, 47(5), 1731–1764.
- Carhart, M. (1997). On persistence in mutual fund performance. *Journal of Finance*, 52(1), 57–82.
- Chopra, V. & Ziemba, W. (1993). The effect of errors in means, variances, and covariances on optimal portfolio choice. *Journal of Portfolio Management*, 19(2), 6–11.
- Clarke, R., de Silva, H. & Thorley, S. (2006). Minimum-variance portfolios in the U.S. equity market. *Journal of Portfolio Management*, 33(1), 10–24.
- Cornuejols, G. & Tütüncü, R. (2007). *Optimization Methods in Finance*. Cambridge University Press.
- Daniel, K. & Moskowitz, T. (2016). Momentum crashes. *Journal of Financial Economics*, 122(2), 221–247.
- DeMiguel, V., Garlappi, L. & Uppal, R. (2009). Optimal versus naive diversification: how inefficient is the 1/N portfolio strategy? *Review of Financial Studies*, 22(5), 1915–1953.
- Faber, M. (2007). A quantitative approach to tactical asset allocation. *Journal of Wealth Management*, 9(4), 69–79.
- Fabozzi, F. (2012). *Bond Markets, Analysis, and Strategies* (8th ed.). Pearson.
- Fama, E. & French, K. (1992). The cross-section of expected stock returns. *Journal of Finance*, 47(2), 427–465.
- Fama, E. & French, K. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3–56.
- Fama, E. & French, K. (2015). A five-factor asset pricing model. *Journal of Financial Economics*, 116(1), 1–22.
- Fama, E. & MacBeth, J. (1973). Risk, return, and equilibrium: empirical tests. *Journal of Political Economy*, 81(3), 607–636.
- Favre, L. & Galeano, J.-A. (2002). Mean-modified value-at-risk optimization with hedge funds. *Journal of Alternative Investments*, 5(2), 21–25.
- Frazzini, A. & Pedersen, L. (2014). Betting against beta. *Journal of Financial Economics*, 111(1), 1–25.
- Getmansky, M., Lo, A. & Makarov, I. (2004). An econometric model of serial correlation and illiquidity in hedge fund returns. *Journal of Financial Economics*, 74(3), 529–609.
- Gordon, M. (1962). *The Investment, Financing, and Valuation of the Corporation*. Irwin.
- Grinold, R. & Kahn, R. (2000). *Active Portfolio Management* (2nd ed.). McGraw-Hill.
- Griveau-Billion, T., Richard, J.-C. & Roncalli, T. (2013). A fast algorithm for computing high-dimensional risk parity portfolios. SSRN 2325255.
- Hampel, F. (1974). The influence curve and its role in robust estimation. *Journal of the American Statistical Association*, 69(346), 383–393.
- Harvey, C., Hoyle, E., Korgaonkar, R., Rattray, S., Sargaison, M. & Van Hemert, O. (2018). The impact of volatility targeting. *Journal of Portfolio Management*, 45(1), 14–33.
- Harvey, C., Liu, Y. & Zhu, H. (2016). … and the cross-section of expected returns. *Review of Financial Studies*, 29(1), 5–68.
- He, G. & Litterman, R. (1999). The intuition behind Black-Litterman model portfolios. Goldman Sachs Investment Management Research.
- Hull, J. (2018). *Options, Futures, and Other Derivatives* (10th ed.). Pearson.
- Hurst, B., Ooi, Y. H. & Pedersen, L. (2017). A century of evidence on trend-following investing. *Journal of Portfolio Management*, 44(1), 15–29.
- J.P. Morgan / Reuters (1996). *RiskMetrics — Technical Document* (4th ed.).
- Jagannathan, R. & Ma, T. (2003). Risk reduction in large portfolios: why imposing the wrong constraints helps. *Journal of Finance*, 58(4), 1651–1683.
- Jegadeesh, N. (1990). Evidence of predictable behavior of security returns. *Journal of Finance*, 45(3), 881–898.
- Jegadeesh, N. & Titman, S. (1993). Returns to buying winners and selling losers. *Journal of Finance*, 48(1), 65–91.
- Jensen, M. (1968). The performance of mutual funds in the period 1945–1964. *Journal of Finance*, 23(2), 389–416.
- Jorion, P. (2006). *Value at Risk* (3rd ed.). McGraw-Hill.
- Keating, C. & Shadwick, W. (2002). A universal performance measure. *Journal of Performance Measurement*, 6(3), 59–84.
- Kelly, J. (1956). A new interpretation of information rate. *Bell System Technical Journal*, 35(4), 917–926.
- Laloux, L., Cizeau, P., Bouchaud, J.-P. & Potters, M. (1999). Noise dressing of financial correlation matrices. *Physical Review Letters*, 83(7), 1467–1470.
- Ledoit, O. & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365–411.
- Lehmann, B. (1990). Fads, martingales, and market efficiency. *Quarterly Journal of Economics*, 105(1), 1–28.
- Lintner, J. (1965). The valuation of risk assets and the selection of risky investments in stock portfolios and capital budgets. *Review of Economics and Statistics*, 47(1), 13–37.
- Lo, A. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal*, 58(4), 36–52.
- Lo, A. & MacKinlay, A. C. (1990). When are contrarian profits due to stock market overreaction? *Review of Financial Studies*, 3(2), 175–205.
- López de Prado, M. (2016). Building diversified portfolios that outperform out of sample. *Journal of Portfolio Management*, 42(4), 59–69.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- López de Prado, M. (2020). *Machine Learning for Asset Managers*. Cambridge University Press.
- Macaulay, F. (1938). *Some Theoretical Problems Suggested by the Movements of Interest Rates, Bond Yields and Stock Prices in the United States since 1856*. NBER.
- MacLean, L., Thorp, E. & Ziemba, W. (2011). *The Kelly Capital Growth Investment Criterion*. World Scientific.
- Maillard, D. (2012). A user's guide to the Cornish Fisher expansion. SSRN 1997178.
- Maillard, S., Roncalli, T. & Teiletche, J. (2010). The properties of equally weighted risk contribution portfolios. *Journal of Portfolio Management*, 36(4), 60–70.
- Marchenko, V. & Pastur, L. (1967). Distribution of eigenvalues for some sets of random matrices. *Matematicheskii Sbornik*, 72(4), 507–536.
- Markowitz, H. (1952). Portfolio selection. *Journal of Finance*, 7(1), 77–91.
- Merton, R. (1973). Theory of rational option pricing. *Bell Journal of Economics and Management Science*, 4(1), 141–183.
- Michaud, R. (1989). The Markowitz optimization enigma: is 'optimized' optimal? *Financial Analysts Journal*, 45(1), 31–42.
- Moreira, A. & Muir, T. (2017). Volatility-managed portfolios. *Journal of Finance*, 72(4), 1611–1644.
- Moskowitz, T., Ooi, Y. H. & Pedersen, L. (2012). Time series momentum. *Journal of Financial Economics*, 104(2), 228–250.
- Newey, W. & West, K. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.
- Newey, W. & West, K. (1994). Automatic lag selection in covariance matrix estimation. *Review of Economic Studies*, 61(4), 631–653.
- Novy-Marx, R. & Velikov, M. (2016). A taxonomy of anomalies and their trading costs. *Review of Financial Studies*, 29(1), 104–147.
- Qian, E. (2005). Risk parity portfolios: efficient portfolios through true diversification. PanAgora Asset Management.
- Rousseeuw, P. & Croux, C. (1993). Alternatives to the median absolute deviation. *Journal of the American Statistical Association*, 88(424), 1273–1283.
- Sharpe, W. (1964). Capital asset prices: a theory of market equilibrium under conditions of risk. *Journal of Finance*, 19(3), 425–442.
- Sharpe, W. (1966). Mutual fund performance. *Journal of Business*, 39(1), 119–138.
- Sharpe, W. (1994). The Sharpe ratio. *Journal of Portfolio Management*, 21(1), 49–58.
- Sortino, F. & Price, L. (1994). Performance measurement in a downside risk framework. *Journal of Investing*, 3(3), 59–64.
- Spinu, F. (2013). An algorithm for computing risk parity weights. SSRN 2297383.
- Thorp, E. (2006). The Kelly criterion in blackjack, sports betting, and the stock market. *Handbook of Asset and Liability Management*, vol. 1, 385–428.
- Tobin, J. (1958). Liquidity preference as behavior towards risk. *Review of Economic Studies*, 25(2), 65–86.
- Tukey, J. (1962). The future of data analysis. *Annals of Mathematical Statistics*, 33(1), 1–67.
- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817–838.
- Young, T. (1991). Calmar ratio: a smoother tool. *Futures*, 20(1), 40.
- Zangari, P. (1996). A VaR methodology for portfolios that include options. *RiskMetrics Monitor*, Q1, 4–12.
