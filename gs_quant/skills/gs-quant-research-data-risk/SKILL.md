---
name: gs-quant-research-data-risk
description: "Limpieza y validación de series de precios y medición de riesgo con gs_quant.research (sin sesión de Marquee): auditoría de datos, outliers con filtro de Hampel, datos point-in-time, covarianza Ledoit-Wolf/EWMA/Marchenko-Pastur, VaR, Expected Shortfall y contribuciones al riesgo. Usar cuando haya que preparar precios de CSV/Yahoo/Bloomberg para un análisis o medir el riesgo de una cartera."
---

# gs_quant.research — Gestión de datos y riesgo

Funciones locales (solo `numpy`, `pandas`, `scipy`, `statsmodels`); **no requieren `GsSession`**.
Cada función cita en su docstring el artículo en el que se basa.

## 0. Instalación

`gs_quant.research` está en el fork `mn526zwwfc-sketch/gs-quant`, rama `claude/financial-data-management-dqpqs6`
(no está en la versión de PyPI). Comprobar primero y, si falta, instalar:

```bash
python -c "import gs_quant.research" 2>/dev/null || \
  pip install "git+https://github.com/mn526zwwfc-sketch/gs-quant@claude/financial-data-management-dqpqs6"
```

## 1. Flujo recomendado

```python
import numpy as np
import pandas as pd
from gs_quant.research import (
    validate_prices, clean_prices, compute_returns, hampel_filter,
    ledoit_wolf_covariance, portfolio_volatility, risk_contributions,
    value_at_risk, expected_shortfall, VaRMethod,
)

prices = pd.read_csv('precios.csv', index_col=0, parse_dates=True)   # fechas x activos

# 1) Auditar ANTES de tocar nada
report = validate_prices(prices)          # DataQualityReport
print(report.to_frame())                  # missing, non_positive, longest_stale_run, outliers por activo
print(report.duplicated_dates, report.is_sorted)

# 2) Limpiar sin sesgo de anticipación (ordena, deduplica, quita precios <= 0, ffill limitado, NUNCA bfill)
prices = clean_prices(prices, max_fill=5)

# 3) Rendimientos y outliers robustos (mediana + MAD, ventana causal por defecto)
returns = hampel_filter(compute_returns(prices), window=63, n_sigmas=8).dropna(how='all')

# 4) Riesgo de una cartera
cov = ledoit_wolf_covariance(returns.tail(252))
w = pd.Series(1 / returns.shape[1], index=returns.columns)
vol_anual = portfolio_volatility(w, cov) * np.sqrt(252)
rc = risk_contributions(w, cov)           # fracciones que suman 1 (asignación de Euler)

port = returns @ w
var99 = value_at_risk(port, confidence=0.99, method=VaRMethod.CORNISH_FISHER)
es975 = expected_shortfall(port, confidence=0.975)   # pérdida como número positivo
```

## 2. Referencia — `data_management.py`

| Función | Firma (parámetros clave) | Notas |
|---|---|---|
| `validate_prices` | `(prices, stale_threshold=5, outlier_window=21, n_sigmas=5.0)` → `DataQualityReport` | `.to_frame()` para tabla por activo |
| `clean_prices` | `(prices, max_fill=5, drop_non_positive=True)` | Solo *forward fill*, como máximo `max_fill` periodos |
| `compute_returns` | `(prices, return_type=ReturnType.SIMPLE, periods=1)` | `ReturnType.LOG` para log-rendimientos |
| `aggregate_returns` | `(returns, frequency='M', return_type=...)` | Compone correctamente (no suma simples) |
| `detect_outliers` / `hampel_filter` | `(x, window=21, n_sigmas=3.0, center=False)` | `detect_outliers` devuelve máscara booleana; `hampel_filter` reemplaza por la mediana. `center=True` mira al futuro: solo para análisis ex post |
| `winsorize` | `(x, lower=0.01, upper=0.99, axis=0)` | `axis=1` = transversal (por fecha) |
| `cross_sectional_zscore` / `cross_sectional_rank` | `(x)` | Normaliza señales por fecha |
| `point_in_time` | `(data, target_index, publication_lag='0D')` | Alinea cada dato a cuando **se publicó** |
| `lag_signal` | `(signal, periods=1)` | Retrasa una señal para evitar anticipación |
| `align` | `(*data)` | Intersección de índices y columnas |

Ejemplo point-in-time (fundamentales trimestrales publicados ~45 días tras el cierre):

```python
from gs_quant.research import point_in_time
eps_diario = point_in_time(eps_trimestral, prices.index, publication_lag='45D')
```

## 3. Referencia — `risk.py`

| Función | Firma | Fundamento |
|---|---|---|
| `sample_covariance` | `(returns)` | Solo como referencia; mal condicionada con muchos activos |
| `ledoit_wolf_covariance` / `ledoit_wolf_shrinkage` | `(returns)` | Ledoit & Wolf (2004). **Opción por defecto** |
| `ewma_covariance` | `(returns, decay=0.94)` | RiskMetrics (1996); reacciona rápido a cambios de régimen |
| `denoised_covariance` | `(returns)` | Marchenko-Pastur (Laloux et al., 1999) |
| `denoise_correlation` | `(corr, n_observations)` | Igual, sobre una matriz de correlación |
| `covariance_to_correlation` | `(cov)` | |
| `value_at_risk` | `(returns, confidence=0.95, method=VaRMethod.HISTORICAL)` | Métodos: `HISTORICAL`, `GAUSSIAN`, `CORNISH_FISHER` |
| `expected_shortfall` | `(returns, confidence=0.95, method=...)` | Medida coherente (Artzner et al., 1999) |
| `portfolio_volatility` | `(weights, cov)` | En la frecuencia de `cov` (diaria → ×√252) |
| `risk_contributions` | `(weights, cov)` | Suman 1 |

VaR y ES se devuelven como **pérdidas positivas** en la frecuencia de los rendimientos (diaria si los datos son
diarios). Con un `DataFrame` devuelven una `Series` por columna.

## 4. Reglas que hay que respetar

1. Validar → limpiar → calcular rendimientos, en ese orden. Nunca `bfill`: introduce información futura.
2. Detectar outliers con mediana/MAD (Hampel), no con media/desviación estándar.
3. No usar la covarianza muestral en bruto para optimizar: usar Ledoit-Wolf o la versión sin ruido.
4. Fundamentales y datos macro: siempre con `point_in_time` y su retraso de publicación.
5. El sesgo de supervivencia no se puede corregir si los datos no incluyen activos deslistados: avisarlo.
6. Es material educativo; no constituye asesoramiento de inversión.

Para estrategias, carteras y backtests usar la skill `gs-quant-research-strategies`; para bonos, opciones, DCF y
la estrategia de empresas de crecimiento, `gs-quant-research-valuation`.
