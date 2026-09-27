"""
Copyright 2026 Goldman Sachs.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

Empirical asset pricing.

* Time-series factor regressions: CAPM (Sharpe, 1964; Jensen, 1968), Fama & French (1993) three-factor,
  Carhart (1997) four-factor and Fama & French (2015) five-factor models, with Newey & West (1987)
  heteroskedasticity and autocorrelation consistent standard errors.
* Fama & MacBeth (1973) cross-sectional regressions to estimate the risk premia of asset characteristics.

Factor returns are not bundled; the standard academic source is the Kenneth R. French data library.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from gs_quant.errors import MqValueError
from gs_quant.research.performance import infer_periods_per_year


@dataclass
class FactorRegressionResult:
    """
    Output of :func:`factor_regression`. ``alpha`` is per period; t-statistics use HAC standard errors
    """

    alpha: float
    alpha_annualized: float
    betas: pd.Series
    t_stats: pd.Series
    p_values: pd.Series
    r_squared: float
    adj_r_squared: float
    residual_volatility: float
    n_observations: int

    def to_series(self) -> pd.Series:
        """Flatten the result into a single series"""
        data = {'alpha': self.alpha, 'alpha_annualized': self.alpha_annualized}
        data.update({f'beta_{k}': v for k, v in self.betas.items()})
        data.update({f't_{k}': v for k, v in self.t_stats.items()})
        data.update(
            {
                'r_squared': self.r_squared,
                'adj_r_squared': self.adj_r_squared,
                'residual_volatility': self.residual_volatility,
                'n_observations': self.n_observations,
            }
        )
        return pd.Series(data)


@dataclass
class FamaMacBethResult:
    """
    Output of :func:`fama_macbeth`: per-period slopes and their time-series averages
    """

    gammas: pd.DataFrame
    premia: pd.Series
    standard_errors: pd.Series
    t_stats: pd.Series

    def summary(self) -> pd.DataFrame:
        """Table of estimated premia, standard errors, t-statistics and p-values"""
        p_values = 2 * (1 - stats.t.cdf(self.t_stats.abs(), df=max(len(self.gammas) - 1, 1)))
        return pd.DataFrame(
            {'premium': self.premia, 'std_error': self.standard_errors, 't_stat': self.t_stats, 'p_value': p_values}
        )


def newey_west_lags(n_observations: int) -> int:
    """
    Rule-of-thumb number of lags for Newey-West standard errors

    :param n_observations: sample size :math:`T`
    :return: :math:`\\lfloor 4 (T / 100)^{2/9} \\rfloor` (Newey & West, 1994)
    """
    return int(np.floor(4 * (n_observations / 100) ** (2 / 9)))


def factor_regression(
    returns: pd.Series,
    factors: pd.DataFrame,
    risk_free: Optional[pd.Series] = None,
    hac_lags: Optional[int] = None,
    periods_per_year: Optional[float] = None,
) -> FactorRegressionResult:
    """
    Time-series regression of returns on factor returns

    :param returns: periodic returns of the asset or strategy
    :param factors: periodic factor returns, e.g. columns ``Mkt-RF``, ``SMB``, ``HML`` (Fama & French, 1993)
    :param risk_free: periodic risk-free rate subtracted from ``returns`` when provided
    :param hac_lags: lags of the Newey-West covariance; a rule of thumb is used when omitted, 0 gives White (1980)
        heteroskedasticity-robust errors
    :param periods_per_year: observations per year used to annualize alpha; inferred from the index when omitted
    :return: :class:`FactorRegressionResult`

    **Usage**

    Estimates :math:`r_t - r_{f,t} = \\alpha + \\sum_k \\beta_k f_{k,t} + \\varepsilon_t`. A significant positive
    alpha (Jensen, 1968) is return not explained by exposure to the factors.
    """
    y = returns if risk_free is None else returns - risk_free.reindex(returns.index)
    data = pd.concat([y.rename('__y__'), factors], axis=1, join='inner').dropna()
    if len(data) <= factors.shape[1] + 1:
        raise MqValueError('not enough overlapping observations for the regression')
    lags = newey_west_lags(len(data)) if hac_lags is None else hac_lags
    x = sm.add_constant(data[factors.columns], has_constant='add')
    model = sm.OLS(data['__y__'], x).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    ppy = periods_per_year or infer_periods_per_year(data.index)
    alpha = float(model.params['const'])
    rename = {'const': 'alpha'}
    return FactorRegressionResult(
        alpha=alpha,
        alpha_annualized=(1 + alpha) ** ppy - 1,
        betas=model.params.drop('const'),
        t_stats=model.tvalues.rename(rename),
        p_values=model.pvalues.rename(rename),
        r_squared=float(model.rsquared),
        adj_r_squared=float(model.rsquared_adj),
        residual_volatility=float(np.std(model.resid, ddof=x.shape[1]) * np.sqrt(ppy)),
        n_observations=int(model.nobs),
    )


def _newey_west_se(x: np.ndarray, lags: int) -> float:
    n = len(x)
    e = x - x.mean()
    variance = e @ e / n
    for lag in range(1, min(lags, n - 1) + 1):
        variance += 2 * (1 - lag / (lags + 1)) * (e[lag:] @ e[:-lag]) / n
    return float(np.sqrt(max(variance, 0.0) / n))


def fama_macbeth(
    returns: pd.DataFrame,
    characteristics: dict[str, pd.DataFrame],
    lag: int = 1,
    hac_lags: int = 0,
    min_assets: Optional[int] = None,
) -> FamaMacBethResult:
    """
    Fama & MacBeth (1973) two-pass cross-sectional regressions

    :param returns: periodic asset returns (dates x assets)
    :param characteristics: characteristic name to values (dates x assets), e.g. size, book-to-market, momentum
    :param lag: number of periods by which characteristics are lagged so that only known information is used
    :param hac_lags: Newey-West lags for the standard errors of the average slopes; 0 assumes serially
        uncorrelated slopes, as in the original Fama-MacBeth procedure
    :param min_assets: minimum cross-section size for a date to be used; defaults to characteristics + 2
    :return: :class:`FamaMacBethResult`

    **Usage**

    For each date :math:`t`, estimates :math:`r_{i,t} = \\gamma_{0,t} + \\sum_k \\gamma_{k,t} z_{k,i,t-1} +
    \\varepsilon_{i,t}` across assets. The risk premium of each characteristic is the time-series average of
    :math:`\\gamma_{k,t}`, and its standard error comes from the time-series variability of the slopes, which
    is robust to cross-sectional correlation of the residuals.
    """
    names = list(characteristics)
    if not names:
        raise MqValueError('at least one characteristic is required')
    min_assets = min_assets or len(names) + 2
    lagged = {k: v.reindex(index=returns.index, columns=returns.columns).shift(lag) for k, v in characteristics.items()}
    rows = {}
    for date in returns.index:
        cross_section = pd.DataFrame({k: v.loc[date] for k, v in lagged.items()})
        cross_section['__y__'] = returns.loc[date]
        cross_section = cross_section.dropna()
        if len(cross_section) < min_assets:
            continue
        x = np.column_stack([np.ones(len(cross_section)), cross_section[names].to_numpy(dtype=float)])
        coef, *_ = np.linalg.lstsq(x, cross_section['__y__'].to_numpy(dtype=float), rcond=None)
        rows[date] = coef
    if len(rows) < 2:
        raise MqValueError('not enough dates with a valid cross-section')
    gammas = pd.DataFrame.from_dict(rows, orient='index', columns=['intercept'] + names)
    premia = gammas.mean()
    errors = gammas.apply(lambda col: _newey_west_se(col.to_numpy(dtype=float), hac_lags))
    return FamaMacBethResult(gammas=gammas, premia=premia, standard_errors=errors, t_stats=premia / errors)
