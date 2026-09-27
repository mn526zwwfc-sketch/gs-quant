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

Performance evaluation.

Risk-adjusted performance measures from the academic literature, including the tools of Bailey & López de Prado
to assess whether a Sharpe ratio is statistically significant once non-normal returns and multiple testing
(backtest overfitting) are taken into account:

* Sharpe ratio (Sharpe, 1966, 1994) and its annualization caveats (Lo, 2002).
* Sortino ratio (Sortino & Price, 1994), Calmar ratio (Young, 1991), Omega ratio (Keating & Shadwick, 2002)
  and Information ratio (Grinold & Kahn, 2000).
* Probabilistic Sharpe ratio and minimum track record length (Bailey & López de Prado, 2012).
* Deflated Sharpe ratio (Bailey & López de Prado, 2014).

All functions take periodic simple returns. A DataFrame input is evaluated column by column.
"""

import functools
from typing import Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

from gs_quant.errors import MqValueError
from gs_quant.research.risk import expected_shortfall, value_at_risk

EULER_MASCHERONI = 0.5772156649015329

ReturnsLike = Union[pd.Series, pd.DataFrame]


def infer_periods_per_year(index: pd.Index) -> float:
    """
    Infer the number of observations per year from the spacing of a DatetimeIndex

    :param index: dates of the observations
    :return: 252 for daily, 52 for weekly, 12 for monthly, 4 for quarterly and 1 for annual data
    """
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return 252.0
    days = np.median(np.diff(index.values).astype('timedelta64[D]').astype(float))
    for max_days, periods in ((1.5, 252.0), (8, 52.0), (35, 12.0), (100, 4.0)):
        if days <= max_days:
            return periods
    return 1.0


def _columnwise(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(returns, *args, **kwargs):
        if isinstance(returns, pd.DataFrame):
            return returns.apply(lambda col: func(col.dropna(), *args, **kwargs))
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns, dtype=float)
        return func(returns.dropna(), *args, **kwargs)

    return wrapper


def _ppy(returns: pd.Series, periods_per_year: Optional[float]) -> float:
    return float(periods_per_year) if periods_per_year else infer_periods_per_year(returns.index)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den and np.isfinite(den) else np.nan


@_columnwise
def annualized_return(returns: ReturnsLike, periods_per_year: Optional[float] = None) -> float:
    """
    Compound annual growth rate

    :param returns: periodic simple returns
    :param periods_per_year: observations per year; inferred from the index when omitted
    :return: :math:`(\\prod (1 + r_t))^{P / T} - 1`
    """
    if returns.empty:
        return np.nan
    growth = float(np.prod(1 + returns.to_numpy()))
    if growth <= 0:
        return -1.0
    return growth ** (_ppy(returns, periods_per_year) / len(returns)) - 1


@_columnwise
def annualized_volatility(returns: ReturnsLike, periods_per_year: Optional[float] = None) -> float:
    """
    Annualized standard deviation of returns

    :param returns: periodic simple returns
    :param periods_per_year: observations per year; inferred from the index when omitted
    :return: :math:`\\sigma \\sqrt{P}`
    """
    return float(returns.std(ddof=1) * np.sqrt(_ppy(returns, periods_per_year)))


@_columnwise
def sharpe_ratio(returns: ReturnsLike, risk_free: float = 0.0, periods_per_year: Optional[float] = None) -> float:
    """
    Annualized Sharpe ratio

    :param returns: periodic simple returns
    :param risk_free: annual risk-free rate, de-annualized linearly
    :param periods_per_year: observations per year; inferred from the index when omitted
    :return: :math:`\\sqrt{P}\\, \\overline{(r - r_f)} / \\sigma(r - r_f)`

    **Usage**

    The square-root-of-time annualization assumes serially uncorrelated returns; autocorrelated strategies
    (e.g. illiquid assets) can have their Sharpe ratio overstated by 65% or more (Lo, 2002).
    """
    ppy = _ppy(returns, periods_per_year)
    excess = returns - risk_free / ppy
    return _safe_div(excess.mean(), excess.std(ddof=1)) * np.sqrt(ppy)


@_columnwise
def sortino_ratio(returns: ReturnsLike, target: float = 0.0, periods_per_year: Optional[float] = None) -> float:
    """
    Annualized Sortino ratio

    :param returns: periodic simple returns
    :param target: annual minimum acceptable return
    :param periods_per_year: observations per year; inferred from the index when omitted
    :return: mean excess return over the target divided by the downside deviation

    **Usage**

    Only returns below the target are penalized, since investors do not dislike upside volatility
    (Sortino & Price, 1994). The downside deviation is :math:`\\sqrt{E[\\min(r - \\tau, 0)^2]}`.
    """
    ppy = _ppy(returns, periods_per_year)
    excess = returns - target / ppy
    downside = np.sqrt(np.mean(np.minimum(excess.to_numpy(), 0) ** 2))
    return _safe_div(excess.mean(), downside) * np.sqrt(ppy)


def drawdown(returns: ReturnsLike) -> ReturnsLike:
    """
    Drawdown from the running peak of the wealth curve

    :param returns: periodic simple returns
    :return: drawdown series, zero at new highs and negative otherwise
    """
    wealth = (1 + returns.fillna(0)).cumprod()
    peak = np.maximum(wealth.cummax(), 1.0)
    return wealth / peak - 1


@_columnwise
def max_drawdown(returns: ReturnsLike) -> float:
    """
    Maximum peak-to-trough decline of the wealth curve

    :param returns: periodic simple returns
    :return: maximum drawdown as a negative fraction, e.g. -0.35
    """
    return float(drawdown(returns).min()) if not returns.empty else np.nan


@_columnwise
def calmar_ratio(returns: ReturnsLike, periods_per_year: Optional[float] = None) -> float:
    """
    Calmar ratio: annualized return over maximum drawdown (Young, 1991)

    :param returns: periodic simple returns
    :param periods_per_year: observations per year; inferred from the index when omitted
    :return: :math:`CAGR / |MDD|`
    """
    return _safe_div(annualized_return(returns, periods_per_year), abs(max_drawdown(returns)))


@_columnwise
def omega_ratio(returns: ReturnsLike, threshold: float = 0.0) -> float:
    """
    Omega ratio (Keating & Shadwick, 2002)

    :param returns: periodic simple returns
    :param threshold: per-period return threshold
    :return: probability-weighted gains over probability-weighted losses relative to the threshold

    **Usage**

    :math:`\\Omega(\\tau) = E[(r - \\tau)^+] / E[(\\tau - r)^+]`. It uses the whole return distribution, so it
    captures skewness and fat tails that mean-variance measures ignore.
    """
    r = returns.to_numpy()
    return _safe_div(np.sum(np.maximum(r - threshold, 0)), np.sum(np.maximum(threshold - r, 0)))


@_columnwise
def information_ratio(returns: ReturnsLike, benchmark: pd.Series, periods_per_year: Optional[float] = None) -> float:
    """
    Annualized information ratio (Grinold & Kahn, 2000)

    :param returns: periodic simple returns of the portfolio
    :param benchmark: periodic simple returns of the benchmark
    :param periods_per_year: observations per year; inferred from the index when omitted
    :return: mean active return over tracking error
    """
    active = (returns - benchmark.reindex(returns.index)).dropna()
    return _safe_div(active.mean(), active.std(ddof=1)) * np.sqrt(_ppy(returns, periods_per_year))


def _moments(returns: pd.Series) -> tuple[float, float, float, int]:
    r = returns.to_numpy(dtype=float)
    n = len(r)
    if n < 3:
        raise MqValueError('at least three observations are required')
    sr = _safe_div(r.mean(), r.std(ddof=1))
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))
    return sr, skew, kurt, n


def _psr(sr: float, sr_benchmark: float, skew: float, kurt: float, n: int) -> float:
    variance = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    return float(stats.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / np.sqrt(max(variance, 1e-12))))


@_columnwise
def probabilistic_sharpe_ratio(
    returns: ReturnsLike, benchmark_sharpe: float = 0.0, periods_per_year: Optional[float] = None
) -> float:
    """
    Probability that the true Sharpe ratio exceeds a benchmark (Bailey & López de Prado, 2012)

    :param returns: periodic simple returns
    :param benchmark_sharpe: annualized Sharpe ratio to beat
    :param periods_per_year: observations per year; inferred from the index when omitted
    :return: probability in [0, 1]

    **Usage**

    :math:`PSR = \\Phi\\left(\\frac{(\\widehat{SR} - SR^*)\\sqrt{T - 1}}
    {\\sqrt{1 - \\gamma_3 \\widehat{SR} + \\frac{\\gamma_4 - 1}{4} \\widehat{SR}^2}}\\right)`, with per-period
    Sharpe ratios, skewness :math:`\\gamma_3` and kurtosis :math:`\\gamma_4`. Negative skewness and fat tails
    widen the confidence band of the estimated Sharpe ratio.
    """
    sr, skew, kurt, n = _moments(returns)
    sr_benchmark = benchmark_sharpe / np.sqrt(_ppy(returns, periods_per_year))
    return _psr(sr, sr_benchmark, skew, kurt, n)


def expected_maximum_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """
    Expected maximum Sharpe ratio among independent trials with zero true Sharpe ratio

    :param n_trials: number of independent strategy configurations tried
    :param sharpe_variance: variance of the Sharpe ratios across trials
    :return: :math:`\\sqrt{V}\\left((1 - \\gamma)\\Phi^{-1}(1 - 1/N) + \\gamma \\Phi^{-1}(1 - 1/(N e))\\right)`,
        where :math:`\\gamma` is the Euler-Mascheroni constant (Bailey & López de Prado, 2014)
    """
    if n_trials < 2:
        return 0.0
    return float(
        np.sqrt(sharpe_variance)
        * (
            (1 - EULER_MASCHERONI) * stats.norm.ppf(1 - 1 / n_trials)
            + EULER_MASCHERONI * stats.norm.ppf(1 - 1 / (n_trials * np.e))
        )
    )


@_columnwise
def deflated_sharpe_ratio(
    returns: ReturnsLike, trial_sharpe_ratios: Sequence[float], periods_per_year: Optional[float] = None
) -> float:
    """
    Deflated Sharpe ratio: significance of a Sharpe ratio after correcting for multiple testing

    :param returns: periodic simple returns of the selected strategy
    :param trial_sharpe_ratios: annualized Sharpe ratios of all the configurations that were tried
    :param periods_per_year: observations per year; inferred from the index when omitted
    :return: probability in [0, 1] that the true Sharpe ratio is positive given the selection process

    **Usage**

    When many backtests are run, the best one will show a high Sharpe ratio by chance alone. The deflated Sharpe
    ratio is the probabilistic Sharpe ratio evaluated against the expected maximum Sharpe ratio of the trials
    (Bailey & López de Prado, 2014, "The Deflated Sharpe Ratio", Journal of Portfolio Management 40(5)).
    Values above 0.95 indicate a discovery that is unlikely to be a false positive.
    """
    ppy = _ppy(returns, periods_per_year)
    trials = np.asarray(trial_sharpe_ratios, dtype=float) / np.sqrt(ppy)
    sr_benchmark = expected_maximum_sharpe(len(trials), float(np.var(trials, ddof=1)) if len(trials) > 1 else 0.0)
    sr, skew, kurt, n = _moments(returns)
    return _psr(sr, sr_benchmark, skew, kurt, n)


@_columnwise
def minimum_track_record_length(
    returns: ReturnsLike,
    benchmark_sharpe: float = 0.0,
    confidence: float = 0.95,
    periods_per_year: Optional[float] = None,
) -> float:
    """
    Minimum number of observations needed for the Sharpe ratio to be significant (Bailey & López de Prado, 2012)

    :param returns: periodic simple returns
    :param benchmark_sharpe: annualized Sharpe ratio to beat
    :param confidence: required confidence level
    :param periods_per_year: observations per year; inferred from the index when omitted
    :return: number of observations; infinite when the Sharpe ratio does not exceed the benchmark
    """
    sr, skew, kurt, _ = _moments(returns)
    sr_benchmark = benchmark_sharpe / np.sqrt(_ppy(returns, periods_per_year))
    if sr <= sr_benchmark:
        return np.inf
    variance = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    return float(1 + variance * (stats.norm.ppf(confidence) / (sr - sr_benchmark)) ** 2)


def performance_summary(
    returns: ReturnsLike,
    risk_free: float = 0.0,
    benchmark: Optional[pd.Series] = None,
    periods_per_year: Optional[float] = None,
) -> Union[pd.Series, pd.DataFrame]:
    """
    Table of the main performance and risk statistics

    :param returns: periodic simple returns; a DataFrame produces one column per strategy
    :param risk_free: annual risk-free rate used by the Sharpe ratio
    :param benchmark: optional benchmark returns for the information ratio
    :param periods_per_year: observations per year; inferred from the index when omitted
    :return: statistics indexed by name
    """
    if isinstance(returns, pd.DataFrame):
        return pd.DataFrame(
            {col: performance_summary(returns[col], risk_free, benchmark, periods_per_year) for col in returns}
        )
    r = returns.dropna()
    ppy = _ppy(r, periods_per_year)
    stats_ = {
        'annualized_return': annualized_return(r, ppy),
        'annualized_volatility': annualized_volatility(r, ppy),
        'sharpe_ratio': sharpe_ratio(r, risk_free, ppy),
        'sortino_ratio': sortino_ratio(r, risk_free, ppy),
        'max_drawdown': max_drawdown(r),
        'calmar_ratio': calmar_ratio(r, ppy),
        'omega_ratio': omega_ratio(r),
        'skewness': float(stats.skew(r)),
        'excess_kurtosis': float(stats.kurtosis(r)),
        'var_95': value_at_risk(r, 0.95),
        'expected_shortfall_95': expected_shortfall(r, 0.95),
        'probabilistic_sharpe_ratio': probabilistic_sharpe_ratio(r, 0.0, ppy),
    }
    if benchmark is not None:
        stats_['information_ratio'] = information_ratio(r, benchmark, ppy)
    return pd.Series(stats_, name=returns.name)
