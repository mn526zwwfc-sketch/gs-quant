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

Data management for empirical finance.

Every result downstream (risk, valuation, backtests) is only as good as the data it is built on. This module
implements the data hygiene practices recommended in the academic literature:

* Robust outlier detection with the Hampel filter (Hampel, 1974), using the median absolute deviation
  (Rousseeuw & Croux, 1993) rather than the non-robust standard deviation.
* Winsorization of extreme observations (Tukey, 1962), standard in cross-sectional asset pricing
  (Fama & French, 1992).
* Stale price detection: stale quotes bias volatility down and create spurious autocorrelation
  (Lo & MacKinlay, 1990; Getmansky, Lo & Makarov, 2004).
* Avoiding look-ahead bias: only forward-fill (never back-fill), align accounting data on the date it became
  public, and lag signals before trading (Fama & French, 1993; Harvey, Liu & Zhu, 2016; López de Prado, 2018).
"""

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Union

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from gs_quant.errors import MqValueError

FrameOrSeries = Union[pd.DataFrame, pd.Series]

# Scale factor that makes the MAD a consistent estimator of the standard deviation under normality
MAD_SCALE = 1.4826


class ReturnType(Enum):
    SIMPLE = 'simple'
    LOG = 'log'


def _to_frame(x: FrameOrSeries) -> tuple[pd.DataFrame, bool]:
    if isinstance(x, pd.Series):
        return x.to_frame(name=x.name if x.name is not None else 0), True
    if isinstance(x, pd.DataFrame):
        return x, False
    raise MqValueError('expected a pandas Series or DataFrame')


def _restore(df: pd.DataFrame, was_series: bool) -> FrameOrSeries:
    return df.iloc[:, 0] if was_series else df


@dataclass
class DataQualityReport:
    """
    Summary of data quality issues found by :func:`validate_prices`, one entry per column
    """

    n_observations: int
    missing: pd.Series
    non_positive: pd.Series
    longest_stale_run: pd.Series
    outliers: pd.Series
    duplicated_dates: int
    is_sorted: bool

    @property
    def is_clean(self) -> bool:
        """True when no issue at all was detected"""
        return (
            self.is_sorted
            and self.duplicated_dates == 0
            and int(self.missing.sum()) == 0
            and int(self.non_positive.sum()) == 0
            and int(self.outliers.sum()) == 0
        )

    def to_frame(self) -> pd.DataFrame:
        """Per-column table of the detected issues"""
        return pd.DataFrame(
            {
                'missing': self.missing,
                'non_positive': self.non_positive,
                'longest_stale_run': self.longest_stale_run,
                'outliers': self.outliers,
            }
        )


def _longest_stale_run(s: pd.Series) -> int:
    s = s.dropna()
    if s.empty:
        return 0
    run_id = s.ne(s.shift()).cumsum()
    return int(s.groupby(run_id).size().max() - 1)


def validate_prices(
    prices: FrameOrSeries, stale_threshold: int = 5, outlier_window: int = 21, n_sigmas: float = 5.0
) -> DataQualityReport:
    """
    Audit a price history for the most common data errors

    :param prices: price levels (dates x assets)
    :param stale_threshold: only runs of unchanged prices at least this long are reported as stale
    :param outlier_window: rolling window of the Hampel filter applied to returns
    :param n_sigmas: number of robust standard deviations beyond which a return is flagged as an outlier
    :return: a :class:`DataQualityReport`

    **Usage**

    Checks for missing values, non-positive prices, stale prices (runs of identical prints), unsorted or
    duplicated dates and suspicious jumps (Hampel filter on simple returns).
    """
    df, _ = _to_frame(prices)
    stale = df.apply(_longest_stale_run)
    stale = stale.where(stale >= stale_threshold, 0)
    rets = compute_returns(df.where(df > 0))
    outliers = detect_outliers(rets, window=outlier_window, n_sigmas=n_sigmas).sum()
    return DataQualityReport(
        n_observations=len(df),
        missing=df.isna().sum(),
        non_positive=(df <= 0).sum(),
        longest_stale_run=stale,
        outliers=outliers,
        duplicated_dates=int(df.index.duplicated().sum()),
        is_sorted=bool(df.index.is_monotonic_increasing),
    )


def clean_prices(prices: FrameOrSeries, max_fill: int = 5, drop_non_positive: bool = True) -> FrameOrSeries:
    """
    Apply look-ahead-free cleaning to a price history

    :param prices: price levels (dates x assets)
    :param max_fill: maximum number of consecutive missing observations to forward-fill
    :param drop_non_positive: treat zero and negative prices as missing
    :return: cleaned prices with the same shape as the input

    **Usage**

    Sorts the dates, keeps the last print of duplicated dates, removes non-positive prices and forward-fills
    short gaps. Data is never back-filled, as that would leak future information into the past.
    """
    df, was_series = _to_frame(prices)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='last')]
    if drop_non_positive:
        df = df.where(df > 0)
    df = df.ffill(limit=max_fill) if max_fill > 0 else df
    df = df.dropna(how='all')
    return _restore(df, was_series)


def _rolling_median_mad(values: np.ndarray, window: int, center: bool) -> tuple[np.ndarray, np.ndarray]:
    n = values.shape[0]
    median = np.full(values.shape, np.nan)
    mad = np.full(values.shape, np.nan)
    if n < window:
        return median, mad
    start = window // 2 if center else window - 1
    stop = start + n - window + 1
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)  # all-NaN windows
        for col in range(values.shape[1]):  # column by column to bound memory usage
            windows = sliding_window_view(values[:, col], window)  # (n - window + 1, window)
            med = np.nanmedian(windows, axis=-1)
            median[start:stop, col] = med
            mad[start:stop, col] = np.nanmedian(np.abs(windows - med[:, None]), axis=-1)
    return median, mad


def detect_outliers(x: FrameOrSeries, window: int = 21, n_sigmas: float = 3.0, center: bool = False) -> FrameOrSeries:
    """
    Flag outliers with the Hampel filter

    :param x: observations to screen, typically returns
    :param window: size of the rolling window
    :param n_sigmas: number of robust standard deviations beyond which a point is an outlier
    :param center: use a centred window. Centred windows look into the future, so keep the default (causal)
        whenever the result feeds a trading decision
    :return: boolean mask with the same shape as the input

    **Usage**

    A point is flagged when :math:`|x_t - m_t| > k \\cdot 1.4826 \\cdot MAD_t`, where :math:`m_t` is the rolling
    median and :math:`MAD_t` the rolling median absolute deviation (Hampel, 1974). Windows with zero dispersion
    (e.g. stale data) never flag points.
    """
    df, was_series = _to_frame(x)
    values = df.to_numpy(dtype=float)
    median, mad = _rolling_median_mad(values, window, center)
    with np.errstate(invalid='ignore'):
        mask = (mad > 0) & (np.abs(values - median) > n_sigmas * MAD_SCALE * mad)
    return _restore(pd.DataFrame(mask, index=df.index, columns=df.columns), was_series)


def hampel_filter(x: FrameOrSeries, window: int = 21, n_sigmas: float = 3.0, center: bool = False) -> FrameOrSeries:
    """
    Replace outliers by the rolling median (Hampel filter)

    :param x: observations to clean, typically returns
    :param window: size of the rolling window
    :param n_sigmas: number of robust standard deviations beyond which a point is replaced
    :param center: use a centred window (looks into the future, see :func:`detect_outliers`)
    :return: cleaned observations
    """
    df, was_series = _to_frame(x)
    values = df.to_numpy(dtype=float)
    median, _ = _rolling_median_mad(values, window, center)
    mask = _to_frame(detect_outliers(df, window, n_sigmas, center))[0].to_numpy()
    cleaned = np.where(mask, median, values)
    return _restore(pd.DataFrame(cleaned, index=df.index, columns=df.columns), was_series)


def winsorize(x: FrameOrSeries, lower: float = 0.01, upper: float = 0.99, axis: int = 0) -> FrameOrSeries:
    """
    Clip observations at the given quantiles

    :param x: observations
    :param lower: lower quantile, between 0 and 1
    :param upper: upper quantile, between 0 and 1
    :param axis: 0 clips each column with its own quantiles (time series), 1 clips each row (cross-section)
    :return: winsorized observations

    **Usage**

    Cross-sectional winsorization (``axis=1``) only uses information available on each date and is therefore
    safe for backtests. Time-series winsorization (``axis=0``) uses full-sample quantiles and should be
    reserved for descriptive statistics.
    """
    if not 0 <= lower < upper <= 1:
        raise MqValueError('quantiles must satisfy 0 <= lower < upper <= 1')
    df, was_series = _to_frame(x)
    lo = df.quantile(lower, axis=axis)
    hi = df.quantile(upper, axis=axis)
    return _restore(df.clip(lower=lo, upper=hi, axis=1 - axis), was_series)


def cross_sectional_zscore(x: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize each date across assets

    :param x: signal values (dates x assets)
    :return: z-scores with zero mean and unit standard deviation on each date
    """
    std = x.std(axis=1).replace(0, np.nan)
    return x.sub(x.mean(axis=1), axis=0).div(std, axis=0)


def cross_sectional_rank(x: pd.DataFrame) -> pd.DataFrame:
    """
    Map each date's values to centred ranks in [-0.5, 0.5]

    :param x: signal values (dates x assets)
    :return: centred percentile ranks

    **Usage**

    Rank transforms are robust to outliers and make signals with different units comparable
    (Grinold & Kahn, 2000; Asness, Moskowitz & Pedersen, 2013).
    """
    counts = x.notna().sum(axis=1)
    ranks = x.rank(axis=1)
    return ranks.sub(1).div((counts - 1).replace(0, np.nan), axis=0) - 0.5


def compute_returns(
    prices: FrameOrSeries, return_type: ReturnType = ReturnType.SIMPLE, periods: int = 1
) -> FrameOrSeries:
    """
    Compute returns from price levels

    :param prices: price levels (dates x assets)
    :param return_type: simple returns :math:`P_t / P_{t-k} - 1` or log returns :math:`\\ln(P_t / P_{t-k})`
    :param periods: number of observations :math:`k` over which the return is measured
    :return: returns, without the leading rows that cannot be computed
    """
    df, was_series = _to_frame(prices)
    ratio = df / df.shift(periods)
    rets = np.log(ratio) if ReturnType(return_type) == ReturnType.LOG else ratio - 1
    return _restore(rets.iloc[periods:], was_series)


def aggregate_returns(
    returns: FrameOrSeries, frequency: str = 'M', return_type: ReturnType = ReturnType.SIMPLE
) -> FrameOrSeries:
    """
    Compound returns to a lower frequency

    :param returns: periodic returns with a DatetimeIndex
    :param frequency: pandas period alias, e.g. 'W', 'M', 'Q' or 'Y'
    :param return_type: whether the inputs are simple (compounded) or log (summed) returns
    :return: returns at the lower frequency, labelled with the last observed date of each period
    """
    df, was_series = _to_frame(returns)
    periods = df.index.to_period(frequency)
    if ReturnType(return_type) == ReturnType.LOG:
        out = df.groupby(periods).sum(min_count=1)
    else:
        out = (1 + df).groupby(periods).prod(min_count=1) - 1
    out.index = pd.DatetimeIndex(df.index.to_series().groupby(periods).max().to_numpy())
    return _restore(out, was_series)


def align(*data: FrameOrSeries) -> list[FrameOrSeries]:
    """
    Restrict several datasets to their common dates

    :param data: series or frames with comparable indices
    :return: the inputs restricted to the intersection of their indices, in the same order
    """
    if not data:
        return []
    common = data[0].index
    for d in data[1:]:
        common = common.intersection(d.index)
    common = common.sort_values()
    return [d.loc[common] for d in data]


def point_in_time(
    data: FrameOrSeries, target_index: pd.DatetimeIndex, publication_lag: Union[str, pd.DateOffset] = '0D'
) -> FrameOrSeries:
    """
    Align low-frequency data (e.g. fundamentals) on the dates it was actually available

    :param data: observations indexed by the end of the period they describe
    :param target_index: dates on which the data is needed, typically trading dates
    :param publication_lag: delay between period end and publication, e.g. '90D'. Fama & French (1993) assume at
        least six months for annual accounting data
    :return: for every target date, the latest value that had been published by that date

    **Usage**

    Using data before its publication date is one of the most common sources of look-ahead bias in backtests.
    This function shifts the index by the publication lag and performs an as-of join.
    """
    df, was_series = _to_frame(data)
    lag = pd.tseries.frequencies.to_offset(publication_lag) if isinstance(publication_lag, str) else publication_lag
    shifted = df.sort_index()
    shifted.index = shifted.index + lag
    shifted = shifted[~shifted.index.duplicated(keep='last')]
    return _restore(shifted.reindex(pd.DatetimeIndex(target_index), method='ffill'), was_series)


def lag_signal(signal: FrameOrSeries, periods: int = 1) -> FrameOrSeries:
    """
    Delay a signal so that it only uses information available before trading

    :param signal: signal or target weights computed with data up to each date
    :param periods: number of observations to delay
    :return: lagged signal
    """
    if periods < 0:
        raise MqValueError('a negative lag would introduce look-ahead bias')
    return signal.shift(periods)
