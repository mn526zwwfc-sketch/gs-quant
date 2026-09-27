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
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from gs_quant.errors import MqValueError
from gs_quant.research import (
    ReturnType,
    aggregate_returns,
    align,
    clean_prices,
    compute_returns,
    cross_sectional_rank,
    cross_sectional_zscore,
    detect_outliers,
    hampel_filter,
    lag_signal,
    point_in_time,
    validate_prices,
    winsorize,
)


@pytest.fixture
def prices():
    rng = np.random.default_rng(0)
    index = pd.bdate_range('2021-01-01', periods=200)
    values = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (200, 3)), axis=0))
    return pd.DataFrame(values, index=index, columns=['a', 'b', 'c'])


def test_validate_prices_clean(prices):
    report = validate_prices(prices)
    assert report.n_observations == 200
    assert report.duplicated_dates == 0 and report.is_sorted
    assert report.missing.sum() == 0 and report.non_positive.sum() == 0
    assert report.is_clean
    assert list(report.to_frame().columns) == ['missing', 'non_positive', 'longest_stale_run', 'outliers']


def test_validate_prices_detects_errors(prices):
    dirty = prices.copy()
    dirty.iloc[10, 0] = 0.0
    dirty.iloc[50, 1] *= 3
    dirty.iloc[100:110, 2] = dirty.iloc[99, 2]
    dirty.iloc[150, 1] = np.nan
    dirty = pd.concat([dirty, dirty.iloc[[-1]]])
    report = validate_prices(dirty, stale_threshold=5)
    assert report.non_positive['a'] == 1
    assert report.missing['b'] == 1
    assert report.longest_stale_run['c'] == 10
    assert report.longest_stale_run['a'] == 0
    assert report.outliers['b'] >= 1
    assert report.duplicated_dates == 1
    assert not report.is_clean


def test_validate_prices_series(prices):
    report = validate_prices(prices['a'])
    assert report.missing.index.tolist() == ['a']


def test_clean_prices(prices):
    dirty = prices.iloc[:10].copy()
    dirty.iloc[3, 0] = -1.0
    dirty.iloc[4:8, 1] = np.nan
    dirty = pd.concat([dirty.iloc[5:], dirty.iloc[:5]])
    cleaned = clean_prices(dirty, max_fill=2)
    assert cleaned.index.is_monotonic_increasing
    assert cleaned.iloc[3, 0] == prices.iloc[2, 0]  # forward-filled, never back-filled
    assert cleaned.iloc[4:6, 1].tolist() == [prices.iloc[3, 1]] * 2
    assert cleaned.iloc[6:8, 1].isna().all()  # beyond the fill limit
    series = clean_prices(dirty['a'])
    assert isinstance(series, pd.Series)


def test_clean_prices_duplicates_keep_last(prices):
    dup = pd.concat([prices.iloc[:3], prices.iloc[[2]] * 2])
    cleaned = clean_prices(dup)
    assert len(cleaned) == 3
    assert cleaned.iloc[-1, 0] == prices.iloc[2, 0] * 2


def test_detect_outliers_and_hampel():
    rng = np.random.default_rng(1)
    x = pd.Series(rng.normal(0, 1, 300))
    x.iloc[150] = 25
    mask = detect_outliers(x, window=21, n_sigmas=5)
    assert mask.iloc[150] and mask.sum() == 1
    assert not mask.iloc[:20].any()  # causal window needs a full window of history
    filtered = hampel_filter(x, window=21, n_sigmas=5)
    assert abs(filtered.iloc[150]) < 3
    assert_series_equal(filtered.drop(150), x.drop(150), check_names=False)


def test_detect_outliers_centered_and_zero_mad():
    x = pd.DataFrame({'flat': [1.0] * 30, 'spike': [0.0, 1.0] * 15})
    x.iloc[20, 1] = 50
    mask = detect_outliers(x, window=5, n_sigmas=3, center=True)
    assert not mask['flat'].any()
    assert mask['spike'].iloc[20]
    assert not detect_outliers(x.iloc[:3], window=5).any().any()


def test_winsorize():
    df = pd.DataFrame({'a': np.arange(101.0), 'b': np.arange(101.0) * 2})
    out = winsorize(df, 0.05, 0.95)
    assert out['a'].min() == 5 and out['a'].max() == 95
    assert out['b'].max() == 190
    rows = winsorize(df.T, 0.1, 0.9, axis=1)
    assert rows.loc['a'].min() == 10 and rows.loc['a'].max() == 90
    series = winsorize(df['a'], 0.1, 0.9)
    assert series.max() == 90
    with pytest.raises(MqValueError):
        winsorize(df, 0.9, 0.1)


def test_cross_sectional_transforms():
    x = pd.DataFrame([[1.0, 2.0, 3.0], [3.0, np.nan, 1.0], [5.0, 5.0, 5.0]], columns=list('abc'))
    z = cross_sectional_zscore(x)
    assert np.allclose(z.iloc[0], [-1, 0, 1])
    assert z.iloc[2].isna().all()
    r = cross_sectional_rank(x)
    assert np.allclose(r.iloc[0], [-0.5, 0, 0.5])
    assert np.allclose(r.iloc[1].dropna(), [0.5, -0.5])


def test_compute_returns(prices):
    simple = compute_returns(prices)
    assert len(simple) == len(prices) - 1
    assert np.isclose(simple.iloc[0, 0], prices.iloc[1, 0] / prices.iloc[0, 0] - 1)
    log = compute_returns(prices['a'], ReturnType.LOG, periods=5)
    assert np.isclose(log.iloc[0], np.log(prices['a'].iloc[5] / prices['a'].iloc[0]))


def test_aggregate_returns(prices):
    daily = compute_returns(prices)
    monthly = aggregate_returns(daily, 'M')
    jan = daily.loc['2021-01']
    assert monthly.index[0] == jan.index[-1]
    assert np.isclose(monthly.iloc[0, 0], (1 + jan['a']).prod() - 1)
    log_monthly = aggregate_returns(np.log(1 + daily['a']), 'M', ReturnType.LOG)
    assert np.isclose(np.exp(log_monthly.iloc[0]) - 1, monthly.iloc[0, 0])


def test_align(prices):
    a, b = align(prices.iloc[:50], prices['b'].iloc[20:80])
    assert len(a) == len(b) == 30
    assert a.index.equals(b.index)
    assert align() == []


def test_point_in_time():
    fundamentals = pd.Series([1.0, 2.0], index=pd.to_datetime(['2020-12-31', '2021-03-31']))
    days = pd.bdate_range('2021-01-01', '2021-08-31')
    known = point_in_time(fundamentals, days, publication_lag='90D')
    # 2020-12-31 + 90 days = 2021-03-31 and 2021-03-31 + 90 days = 2021-06-29
    assert known.loc[:'2021-03-30'].isna().all()
    assert (known.loc['2021-03-31':'2021-06-28'] == 1.0).all()
    assert (known.loc['2021-06-29':] == 2.0).all()
    frame = point_in_time(fundamentals.to_frame('bv'), days, pd.DateOffset(months=6))
    assert frame.loc['2021-07-01', 'bv'] == 1.0


def test_lag_signal(prices):
    assert_frame_equal(lag_signal(prices, 2), prices.shift(2))
    with pytest.raises(MqValueError):
        lag_signal(prices, -1)
