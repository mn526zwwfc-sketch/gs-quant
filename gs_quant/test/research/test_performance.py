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
from scipy import stats

from gs_quant.errors import MqValueError
from gs_quant.research import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    deflated_sharpe_ratio,
    drawdown,
    expected_maximum_sharpe,
    infer_periods_per_year,
    information_ratio,
    max_drawdown,
    minimum_track_record_length,
    omega_ratio,
    performance_summary,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    sortino_ratio,
)


@pytest.fixture
def daily():
    rng = np.random.default_rng(7)
    index = pd.bdate_range('2015-01-01', periods=1000)
    return pd.Series(rng.normal(0.001, 0.01, 1000), index=index, name='strategy')


def test_infer_periods_per_year():
    assert infer_periods_per_year(pd.bdate_range('2020-01-01', periods=10)) == 252
    assert infer_periods_per_year(pd.date_range('2020-01-01', periods=10, freq='W-FRI')) == 52
    assert infer_periods_per_year(pd.date_range('2020-01-31', periods=10, freq='BME')) == 12
    assert infer_periods_per_year(pd.date_range('2020-03-31', periods=10, freq='QE')) == 4
    assert infer_periods_per_year(pd.date_range('2020-12-31', periods=10, freq='YE')) == 1
    assert infer_periods_per_year(pd.RangeIndex(10)) == 252


def test_annualized_return_and_volatility(daily):
    growth = np.prod(1 + daily.to_numpy())
    assert np.isclose(annualized_return(daily), growth ** (252 / 1000) - 1)
    assert np.isclose(annualized_return(daily, periods_per_year=12), growth ** (12 / 1000) - 1)
    assert np.isclose(annualized_volatility(daily), daily.std() * np.sqrt(252))
    assert annualized_return(pd.Series([-1.0, 0.1])) == -1.0
    assert np.isnan(annualized_return(pd.Series([], dtype=float)))


def test_sharpe_and_sortino(daily):
    rf = 0.02
    excess = daily - rf / 252
    assert np.isclose(sharpe_ratio(daily, rf), excess.mean() / excess.std() * np.sqrt(252))
    downside = np.sqrt(np.mean(np.minimum(daily, 0) ** 2))
    assert np.isclose(sortino_ratio(daily), daily.mean() / downside * np.sqrt(252))
    assert np.isnan(sharpe_ratio(pd.Series([0.01, 0.01, 0.01])))
    assert np.isnan(sortino_ratio(pd.Series([0.01, 0.02, 0.03])))


def test_columnwise_evaluation(daily):
    frame = pd.DataFrame({'a': daily, 'b': daily * 2})
    frame.iloc[:10, 1] = np.nan
    result = sharpe_ratio(frame)
    assert isinstance(result, pd.Series)
    assert np.isclose(result['a'], sharpe_ratio(daily))
    assert np.isclose(result['b'], sharpe_ratio(daily.iloc[10:]))
    assert np.isclose(sharpe_ratio(daily.to_numpy(), periods_per_year=252), sharpe_ratio(daily))


def test_drawdown():
    r = pd.Series([-0.1, 0.05, 0.1, -0.2, 0.1])
    wealth = np.cumprod(1 + r.to_numpy())
    dd = drawdown(r)
    assert np.isclose(dd.iloc[0], -0.1)  # measured from the initial investment
    assert np.isclose(dd.iloc[2], 0)
    assert np.isclose(dd.iloc[3], -0.2)
    assert np.isclose(max_drawdown(r), -0.2)
    assert np.isclose(calmar_ratio(r, periods_per_year=5), (wealth[-1] - 1) / 0.2)
    assert isinstance(drawdown(pd.DataFrame({'a': r})), pd.DataFrame)
    assert np.isnan(max_drawdown(pd.Series([], dtype=float)))


def test_omega_ratio():
    r = pd.Series([0.02, -0.01, 0.03, -0.02])
    assert np.isclose(omega_ratio(r), 0.05 / 0.03)
    assert np.isclose(omega_ratio(r, threshold=0.01), (0.01 + 0.02) / (0.02 + 0.03))


def test_information_ratio(daily):
    benchmark = daily * 0.5 + 0.0001
    active = daily - benchmark
    assert np.isclose(information_ratio(daily, benchmark), active.mean() / active.std() * np.sqrt(252))


def test_probabilistic_sharpe_ratio(daily):
    sr = daily.mean() / daily.std()
    skew, kurt = stats.skew(daily), stats.kurtosis(daily, fisher=False)
    expected = stats.norm.cdf(sr * np.sqrt(999) / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2))
    assert np.isclose(probabilistic_sharpe_ratio(daily), expected)
    assert probabilistic_sharpe_ratio(daily, benchmark_sharpe=5.0) < 0.01
    assert probabilistic_sharpe_ratio(daily, benchmark_sharpe=-5.0) > 0.99
    with pytest.raises(MqValueError):
        probabilistic_sharpe_ratio(pd.Series([0.1, 0.2]))


def test_probabilistic_sharpe_ratio_penalizes_negative_skew():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 2000)
    symmetric = pd.Series(0.0005 + 0.01 * (base - base.mean()) / base.std())
    crash = rng.choice([0.0, -8.0], size=2000, p=[0.98, 0.02]) + rng.normal(0, 0.5, 2000)
    skewed = pd.Series(0.0005 + 0.01 * (crash - crash.mean()) / crash.std())
    assert stats.skew(skewed) < -3
    assert np.isclose(sharpe_ratio(skewed), sharpe_ratio(symmetric))
    assert probabilistic_sharpe_ratio(skewed) < probabilistic_sharpe_ratio(symmetric)


def test_expected_maximum_sharpe():
    assert expected_maximum_sharpe(1, 1.0) == 0.0
    # Bailey & López de Prado (2014): the expected maximum of N standard normals grows like sqrt(2 ln N)
    rng = np.random.default_rng(0)
    simulated = rng.normal(size=(20_000, 100)).max(axis=1).mean()
    assert np.isclose(expected_maximum_sharpe(100, 1.0), simulated, rtol=0.02)
    assert np.isclose(expected_maximum_sharpe(100, 4.0), 2 * expected_maximum_sharpe(100, 1.0))


def test_deflated_sharpe_ratio(daily):
    single = deflated_sharpe_ratio(daily, [sharpe_ratio(daily)])
    assert np.isclose(single, probabilistic_sharpe_ratio(daily))
    many = deflated_sharpe_ratio(daily, np.linspace(-1, 1.5, 200))
    assert many < single
    frame = deflated_sharpe_ratio(pd.DataFrame({'a': daily}), [0.5, 1.0, 1.5])
    assert isinstance(frame, pd.Series)


def test_minimum_track_record_length(daily):
    sr = daily.mean() / daily.std()
    assert sr > 0
    skew, kurt = stats.skew(daily), stats.kurtosis(daily, fisher=False)
    expected = 1 + (1 - skew * sr + (kurt - 1) / 4 * sr**2) * (stats.norm.ppf(0.95) / sr) ** 2
    assert np.isclose(minimum_track_record_length(daily), expected)
    assert minimum_track_record_length(daily, benchmark_sharpe=10) == np.inf


def test_performance_summary(daily):
    summary = performance_summary(daily, risk_free=0.01, benchmark=daily * 0.5)
    assert summary.name == 'strategy'
    assert np.isclose(summary['sharpe_ratio'], sharpe_ratio(daily, 0.01))
    assert 'information_ratio' in summary.index
    assert summary['var_95'] > 0 and summary['expected_shortfall_95'] > summary['var_95']
    table = performance_summary(pd.DataFrame({'a': daily, 'b': -daily}))
    assert list(table.columns) == ['a', 'b']
    assert 'information_ratio' not in table.index
