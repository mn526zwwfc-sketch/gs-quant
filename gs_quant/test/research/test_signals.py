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

from gs_quant.errors import MqValueError
from gs_quant.research import (
    combine_signals,
    cross_sectional_momentum,
    ex_ante_volatility,
    low_volatility,
    moving_average_crossover,
    short_term_reversal,
    time_series_momentum,
    volatility_scaling,
)


@pytest.fixture
def prices():
    rng = np.random.default_rng(3)
    index = pd.bdate_range('2018-01-01', periods=600)
    drifts = np.array([0.001, 0.0005, 0.0, -0.0005, -0.001])
    vols = np.array([0.005, 0.01, 0.015, 0.02, 0.025])
    returns = drifts + rng.normal(size=(600, 5)) * vols
    return pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)), index=index, columns=list('abcde'))


def _no_look_ahead(strategy, prices, **kwargs):
    """Changing prices after a date must not change the weights up to that date"""
    full = strategy(prices, **kwargs)
    shocked = prices.copy()
    shocked.iloc[400:] *= np.random.default_rng(0).uniform(0.5, 1.5, shocked.iloc[400:].shape)
    truncated = strategy(shocked, **kwargs)
    pd.testing.assert_frame_equal(full.iloc[:400], truncated.iloc[:400])


def test_ex_ante_volatility(prices):
    vol = ex_ante_volatility(prices)
    assert vol.iloc[:60].isna().all().all()
    assert vol.iloc[-1].is_monotonic_increasing  # assets are sorted by volatility
    assert np.isclose(vol.iloc[-1]['c'], 0.015 * np.sqrt(261), rtol=0.3)


def test_time_series_momentum(prices):
    w = time_series_momentum(prices, lookback=126)
    last = w.iloc[-1]
    assert (np.sign(last) == np.sign(prices.iloc[-1] / prices.iloc[-127] - 1)).all()
    vol = ex_ante_volatility(prices).iloc[-1]
    assert np.allclose(last.abs(), 0.40 / vol / 5)
    assert w.iloc[:126].isna().all().all()
    _no_look_ahead(time_series_momentum, prices, lookback=126)


def test_cross_sectional_momentum(prices):
    w = cross_sectional_momentum(prices, lookback=252, skip=21, quantile=0.2)
    last = w.iloc[-1]
    score = prices.iloc[-22] / prices.iloc[-253] - 1
    assert last[score.idxmax()] == 1.0 and last[score.idxmin()] == -1.0
    assert np.isclose(last.sum(), 0)
    assert w.iloc[:252].isna().all().all()
    long_only = cross_sectional_momentum(prices, quantile=0.4, long_only=True).iloc[-1]
    assert np.isclose(long_only.sum(), 1) and (long_only >= 0).all() and (long_only > 0).sum() == 2
    with pytest.raises(MqValueError):
        cross_sectional_momentum(prices, lookback=21, skip=21)
    with pytest.raises(MqValueError):
        cross_sectional_momentum(prices, quantile=0.8)
    _no_look_ahead(cross_sectional_momentum, prices)


def test_cross_sectional_momentum_small_universe(prices):
    # with 3 assets and a 20% quantile the short leg is empty: the long leg must survive
    w = cross_sectional_momentum(prices.iloc[:, :3], quantile=0.2).iloc[-1]
    assert np.isclose(w.sum(), 1)


def test_short_term_reversal(prices):
    w = short_term_reversal(prices, lookback=5)
    last = w.iloc[-1]
    past = prices.iloc[-1] / prices.iloc[-6] - 1
    assert np.isclose(last.sum(), 0) and np.isclose(last.abs().sum(), 2)
    assert last[past.idxmax()] == last.min() and last[past.idxmin()] == last.max()
    assert np.isclose(np.corrcoef(last, -(past - past.mean()))[0, 1], 1)
    _no_look_ahead(short_term_reversal, prices, lookback=5)


def test_moving_average_crossover(prices):
    w = moving_average_crossover(prices, fast=20, slow=100)
    assert w.iloc[:99].isna().all().all()
    last = w.iloc[-1]
    fast, slow = prices.iloc[-20:].mean(), prices.iloc[-100:].mean()
    assert ((last > 0) == (fast > slow)).all()
    assert set(np.round(last.unique(), 10)) <= {0.0, 0.2}
    long_short = moving_average_crossover(prices, fast=20, slow=100, long_short=True).iloc[-1]
    assert set(np.round(long_short.unique(), 10)) <= {-0.2, 0.2}
    with pytest.raises(MqValueError):
        moving_average_crossover(prices, fast=100, slow=20)
    _no_look_ahead(moving_average_crossover, prices, fast=20, slow=100)


def test_low_volatility(prices):
    w = low_volatility(prices, lookback=126, quantile=0.2).iloc[-1]
    assert w['a'] == 1.0 and w.drop('a').eq(0).all()
    long_short = low_volatility(prices, lookback=126, quantile=0.2, long_only=False).iloc[-1]
    assert long_short['a'] == 1.0 and long_short['e'] == -1.0
    _no_look_ahead(low_volatility, prices, lookback=126)


def test_volatility_scaling(prices):
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    weights['e'] = 1.0
    scaled = volatility_scaling(weights, prices, target_volatility=0.10, lookback=63, max_leverage=10)
    realized = (prices['e'].pct_change().iloc[-63:]).std() * np.sqrt(252)
    assert np.isclose(scaled['e'].iloc[-1], 0.10 / realized, rtol=1e-6)
    assert scaled['e'].iloc[:31].isna().all()
    capped = volatility_scaling(weights, prices, target_volatility=5.0, max_leverage=2.0)
    assert capped['e'].max() == 2.0
    _no_look_ahead(lambda p: volatility_scaling(weights, p), prices)


def test_combine_signals():
    index = pd.RangeIndex(3)
    momentum = pd.DataFrame([[1, 2, 3], [3, 2, 1], [np.nan] * 3], index=index, columns=list('xyz'))
    value = pd.DataFrame([[3, 2, 1], [3, 2, 1], [np.nan] * 3], index=index, columns=list('xyz'))
    combined = combine_signals({'momentum': momentum, 'value': value})
    assert np.allclose(combined.iloc[0], 0)  # opposite signals cancel out
    assert combined.iloc[1].idxmax() == 'x'
    assert combined.iloc[2].isna().all()
    tilted = combine_signals({'momentum': momentum, 'value': value}, {'momentum': 3, 'value': 1})
    assert tilted.iloc[0].idxmax() == 'z'
    with pytest.raises(MqValueError):
        combine_signals({})
