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
import statsmodels.api as sm

from gs_quant.errors import MqValueError
from gs_quant.research import factor_regression, fama_macbeth, newey_west_lags


@pytest.fixture
def factor_data():
    rng = np.random.default_rng(11)
    index = pd.date_range('2000-01-31', periods=240, freq='BME')
    factors = pd.DataFrame(rng.normal(0.005, 0.04, (240, 3)), index=index, columns=['Mkt-RF', 'SMB', 'HML'])
    rf = pd.Series(0.002, index=index)
    excess = 0.003 + factors @ np.array([1.1, 0.4, -0.3]) + rng.normal(0, 0.01, 240)
    return excess + rf, factors, rf


def test_newey_west_lags():
    assert newey_west_lags(100) == 4
    assert newey_west_lags(1000) == 6


def test_factor_regression_recovers_loadings(factor_data):
    returns, factors, rf = factor_data
    result = factor_regression(returns, factors, risk_free=rf)
    assert np.allclose(result.betas.to_numpy(), [1.1, 0.4, -0.3], atol=0.05)
    assert np.isclose(result.alpha, 0.003, atol=0.0015)
    assert np.isclose(result.alpha_annualized, (1 + result.alpha) ** 12 - 1)
    assert result.t_stats['alpha'] > 2 and result.p_values['alpha'] < 0.05
    assert result.r_squared > 0.9 and result.adj_r_squared <= result.r_squared
    assert np.isclose(result.residual_volatility, 0.01 * np.sqrt(12), rtol=0.15)
    assert result.n_observations == 240
    flat = result.to_series()
    assert {'alpha', 'beta_Mkt-RF', 't_HML', 'r_squared'} <= set(flat.index)


def test_factor_regression_matches_statsmodels_hac(factor_data):
    returns, factors, _ = factor_data
    result = factor_regression(returns, factors, hac_lags=3, periods_per_year=12)
    reference = sm.OLS(returns, sm.add_constant(factors)).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
    assert np.allclose(result.t_stats.to_numpy(), reference.tvalues.to_numpy())


def test_factor_regression_requires_overlap(factor_data):
    returns, factors, _ = factor_data
    with pytest.raises(MqValueError):
        factor_regression(returns.iloc[:3], factors)


def test_fama_macbeth_recovers_premia():
    rng = np.random.default_rng(5)
    dates = pd.date_range('2005-01-31', periods=180, freq='BME')
    assets = [f's{i}' for i in range(200)]
    size = pd.DataFrame(rng.normal(size=(180, 200)), index=dates, columns=assets)
    value = pd.DataFrame(rng.normal(size=(180, 200)), index=dates, columns=assets)
    premia = {'size': -0.002, 'value': 0.004}
    returns = 0.01 + premia['size'] * size.shift(1) + premia['value'] * value.shift(1)
    returns = returns + rng.normal(0, 0.03, returns.shape)
    result = fama_macbeth(returns, {'size': size, 'value': value})
    assert len(result.gammas) == 179  # the first date has no lagged characteristics
    assert np.isclose(result.premia['size'], premia['size'], atol=0.0008)
    assert np.isclose(result.premia['value'], premia['value'], atol=0.0008)
    assert result.t_stats['value'] > 3
    table = result.summary()
    assert list(table.columns) == ['premium', 'std_error', 't_stat', 'p_value']
    assert table.loc['value', 'p_value'] < 0.01
    classic = result.gammas['value'].std(ddof=0) / np.sqrt(len(result.gammas))
    assert np.isclose(result.standard_errors['value'], classic)
    hac = fama_macbeth(returns, {'size': size, 'value': value}, hac_lags=3)
    assert np.allclose(hac.premia, result.premia)
    assert not np.allclose(hac.standard_errors, result.standard_errors)


def test_fama_macbeth_validation():
    returns = pd.DataFrame(np.zeros((5, 2)))
    with pytest.raises(MqValueError):
        fama_macbeth(returns, {})
    with pytest.raises(MqValueError):
        fama_macbeth(returns, {'x': returns})  # two assets are too few for a cross-sectional regression
