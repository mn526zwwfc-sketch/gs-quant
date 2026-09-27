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
    VaRMethod,
    covariance_to_correlation,
    denoise_correlation,
    denoised_covariance,
    ewma_covariance,
    expected_shortfall,
    ledoit_wolf_covariance,
    ledoit_wolf_shrinkage,
    portfolio_volatility,
    risk_contributions,
    sample_covariance,
    value_at_risk,
)


@pytest.fixture
def returns():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 0.01, (120, 30)) * np.linspace(0.5, 2, 30)
    return pd.DataFrame(x, columns=[f'a{i}' for i in range(30)])


def _reference_ledoit_wolf(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Direct transcription of Ledoit & Wolf (2004), Section 3.3"""
    n, p = x.shape
    x = x - x.mean(axis=0)
    s = x.T @ x / n
    m = np.trace(s) / p
    d2 = np.linalg.norm(s - m * np.eye(p), 'fro') ** 2
    b_bar2 = sum(np.linalg.norm(np.outer(row, row) - s, 'fro') ** 2 for row in x) / n**2
    b2 = min(b_bar2, d2)
    return b2 / d2 * m * np.eye(p) + (d2 - b2) / d2 * s, b2 / d2


def test_ledoit_wolf_matches_paper(returns):
    expected, shrinkage = _reference_ledoit_wolf(returns.to_numpy())
    assert np.isclose(ledoit_wolf_shrinkage(returns), shrinkage)
    result = ledoit_wolf_covariance(returns)
    assert np.allclose(result.to_numpy(), expected)
    assert list(result.columns) == list(returns.columns)


def test_ledoit_wolf_is_well_conditioned(returns):
    short = returns.iloc[:20]  # fewer observations than assets: sample covariance is singular
    assert np.linalg.matrix_rank(sample_covariance(short).to_numpy()) < short.shape[1]
    assert np.linalg.eigvalsh(ledoit_wolf_covariance(short).to_numpy()).min() > 0


def test_ledoit_wolf_identity_target():
    x = pd.DataFrame(np.eye(4) - 0.25)
    cov, shrinkage = _reference_ledoit_wolf(x.to_numpy())
    assert 0 <= ledoit_wolf_shrinkage(x) <= 1
    assert np.allclose(ledoit_wolf_covariance(x).to_numpy(), cov)


def test_sample_covariance_requires_data():
    with pytest.raises(MqValueError):
        sample_covariance(pd.DataFrame({'a': [0.1, np.nan], 'b': [0.2, 0.3]}))


def test_ewma_covariance(returns):
    decay = 0.9
    cov = ewma_covariance(returns.iloc[:, :3], decay)
    x = returns.iloc[:, :3].to_numpy()
    expected = np.zeros((3, 3))
    for row in x:  # RiskMetrics recursion started from zero, then normalized by the total weight
        expected = decay * expected + (1 - decay) * np.outer(row, row)
    expected /= 1 - decay ** len(x)
    assert np.allclose(cov.to_numpy(), expected)
    with pytest.raises(MqValueError):
        ewma_covariance(returns, 1.0)


def test_correlation_conversion(returns):
    cov = sample_covariance(returns)
    assert np.allclose(covariance_to_correlation(cov).to_numpy(), returns.corr().to_numpy())
    assert isinstance(covariance_to_correlation(cov.to_numpy()), np.ndarray)


def test_denoise_correlation():
    rng = np.random.default_rng(3)
    n_obs, n_assets = 250, 50
    factor = rng.normal(size=(n_obs, 1))
    x = 0.6 * factor + rng.normal(size=(n_obs, n_assets))
    corr = np.corrcoef(x, rowvar=False)
    denoised = denoise_correlation(corr, n_obs)
    true_corr = np.full((n_assets, n_assets), 0.36 / 1.36)
    np.fill_diagonal(true_corr, 1)
    assert np.allclose(np.diag(denoised), 1)
    assert np.allclose(denoised, denoised.T)
    assert np.linalg.norm(denoised - true_corr) < np.linalg.norm(corr - true_corr)
    assert np.linalg.eigvalsh(denoised).min() > 0
    frame = denoise_correlation(pd.DataFrame(corr), n_obs)
    assert isinstance(frame, pd.DataFrame)


def test_denoise_correlation_all_noise_or_signal():
    identity = np.eye(3)
    assert denoise_correlation(identity, 1000) is identity
    strong = np.full((3, 3), 0.99)
    np.fill_diagonal(strong, 1)
    assert np.allclose(denoise_correlation(strong, 10**9), strong)


def test_denoised_covariance_keeps_variances(returns):
    cov = denoised_covariance(returns)
    assert np.allclose(np.diag(cov), returns.var().to_numpy())


def test_risk_contributions_and_volatility():
    cov = pd.DataFrame([[0.04, 0.006], [0.006, 0.09]], index=['x', 'y'], columns=['x', 'y'])
    w = pd.Series({'y': 0.4, 'x': 0.6})
    vol = portfolio_volatility(w, cov)
    assert np.isclose(vol, np.sqrt(0.36 * 0.04 + 0.16 * 0.09 + 2 * 0.24 * 0.006))
    rc = risk_contributions(w, cov)
    assert np.isclose(rc.sum(), 1)
    assert np.isclose(rc['x'], 0.6 * (0.6 * 0.04 + 0.4 * 0.006) / vol**2)
    assert list(risk_contributions(np.array([0.5, 0.5]), cov.to_numpy()).index) == [0, 1]
    assert list(risk_contributions(w, cov.to_numpy()).index) == ['y', 'x']


def test_value_at_risk_methods():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.02, 200_000))
    gaussian = -(0.001 + stats.norm.ppf(0.01) * 0.02)
    assert np.isclose(value_at_risk(r, 0.99, VaRMethod.HISTORICAL), gaussian, rtol=0.02)
    assert np.isclose(value_at_risk(r, 0.99, VaRMethod.GAUSSIAN), gaussian, rtol=0.01)
    assert np.isclose(value_at_risk(r, 0.99, 'cornish_fisher'), gaussian, rtol=0.01)
    frame = value_at_risk(pd.DataFrame({'a': r, 'b': 2 * r}), 0.99)
    assert np.isclose(frame['b'], 2 * frame['a'])


def test_cornish_fisher_captures_skewness():
    rng = np.random.default_rng(1)
    # negatively skewed returns with a long left tail
    r = pd.Series(-(rng.lognormal(0, 0.35, 100_000) - np.exp(0.35**2 / 2)) * 0.02)
    historical = value_at_risk(r, 0.99)
    cornish_fisher = value_at_risk(r, 0.99, 'cornish_fisher')
    gaussian = value_at_risk(r, 0.99, 'gaussian')
    assert gaussian < cornish_fisher
    assert abs(cornish_fisher - historical) < abs(gaussian - historical) / 3


def test_expected_shortfall():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 500_000))
    expected = 0.01 * stats.norm.pdf(stats.norm.ppf(0.025)) / 0.025
    assert np.isclose(expected_shortfall(r, 0.975), expected, rtol=0.01)
    assert np.isclose(expected_shortfall(r, 0.975, 'gaussian'), expected, rtol=0.01)
    assert expected_shortfall(r, 0.975) > value_at_risk(r, 0.975)
    assert isinstance(expected_shortfall(pd.DataFrame({'a': r}), 0.975), pd.Series)
    with pytest.raises(MqValueError):
        expected_shortfall(r, 0.975, VaRMethod.CORNISH_FISHER)


def test_tail_input_validation():
    with pytest.raises(MqValueError):
        value_at_risk(pd.Series([0.01]), 0.95)
    with pytest.raises(MqValueError):
        value_at_risk(pd.Series([0.01, 0.02]), 1.5)
