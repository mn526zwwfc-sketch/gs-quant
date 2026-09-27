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
    black_litterman,
    efficient_frontier,
    equal_weight,
    hierarchical_risk_parity,
    inverse_volatility,
    kelly_fraction,
    kelly_weights,
    maximum_sharpe,
    mean_variance,
    minimum_variance,
    portfolio_volatility,
    risk_contributions,
    risk_parity,
)


@pytest.fixture
def cov():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 8))
    x[:, 4:] += 0.8 * x[:, [0]]
    returns = pd.DataFrame(x * 0.01 * np.linspace(0.5, 2, 8), columns=list('abcdefgh'))
    return returns.cov()


@pytest.fixture
def mu(cov):
    return pd.Series(np.linspace(0.0002, 0.0009, 8), index=cov.columns)


def _sharpe(w, mu, cov):
    return (w @ mu) / np.sqrt(w @ cov @ w)


def test_equal_weight(cov):
    w = equal_weight(cov)
    assert np.allclose(w, 1 / 8) and list(w.index) == list(cov.columns)
    assert equal_weight(['x', 'y']).tolist() == [0.5, 0.5]
    with pytest.raises(MqValueError):
        equal_weight([])


def test_inverse_volatility():
    cov = np.diag([0.01, 0.04])
    assert np.allclose(inverse_volatility(cov), [2 / 3, 1 / 3])


def test_minimum_variance(cov):
    unconstrained = minimum_variance(cov, long_only=False)
    inv = np.linalg.solve(cov, np.ones(8))
    assert np.allclose(unconstrained, inv / inv.sum())
    long_only = minimum_variance(cov)
    assert np.isclose(long_only.sum(), 1) and (long_only >= -1e-10).all()
    assert portfolio_volatility(unconstrained, cov) <= portfolio_volatility(long_only, cov) + 1e-12
    for w in np.random.default_rng(0).dirichlet(np.ones(8), 200):
        assert portfolio_volatility(long_only, cov) <= portfolio_volatility(pd.Series(w, cov.columns), cov) + 1e-12
    capped = minimum_variance(cov, max_weight=0.15)
    assert capped.max() <= 0.15 + 1e-8
    with pytest.raises(MqValueError):
        minimum_variance(cov, max_weight=0.1)


def test_mean_variance_satisfies_kkt(cov, mu):
    risk_aversion = 5.0
    unconstrained = mean_variance(mu, cov, risk_aversion, long_only=False)
    gradient = mu - risk_aversion * cov @ unconstrained
    assert np.isclose(unconstrained.sum(), 1)
    assert np.ptp(gradient) < 1e-12  # equal marginal utility across assets
    long_only = mean_variance(mu, cov, risk_aversion)
    gradient = (mu - risk_aversion * cov @ long_only).to_numpy()
    active = long_only.to_numpy() > 1e-6
    assert np.ptp(gradient[active]) < 1e-9
    assert (gradient[~active] <= gradient[active].mean() + 1e-9).all()
    with pytest.raises(MqValueError):
        mean_variance(mu, cov, risk_aversion=0)


def test_mean_variance_input_validation(cov, mu):
    with pytest.raises(MqValueError):
        mean_variance(mu.iloc[:3], cov)
    with pytest.raises(MqValueError):
        mean_variance(np.ones(3), cov)
    assert np.allclose(mean_variance(mu.to_numpy(), cov), mean_variance(mu, cov))
    assert np.allclose(mean_variance(mu.iloc[::-1], cov), mean_variance(mu, cov))


def test_maximum_sharpe(cov, mu):
    unconstrained = maximum_sharpe(mu, cov, long_only=False)
    x = np.linalg.solve(cov, mu)
    assert np.allclose(unconstrained, x / x.sum())
    long_only = maximum_sharpe(mu, cov)
    assert (long_only >= -1e-10).all() and np.isclose(long_only.sum(), 1)
    best_random = max(_sharpe(w, mu, cov) for w in np.random.default_rng(0).dirichlet(np.ones(8) * 0.5, 5000))
    assert _sharpe(long_only, mu, cov) >= best_random
    assert _sharpe(unconstrained, mu, cov) >= _sharpe(long_only, mu, cov) - 1e-12
    capped = maximum_sharpe(mu, cov, max_weight=0.2)
    assert capped.max() <= 0.2 + 1e-8 and np.isclose(capped.sum(), 1)
    assert _sharpe(capped, mu, cov) <= _sharpe(long_only, mu, cov) + 1e-12


def test_maximum_sharpe_errors(cov, mu):
    with pytest.raises(MqValueError):
        maximum_sharpe(mu, cov, risk_free=0.01)
    with pytest.raises(MqValueError):
        maximum_sharpe(mu, cov, long_only=False, max_weight=0.5)
    with pytest.raises(MqValueError):
        maximum_sharpe(-mu, cov, long_only=False)


def test_risk_parity(cov):
    w = risk_parity(cov)
    assert np.isclose(w.sum(), 1) and (w > 0).all()
    assert np.allclose(risk_contributions(w, cov), 1 / 8)
    budget = np.arange(1, 9, dtype=float)
    budgeted = risk_parity(cov, budget)
    assert np.allclose(risk_contributions(budgeted, cov), budget / budget.sum())
    diagonal = pd.DataFrame(np.diag(np.diag(cov)), index=cov.index, columns=cov.columns)
    assert np.allclose(risk_parity(diagonal), inverse_volatility(diagonal))
    with pytest.raises(MqValueError):
        risk_parity(cov, -budget)
    with pytest.raises(MqValueError):
        risk_parity(cov, max_iterations=1)


def test_hierarchical_risk_parity(cov):
    w = hierarchical_risk_parity(cov)
    assert np.isclose(w.sum(), 1) and (w > 0).all()
    assert list(w.index) == list(cov.columns)
    # uncorrelated assets: HRP reduces to inverse-variance weights
    diagonal = pd.DataFrame(np.diag([0.01, 0.02, 0.04, 0.08]), columns=list('wxyz'), index=list('wxyz'))
    ivp = 1 / np.diag(diagonal)
    assert np.allclose(hierarchical_risk_parity(diagonal), ivp / ivp.sum())
    assert hierarchical_risk_parity(diagonal.iloc[:1, :1]).tolist() == [1.0]


def test_hierarchical_risk_parity_matches_reference_example():
    # two tight clusters of equal volatility: each cluster receives half of the capital
    corr = np.array([[1, 0.9, 0, 0], [0.9, 1, 0, 0], [0, 0, 1, 0.9], [0, 0, 0.9, 1]])
    w = hierarchical_risk_parity(corr * 0.04)
    assert np.allclose(w, 0.25)


def test_kelly():
    assert np.isclose(kelly_fraction(0.6, 1), 0.2)
    assert kelly_fraction(0.4, 1) < 0
    with pytest.raises(MqValueError):
        kelly_fraction(1.2, 1)
    cov = np.array([[0.04]])
    assert np.isclose(kelly_weights([0.08], cov, risk_free=0.02, fraction=1)[0], 1.5)  # (mu - r) / sigma^2
    assert np.isclose(kelly_weights([0.08], cov, risk_free=0.02)[0], 0.75)


def test_black_litterman(cov):
    market = pd.Series(1 / 8, index=cov.columns)
    prior = black_litterman(cov, market, risk_aversion=3)
    assert np.allclose(prior.implied_returns, 3 * cov @ market)
    assert np.allclose(prior.posterior_returns, prior.implied_returns)
    # reverse optimization: the unconstrained mean-variance portfolio of the implied returns is the market
    assert np.allclose(mean_variance(prior.implied_returns, cov, 3, long_only=False), market)

    views = pd.DataFrame([{'a': 1.0, 'b': -1.0}, {'h': 1.0}])
    result = black_litterman(cov, market, 3, views, [0.002, 0.0])
    p, q, tau_sigma = views.reindex(columns=cov.columns).fillna(0).to_numpy(), np.array([0.002, 0.0]), 0.05 * cov
    omega = np.diag(np.diag(p @ tau_sigma @ p.T))
    precision = np.linalg.inv(tau_sigma) + p.T @ np.linalg.inv(omega) @ p
    expected = np.linalg.solve(
        precision, np.linalg.inv(tau_sigma) @ prior.implied_returns + p.T @ np.linalg.inv(omega) @ q
    )
    assert np.allclose(result.posterior_returns, expected)
    assert np.allclose(result.posterior_covariance, cov + np.linalg.inv(precision))
    spread = result.posterior_returns['a'] - result.posterior_returns['b']
    assert prior.implied_returns['a'] - prior.implied_returns['b'] < spread < 0.002

    confident = black_litterman(cov, market, 3, views.iloc[:1], [0.002], view_uncertainty=np.array([[1e-14]]))
    assert np.isclose(confident.posterior_returns['a'] - confident.posterior_returns['b'], 0.002)


def test_black_litterman_validation(cov):
    market = pd.Series(1 / 8, index=cov.columns)
    views = pd.DataFrame([{'a': 1.0}])
    with pytest.raises(MqValueError):
        black_litterman(cov, market, views=views)
    with pytest.raises(MqValueError):
        black_litterman(cov, market, views=views, view_returns=[0.1, 0.2])


def test_efficient_frontier(cov, mu):
    frontier = efficient_frontier(mu, cov, n_points=8)
    assert len(frontier) == 8
    assert np.isclose(frontier['expected_return'].iloc[-1], mu.max())
    assert frontier['volatility'].is_monotonic_increasing
    assert np.allclose(frontier[list(cov.columns)].sum(axis=1), 1)
    min_var = minimum_variance(cov)
    assert np.isclose(frontier['volatility'].iloc[0], portfolio_volatility(min_var, cov))
    capped = efficient_frontier(mu, cov, n_points=5, max_weight=0.3)
    assert capped[list(cov.columns)].max().max() <= 0.3 + 1e-8
    assert np.isclose(capped['expected_return'].iloc[-1], mu.nlargest(4) @ [0.3, 0.3, 0.3, 0.1])
    unconstrained = efficient_frontier(mu, cov, n_points=5, long_only=False)
    assert unconstrained['expected_return'].iloc[-1] > mu.max()
