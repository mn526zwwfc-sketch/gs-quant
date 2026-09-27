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

Portfolio construction.

Allocation rules from the academic literature, ordered roughly by how much they rely on estimated inputs:

* Equal weight, the 1/N rule that DeMiguel, Garlappi & Uppal (2009) show is hard to beat out of sample.
* Inverse volatility and equal risk contribution / risk parity (Maillard, Roncalli & Teiletche, 2010), solved
  with the cyclical coordinate descent of Griveau-Billion, Richard & Roncalli (2013).
* Hierarchical risk parity (López de Prado, 2016), which avoids inverting the covariance matrix.
* Minimum variance (Clarke, de Silva & Thorley, 2006), mean-variance (Markowitz, 1952) and maximum Sharpe
  ratio / tangency portfolios (Tobin, 1958; Sharpe, 1964).
* Black & Litterman (1992) posterior expected returns, following He & Litterman (1999).
* Kelly (1956) growth-optimal sizing, usually applied fractionally (MacLean, Thorp & Ziemba, 2011).

Covariance matrices should preferably come from a robust estimator in :mod:`gs_quant.research.risk`.
All weights are returned as a ``pd.Series`` indexed by asset.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

from gs_quant.errors import MqValueError
from gs_quant.research.risk import covariance_to_correlation

CovLike = Union[pd.DataFrame, np.ndarray]
VectorLike = Union[pd.Series, np.ndarray, Sequence[float]]


@dataclass
class BlackLittermanResult:
    """
    Output of :func:`black_litterman`
    """

    implied_returns: pd.Series
    posterior_returns: pd.Series
    posterior_covariance: pd.DataFrame


def _cov(cov: CovLike) -> tuple[np.ndarray, pd.Index]:
    if isinstance(cov, pd.DataFrame):
        matrix, labels = cov.to_numpy(dtype=float), cov.columns
    else:
        matrix = np.asarray(cov, dtype=float)
        labels = pd.RangeIndex(matrix.shape[0])
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise MqValueError('covariance matrix must be square')
    return matrix, labels


def _vector(values: VectorLike, labels: pd.Index) -> np.ndarray:
    if isinstance(values, pd.Series):
        if values.index.isin(labels).all() and labels.isin(values.index).all():
            return values.reindex(labels).to_numpy(dtype=float)
        raise MqValueError('expected returns and covariance matrix must cover the same assets')
    values = np.asarray(values, dtype=float)
    if values.shape != (len(labels),):
        raise MqValueError('expected returns and covariance matrix must have the same length')
    return values


def _check_bounds(n: int, long_only: bool, max_weight: Optional[float]):
    if max_weight is not None and (max_weight <= 0 or (long_only and max_weight * n < 1 - 1e-12)):
        raise MqValueError(f'max_weight={max_weight} makes a fully invested portfolio of {n} assets infeasible')


def _bounds(n: int, long_only: bool, max_weight: Optional[float]) -> list[tuple[Optional[float], Optional[float]]]:
    lower = 0.0 if long_only else (-max_weight if max_weight is not None else None)
    return [(lower, max_weight)] * n


def _solve(
    objective, gradient, scale: float, n: int, long_only: bool, max_weight: Optional[float], constraints=()
) -> np.ndarray:
    """Minimize ``objective / scale`` over fully invested portfolios; scaling keeps SLSQP tolerances meaningful"""
    _check_bounds(n, long_only, max_weight)
    budget = {'type': 'eq', 'fun': lambda w: w.sum() - 1, 'jac': lambda w: np.ones(n)}
    result = optimize.minimize(
        lambda w: objective(w) / scale,
        np.full(n, 1 / n),
        jac=lambda w: gradient(w) / scale,
        method='SLSQP',
        bounds=_bounds(n, long_only, max_weight),
        constraints=[budget, *constraints],
        options={'maxiter': 1000, 'ftol': 1e-15},
    )
    if not result.success:
        raise MqValueError(f'portfolio optimization failed: {result.message}')
    return result.x


def _scale(matrix: np.ndarray) -> float:
    return float(np.mean(np.diag(matrix)))


def _unconstrained(long_only: bool, max_weight: Optional[float]) -> bool:
    return not long_only and max_weight is None


def equal_weight(assets: Union[Sequence, pd.Index, pd.DataFrame]) -> pd.Series:
    """
    Naive 1/N diversification (DeMiguel, Garlappi & Uppal, 2009)

    :param assets: asset identifiers, or a covariance matrix / returns frame whose columns are the assets
    :return: equal weights summing to one
    """
    labels = assets.columns if isinstance(assets, pd.DataFrame) else pd.Index(assets)
    if len(labels) == 0:
        raise MqValueError('at least one asset is required')
    return pd.Series(1 / len(labels), index=labels)


def inverse_volatility(cov: CovLike) -> pd.Series:
    """
    Weights inversely proportional to volatility

    :param cov: covariance matrix
    :return: :math:`w_i \\propto 1 / \\sigma_i`, summing to one. Identical to risk parity when correlations are equal
    """
    matrix, labels = _cov(cov)
    inv = 1 / np.sqrt(np.diag(matrix))
    return pd.Series(inv / inv.sum(), index=labels)


def minimum_variance(cov: CovLike, long_only: bool = True, max_weight: Optional[float] = None) -> pd.Series:
    """
    Global minimum variance portfolio

    :param cov: covariance matrix
    :param long_only: forbid short positions
    :param max_weight: optional cap on the absolute weight of each asset
    :return: fully invested weights minimising :math:`w^\\top \\Sigma w`

    **Usage**

    Needs no expected returns, which are the noisiest inputs of mean-variance optimization; minimum variance
    portfolios have historically delivered market-like returns with lower risk (Clarke, de Silva & Thorley, 2006).
    Imposing long-only constraints acts as a form of covariance shrinkage (Jagannathan & Ma, 2003).
    """
    matrix, labels = _cov(cov)
    n = len(labels)
    if _unconstrained(long_only, max_weight):
        x = np.linalg.solve(matrix, np.ones(n))
        return pd.Series(x / x.sum(), index=labels)
    w = _solve(lambda x: 0.5 * x @ matrix @ x, lambda x: matrix @ x, _scale(matrix), n, long_only, max_weight)
    return pd.Series(w, index=labels)


def mean_variance(
    expected_returns: VectorLike,
    cov: CovLike,
    risk_aversion: float = 2.5,
    long_only: bool = True,
    max_weight: Optional[float] = None,
) -> pd.Series:
    """
    Markowitz (1952) mean-variance optimal portfolio

    :param expected_returns: expected returns, in the same units as the covariance matrix
    :param cov: covariance matrix
    :param risk_aversion: coefficient :math:`\\lambda` of the quadratic utility
    :param long_only: forbid short positions
    :param max_weight: optional cap on the absolute weight of each asset
    :return: fully invested weights maximising :math:`w^\\top \\mu - \\frac{\\lambda}{2} w^\\top \\Sigma w`

    **Usage**

    Mean-variance weights are very sensitive to errors in expected returns (Michaud, 1989; Chopra & Ziemba, 1993).
    Consider shrunk covariances, :func:`black_litterman` expected returns and weight constraints.
    """
    matrix, labels = _cov(cov)
    mu = _vector(expected_returns, labels)
    n = len(labels)
    if risk_aversion <= 0:
        raise MqValueError('risk_aversion must be positive')
    if _unconstrained(long_only, max_weight):
        inv_mu, inv_one = np.linalg.solve(matrix, mu), np.linalg.solve(matrix, np.ones(n))
        eta = (inv_mu.sum() - risk_aversion) / inv_one.sum()
        return pd.Series((inv_mu - eta * inv_one) / risk_aversion, index=labels)
    w = _solve(
        lambda x: -(x @ mu) + 0.5 * risk_aversion * x @ matrix @ x,
        lambda x: -mu + risk_aversion * matrix @ x,
        _scale(matrix),
        n,
        long_only,
        max_weight,
    )
    return pd.Series(w, index=labels)


def maximum_sharpe(
    expected_returns: VectorLike,
    cov: CovLike,
    risk_free: float = 0.0,
    long_only: bool = True,
    max_weight: Optional[float] = None,
) -> pd.Series:
    """
    Tangency portfolio with the highest Sharpe ratio (Tobin, 1958; Sharpe, 1964)

    :param expected_returns: expected returns, in the same units as the covariance matrix
    :param cov: covariance matrix
    :param risk_free: risk-free rate, in the same units as the expected returns
    :param long_only: forbid short positions
    :param max_weight: optional cap on the absolute weight of each asset (long-only portfolios)
    :return: fully invested weights maximising :math:`(w^\\top \\mu - r_f) / \\sqrt{w^\\top \\Sigma w}`

    **Usage**

    The unconstrained solution is :math:`w \\propto \\Sigma^{-1}(\\mu - r_f)`. With constraints the problem is
    solved as the convex program :math:`\\min y^\\top \\Sigma y` s.t. :math:`(\\mu - r_f)^\\top y = 1, y \\geq 0`,
    then :math:`w = y / \\sum y` (Cornuejols & Tütüncü, 2007).
    """
    matrix, labels = _cov(cov)
    excess = _vector(expected_returns, labels) - risk_free
    n = len(labels)
    if _unconstrained(long_only, max_weight):
        x = np.linalg.solve(matrix, excess)
        if x.sum() <= 0:
            raise MqValueError('no portfolio has a positive excess return with positive net exposure')
        return pd.Series(x / x.sum(), index=labels)
    if not long_only:
        raise MqValueError('max_weight is only supported for long-only maximum Sharpe portfolios')
    if (excess <= 0).all():
        raise MqValueError('at least one asset must have an expected return above the risk-free rate')
    _check_bounds(n, long_only, max_weight)
    # the solution is invariant to rescaling, so normalize the inputs to keep the problem well scaled
    matrix, excess = matrix / _scale(matrix), excess / np.abs(excess).max()
    constraints = [{'type': 'eq', 'fun': lambda y: excess @ y - 1, 'jac': lambda y: excess}]
    if max_weight is not None:
        # w_i <= m  <=>  m * sum(y) - y_i >= 0 (homogeneous in y)
        cap = max_weight * np.ones((n, n)) - np.eye(n)
        constraints.append({'type': 'ineq', 'fun': lambda y: cap @ y, 'jac': lambda y: cap})
    start = np.where(excess > 0, 1.0, 0.0)
    start /= excess @ start
    result = optimize.minimize(
        lambda y: y @ matrix @ y,
        start,
        jac=lambda y: 2 * matrix @ y,
        method='SLSQP',
        bounds=[(0, None)] * n,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-15},
    )
    if not result.success:
        raise MqValueError(f'portfolio optimization failed: {result.message}')
    return pd.Series(result.x / result.x.sum(), index=labels)


def risk_parity(
    cov: CovLike, risk_budget: Optional[VectorLike] = None, tolerance: float = 1e-10, max_iterations: int = 10_000
) -> pd.Series:
    """
    Equal risk contribution (risk budgeting) portfolio

    :param cov: covariance matrix
    :param risk_budget: target share of risk of each asset; equal budgets when omitted
    :param tolerance: convergence tolerance on the weights
    :param max_iterations: maximum number of coordinate descent sweeps
    :return: long-only weights summing to one whose risk contributions match the budget

    **Usage**

    Solves :math:`\\min_y \\frac{1}{2} y^\\top \\Sigma y - \\sum_i b_i \\ln y_i`, whose solution satisfies
    :math:`y_i (\\Sigma y)_i = b_i` (Spinu, 2013), by cyclical coordinate descent (Griveau-Billion, Richard &
    Roncalli, 2013), then normalizes :math:`w = y / \\sum y`. Each asset contributes the same share of portfolio
    variance (Maillard, Roncalli & Teiletche, 2010; Qian, 2005).
    """
    matrix, labels = _cov(cov)
    n = len(labels)
    budget = np.full(n, 1 / n) if risk_budget is None else _vector(risk_budget, labels)
    if (budget <= 0).any():
        raise MqValueError('risk budgets must be positive')
    budget = budget / budget.sum()
    diag = np.diag(matrix)
    y = 1 / np.sqrt(diag)
    for _ in range(max_iterations):
        previous = y.copy()
        for i in range(n):
            c = matrix[i] @ y - diag[i] * y[i]
            y[i] = (-c + np.sqrt(c * c + 4 * diag[i] * budget[i])) / (2 * diag[i])
        if np.max(np.abs(y - previous)) <= tolerance * np.max(np.abs(y)):
            break
    else:
        raise MqValueError('risk parity did not converge')
    return pd.Series(y / y.sum(), index=labels)


def _cluster_variance(matrix: np.ndarray, members: np.ndarray) -> float:
    sub = matrix[np.ix_(members, members)]
    ivp = 1 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def hierarchical_risk_parity(cov: CovLike, linkage_method: str = 'single') -> pd.Series:
    """
    Hierarchical Risk Parity (López de Prado, 2016)

    :param cov: covariance matrix
    :param linkage_method: agglomeration method passed to ``scipy.cluster.hierarchy.linkage``
    :return: long-only weights summing to one

    **Usage**

    1. Tree clustering of the assets on the correlation distance :math:`d_{ij} = \\sqrt{(1 - \\rho_{ij}) / 2}`.
    2. Quasi-diagonalization: reorder assets so that similar ones are adjacent.
    3. Recursive bisection: split the ordered list in halves and allocate between them inversely to their
       inverse-variance cluster variances.

    HRP never inverts the covariance matrix, so it is robust when the matrix is ill-conditioned, and it has shown
    lower out-of-sample variance than minimum variance in Monte Carlo studies (López de Prado, 2016, "Building
    Diversified Portfolios that Outperform Out of Sample", Journal of Portfolio Management 42(4)).
    """
    matrix, labels = _cov(cov)
    n = len(labels)
    if n == 1:
        return pd.Series([1.0], index=labels)
    distance = np.sqrt(np.clip(0.5 * (1 - covariance_to_correlation(matrix)), 0, None))
    order = leaves_list(linkage(pdist(distance), method=linkage_method))
    weights = np.ones(n)
    clusters = [order]
    while clusters:
        clusters = [half for c in clusters if len(c) > 1 for half in (c[: len(c) // 2], c[len(c) // 2 :])]
        for left, right in zip(clusters[::2], clusters[1::2]):
            var_left, var_right = _cluster_variance(matrix, left), _cluster_variance(matrix, right)
            alpha = 1 - var_left / (var_left + var_right)
            weights[left] *= alpha
            weights[right] *= 1 - alpha
    return pd.Series(weights, index=labels)


def kelly_weights(
    expected_returns: VectorLike, cov: CovLike, risk_free: float = 0.0, fraction: float = 0.5
) -> pd.Series:
    """
    Growth-optimal (Kelly) portfolio for continuous returns

    :param expected_returns: expected returns, in the same units as the covariance matrix
    :param cov: covariance matrix
    :param risk_free: risk-free rate, in the same units as the expected returns
    :param fraction: fraction of the full Kelly allocation; half Kelly keeps most of the growth with far less risk
    :return: weights :math:`f \\Sigma^{-1}(\\mu - r_f)`; they do not sum to one and the remainder is held in cash

    **Usage**

    Full Kelly maximises the long-run growth rate of wealth (Kelly, 1956; Thorp, 2006) but is extremely
    sensitive to estimation error, hence the widespread use of fractional Kelly (MacLean, Thorp & Ziemba, 2011).
    """
    matrix, labels = _cov(cov)
    excess = _vector(expected_returns, labels) - risk_free
    return pd.Series(fraction * np.linalg.solve(matrix, excess), index=labels)


def kelly_fraction(win_probability: float, win_loss_ratio: float) -> float:
    """
    Kelly fraction for a repeated binary bet (Kelly, 1956)

    :param win_probability: probability :math:`p` of winning
    :param win_loss_ratio: amount won per unit lost, :math:`b`
    :return: optimal fraction of capital to stake, :math:`p - (1 - p) / b`; negative means do not bet
    """
    if not 0 <= win_probability <= 1 or win_loss_ratio <= 0:
        raise MqValueError('win_probability must be in [0, 1] and win_loss_ratio positive')
    return win_probability - (1 - win_probability) / win_loss_ratio


def black_litterman(
    cov: pd.DataFrame,
    market_weights: pd.Series,
    risk_aversion: float = 2.5,
    views: Optional[pd.DataFrame] = None,
    view_returns: Optional[VectorLike] = None,
    view_uncertainty: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    tau: float = 0.05,
) -> BlackLittermanResult:
    """
    Black & Litterman (1992) combination of equilibrium returns and investor views

    :param cov: covariance matrix of asset returns
    :param market_weights: market capitalization weights of the assets
    :param risk_aversion: market risk aversion :math:`\\delta`
    :param views: pick matrix :math:`P` (one row per view, one column per asset); absolute view on asset A:
        a row with 1 on A; relative view "A beats B": 1 on A and -1 on B
    :param view_returns: expected return :math:`Q` of each view
    :param view_uncertainty: covariance :math:`\\Omega` of the views; defaults to
        :math:`\\mathrm{diag}(P \\tau \\Sigma P^\\top)` (He & Litterman, 1999)
    :param tau: scaling of the uncertainty of the equilibrium returns
    :return: :class:`BlackLittermanResult`

    **Usage**

    Implied equilibrium returns are :math:`\\Pi = \\delta \\Sigma w_{mkt}` (reverse optimization). The posterior
    is :math:`\\mu = \\Pi + \\tau \\Sigma P^\\top (P \\tau \\Sigma P^\\top + \\Omega)^{-1} (Q - P \\Pi)`. Feeding
    it to :func:`mean_variance` yields intuitive, well-diversified portfolios that tilt towards the views.
    """
    matrix, labels = _cov(cov)
    w_mkt = _vector(market_weights.reindex(labels).fillna(0), labels)
    implied = risk_aversion * matrix @ w_mkt
    posterior, posterior_cov = implied, matrix.copy()
    if views is not None:
        if view_returns is None:
            raise MqValueError('view_returns are required when views are given')
        p = views.reindex(columns=labels).fillna(0).to_numpy(dtype=float)
        q = np.asarray(view_returns, dtype=float)
        if q.shape != (p.shape[0],):
            raise MqValueError('view_returns must have one entry per view')
        tau_sigma = tau * matrix
        omega = (
            np.diag(np.diag(p @ tau_sigma @ p.T)) if view_uncertainty is None else np.asarray(view_uncertainty, float)
        )
        gain = np.linalg.solve(p @ tau_sigma @ p.T + omega, p @ tau_sigma).T
        posterior = implied + gain @ (q - p @ implied)
        posterior_cov = matrix + tau_sigma - gain @ p @ tau_sigma
    return BlackLittermanResult(
        implied_returns=pd.Series(implied, index=labels),
        posterior_returns=pd.Series(posterior, index=labels),
        posterior_covariance=pd.DataFrame(posterior_cov, index=labels, columns=labels),
    )


def efficient_frontier(
    expected_returns: VectorLike,
    cov: CovLike,
    n_points: int = 25,
    long_only: bool = True,
    max_weight: Optional[float] = None,
) -> pd.DataFrame:
    """
    Minimum-variance frontier (Markowitz, 1952)

    :param expected_returns: expected returns, in the same units as the covariance matrix
    :param cov: covariance matrix
    :param n_points: number of portfolios on the frontier
    :param long_only: forbid short positions
    :param max_weight: optional cap on the absolute weight of each asset
    :return: one row per portfolio with its expected return, volatility and weights
    """
    matrix, labels = _cov(cov)
    mu = _vector(expected_returns, labels)
    n = len(labels)
    w_min = minimum_variance(pd.DataFrame(matrix, index=labels, columns=labels), long_only, max_weight).to_numpy()
    if _unconstrained(long_only, max_weight):
        top = max(mu.max(), w_min @ mu) * 2 - w_min @ mu
    else:
        best = optimize.linprog(-mu, A_eq=np.ones((1, n)), b_eq=[1], bounds=_bounds(n, long_only, max_weight))
        top = -best.fun
    rows = []
    for target in np.linspace(w_min @ mu, top, n_points):
        on_target = {'type': 'eq', 'fun': lambda x, t=target: x @ mu - t, 'jac': lambda x: mu}
        w = _solve(
            lambda x: 0.5 * x @ matrix @ x, lambda x: matrix @ x, _scale(matrix), n, long_only, max_weight, [on_target]
        )
        rows.append([w @ mu, np.sqrt(w @ matrix @ w), *w])
    return pd.DataFrame(rows, columns=['expected_return', 'volatility', *labels])
