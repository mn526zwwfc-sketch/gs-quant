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

Risk estimation.

The sample covariance matrix is a poor input for portfolio construction when the number of assets is not small
relative to the number of observations: its extreme eigenvalues are biased and optimizers amplify that estimation
error (Michaud, 1989). This module provides the estimators recommended in the literature:

* Exponentially weighted covariance (J.P. Morgan RiskMetrics, 1996).
* Ledoit & Wolf (2004) shrinkage towards a scaled identity matrix.
* Random matrix theory denoising based on the Marchenko-Pastur distribution (Laloux, Cizeau, Bouchaud & Potters,
  1999; López de Prado, 2020).

and the standard downside risk measures: Value-at-Risk (Jorion, 2006), its Cornish-Fisher adjustment for skewness
and kurtosis (Zangari, 1996; Favre & Galeano, 2002) and Expected Shortfall, a coherent risk measure
(Artzner, Delbaen, Eber & Heath, 1999; Acerbi & Tasche, 2002).
"""

from enum import Enum
from typing import Union

import numpy as np
import pandas as pd
from scipy import stats

from gs_quant.errors import MqValueError

ArrayLike = Union[pd.DataFrame, np.ndarray]


class VaRMethod(Enum):
    HISTORICAL = 'historical'
    GAUSSIAN = 'gaussian'
    CORNISH_FISHER = 'cornish_fisher'


def _clean_returns(returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame):
        returns = pd.DataFrame(returns)
    clean = returns.dropna(how='any')
    if len(clean) < 2:
        raise MqValueError('at least two complete observations are required to estimate a covariance matrix')
    return clean


def _wrap(matrix: np.ndarray, like: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=like.columns, columns=like.columns)


def _as_matrix(cov: ArrayLike) -> tuple[np.ndarray, pd.Index]:
    if isinstance(cov, pd.DataFrame):
        return cov.to_numpy(dtype=float), cov.columns
    cov = np.asarray(cov, dtype=float)
    return cov, pd.RangeIndex(cov.shape[0])


def sample_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Unbiased sample covariance matrix of complete observations

    :param returns: periodic returns (dates x assets)
    :return: covariance matrix per period
    """
    return _clean_returns(returns).cov()


def ewma_covariance(returns: pd.DataFrame, decay: float = 0.94) -> pd.DataFrame:
    """
    Exponentially weighted covariance matrix (RiskMetrics, 1996)

    :param returns: periodic returns (dates x assets)
    :param decay: decay factor :math:`\\lambda`; RiskMetrics uses 0.94 for daily and 0.97 for monthly data
    :return: covariance matrix per period, as of the last date

    **Usage**

    :math:`\\Sigma_T = \\sum_t w_t r_t r_t^\\top` with :math:`w_t \\propto \\lambda^{T-t}` and a zero mean, as
    in RiskMetrics. Recent observations weigh more, so the estimate adapts quickly to volatility regimes.
    """
    if not 0 < decay < 1:
        raise MqValueError('decay must be in (0, 1)')
    clean = _clean_returns(returns)
    x = clean.to_numpy(dtype=float)
    weights = decay ** np.arange(len(x) - 1, -1, -1)
    weights /= weights.sum()
    return _wrap((x * weights[:, None]).T @ x, clean)


def ledoit_wolf_shrinkage(returns: pd.DataFrame) -> float:
    """
    Optimal shrinkage intensity of Ledoit & Wolf (2004) towards a scaled identity matrix

    :param returns: periodic returns (dates x assets)
    :return: shrinkage intensity in [0, 1]
    """
    x = _clean_returns(returns).to_numpy(dtype=float)
    return _ledoit_wolf(x)[1]


def _ledoit_wolf(x: np.ndarray) -> tuple[np.ndarray, float]:
    n, p = x.shape
    x = x - x.mean(axis=0)
    sample = x.T @ x / n
    mu = np.trace(sample) / p
    target = mu * np.eye(p)
    d2 = np.sum((sample - target) ** 2)
    if d2 == 0:
        return sample, 0.0
    # b_bar^2 = 1/n^2 sum_k ||x_k x_k' - S||_F^2 = (sum_k ||x_k||^4 - n ||S||_F^2) / n^2
    b_bar2 = (np.sum(np.sum(x**2, axis=1) ** 2) - n * np.sum(sample**2)) / n**2
    shrinkage = float(min(max(b_bar2, 0.0), d2) / d2)
    return shrinkage * target + (1 - shrinkage) * sample, shrinkage


def ledoit_wolf_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Ledoit & Wolf (2004) shrinkage covariance matrix

    :param returns: periodic returns (dates x assets)
    :return: covariance matrix per period

    **Usage**

    :math:`\\hat\\Sigma = \\delta^* \\mu I + (1 - \\delta^*) S`, where :math:`S` is the sample covariance,
    :math:`\\mu` its average variance and :math:`\\delta^*` the shrinkage intensity minimising the expected
    Frobenius loss. The estimator is always well-conditioned, even when there are more assets than observations.
    Reference: Ledoit, O. & Wolf, M. (2004), "A well-conditioned estimator for large-dimensional covariance
    matrices", Journal of Multivariate Analysis 88(2).
    """
    clean = _clean_returns(returns)
    return _wrap(_ledoit_wolf(clean.to_numpy(dtype=float))[0], clean)


def covariance_to_correlation(cov: ArrayLike) -> ArrayLike:
    """
    Convert a covariance matrix to a correlation matrix

    :param cov: covariance matrix
    :return: correlation matrix of the same type
    """
    matrix, labels = _as_matrix(cov)
    vols = np.sqrt(np.diag(matrix))
    corr = np.clip(matrix / np.outer(vols, vols), -1, 1)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=labels, columns=labels) if isinstance(cov, pd.DataFrame) else corr


def denoise_correlation(corr: ArrayLike, n_observations: int) -> ArrayLike:
    """
    Remove noise from a correlation matrix with random matrix theory

    :param corr: empirical correlation matrix
    :param n_observations: number of observations used to estimate it
    :return: denoised correlation matrix of the same type

    **Usage**

    Under the null of pure noise, the eigenvalues of a correlation matrix estimated from :math:`T` observations of
    :math:`N` assets are bounded by :math:`\\lambda_+ = (1 + \\sqrt{N/T})^2` (Marchenko & Pastur, 1967).
    Eigenvalues below :math:`\\lambda_+` are indistinguishable from noise and are replaced by their average,
    which preserves the trace (constant residual eigenvalue method, Laloux et al., 1999; López de Prado, 2020).
    """
    matrix, labels = _as_matrix(corr)
    n_assets = matrix.shape[0]
    lambda_max = (1 + np.sqrt(n_assets / n_observations)) ** 2
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    noise = eigenvalues < lambda_max
    if noise.all() or not noise.any():
        return corr
    eigenvalues = eigenvalues.copy()
    eigenvalues[noise] = eigenvalues[noise].mean()
    denoised = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    denoised = covariance_to_correlation(denoised)
    return pd.DataFrame(denoised, index=labels, columns=labels) if isinstance(corr, pd.DataFrame) else denoised


def denoised_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Covariance matrix with a random-matrix-theory denoised correlation structure

    :param returns: periodic returns (dates x assets)
    :return: covariance matrix per period, keeping the sample volatilities
    """
    clean = _clean_returns(returns)
    cov = clean.cov()
    vols = np.sqrt(np.diag(cov.to_numpy()))
    corr = denoise_correlation(covariance_to_correlation(cov.to_numpy()), len(clean))
    return _wrap(corr * np.outer(vols, vols), clean)


def _align_weights(weights: Union[pd.Series, np.ndarray], cov: ArrayLike) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    matrix, labels = _as_matrix(cov)
    if isinstance(weights, pd.Series):
        if isinstance(cov, pd.DataFrame):
            return weights.reindex(labels).fillna(0).to_numpy(dtype=float), matrix, labels
        return weights.to_numpy(dtype=float), matrix, weights.index
    return np.asarray(weights, dtype=float), matrix, labels


def portfolio_volatility(weights: pd.Series, cov: ArrayLike) -> float:
    """
    Volatility of a portfolio

    :param weights: portfolio weights
    :param cov: covariance matrix of the assets
    :return: :math:`\\sqrt{w^\\top \\Sigma w}`, in the units of the covariance matrix
    """
    w, matrix, _ = _align_weights(weights, cov)
    return float(np.sqrt(w @ matrix @ w))


def risk_contributions(weights: pd.Series, cov: ArrayLike) -> pd.Series:
    """
    Percentage contribution of each asset to portfolio variance (Euler allocation)

    :param weights: portfolio weights
    :param cov: covariance matrix of the assets
    :return: contributions :math:`w_i (\\Sigma w)_i / w^\\top \\Sigma w`, summing to one
    """
    w, matrix, labels = _align_weights(weights, cov)
    marginal = matrix @ w
    return pd.Series(w * marginal / (w @ marginal), index=labels)


def _tail_inputs(returns: Union[pd.Series, np.ndarray]) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        raise MqValueError('at least two observations are required')
    return r


def _check_confidence(confidence: float):
    if not 0 < confidence < 1:
        raise MqValueError('confidence must be in (0, 1)')


def value_at_risk(
    returns: Union[pd.Series, pd.DataFrame], confidence: float = 0.95, method: VaRMethod = VaRMethod.HISTORICAL
) -> Union[float, pd.Series]:
    """
    Value-at-Risk: the loss that is not exceeded with the given confidence

    :param returns: periodic returns; a DataFrame yields one value per column
    :param confidence: confidence level, e.g. 0.95 or 0.99
    :param method: historical quantile, Gaussian, or Cornish-Fisher (adjusted for skewness and excess kurtosis)
    :return: VaR as a positive fraction of portfolio value, over one period

    **Usage**

    The Cornish-Fisher expansion corrects the Gaussian quantile :math:`z` for skewness :math:`S` and excess
    kurtosis :math:`K`:
    :math:`z_{cf} = z + (z^2 - 1) S / 6 + (z^3 - 3z) K / 24 - (2z^3 - 5z) S^2 / 36` (Favre & Galeano, 2002).
    The expansion is only reliable for moderate departures from normality; with very heavy tails (infinite
    fourth moment) the sample kurtosis explodes and so does the adjustment (Maillard, 2012).
    """
    if isinstance(returns, pd.DataFrame):
        return returns.apply(lambda col: value_at_risk(col, confidence, method))
    _check_confidence(confidence)
    r = _tail_inputs(returns)
    method = VaRMethod(method)
    if method == VaRMethod.HISTORICAL:
        return float(-np.quantile(r, 1 - confidence))
    mu, sigma = r.mean(), r.std(ddof=1)
    z = stats.norm.ppf(1 - confidence)
    if method == VaRMethod.CORNISH_FISHER:
        s = stats.skew(r, bias=False)
        k = stats.kurtosis(r, fisher=True, bias=False)
        z = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
    return float(-(mu + z * sigma))


def expected_shortfall(
    returns: Union[pd.Series, pd.DataFrame], confidence: float = 0.95, method: VaRMethod = VaRMethod.HISTORICAL
) -> Union[float, pd.Series]:
    """
    Expected Shortfall (Conditional VaR): the average loss beyond the VaR

    :param returns: periodic returns; a DataFrame yields one value per column
    :param confidence: confidence level, e.g. 0.975 as in the Basel Fundamental Review of the Trading Book
    :param method: historical or Gaussian
    :return: expected shortfall as a positive fraction of portfolio value, over one period

    **Usage**

    Unlike VaR, Expected Shortfall is sub-additive and therefore rewards diversification (Artzner et al., 1999;
    Acerbi & Tasche, 2002).
    """
    if isinstance(returns, pd.DataFrame):
        return returns.apply(lambda col: expected_shortfall(col, confidence, method))
    _check_confidence(confidence)
    r = _tail_inputs(returns)
    method = VaRMethod(method)
    if method == VaRMethod.HISTORICAL:
        cutoff = np.quantile(r, 1 - confidence)
        return float(-r[r <= cutoff].mean())
    if method == VaRMethod.GAUSSIAN:
        mu, sigma = r.mean(), r.std(ddof=1)
        z = stats.norm.ppf(1 - confidence)
        return float(-(mu - sigma * stats.norm.pdf(z) / (1 - confidence)))
    raise MqValueError(f'expected shortfall is not available for method {method.value}')
