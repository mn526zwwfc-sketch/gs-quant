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

Backtesting and out-of-sample validation.

A backtest is a historical simulation, not a proof. This module bakes in the safeguards recommended by the
literature on backtest overfitting (Bailey, Borwein, López de Prado & Zhu, 2014; Harvey, Liu & Zhu, 2016):

* Execution is delayed by at least one period after the signal, removing look-ahead bias.
* Weights drift with prices between rebalances and transaction costs are charged on the actual turnover
  (Novy-Marx & Velikov, 2016, show that costs eliminate many published anomalies).
* Allocations are estimated on a rolling window strictly before each rebalance date (walk-forward).
* Cross-validation splits for overlapping labels are purged and embargoed (López de Prado, 2018, ch. 7).
"""

from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import numpy as np
import pandas as pd

from gs_quant.errors import MqValueError
from gs_quant.research.performance import infer_periods_per_year, performance_summary


@dataclass
class BacktestResult:
    """
    Output of :func:`backtest`. ``weights`` are the positions held during each period, after drift
    """

    returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series

    @property
    def equity_curve(self) -> pd.Series:
        """Cumulative wealth of one unit invested, net of costs"""
        return (1 + self.returns).cumprod()

    def summary(self, risk_free: float = 0.0, periods_per_year: Optional[float] = None) -> pd.Series:
        """
        Performance statistics of the net returns, plus average annual turnover and cost drag

        :param risk_free: annual risk-free rate
        :param periods_per_year: observations per year; inferred from the index when omitted
        :return: statistics indexed by name
        """
        stats = performance_summary(self.returns, risk_free, periods_per_year=periods_per_year)
        ppy = periods_per_year or infer_periods_per_year(self.returns.index)
        stats['annual_turnover'] = float(self.turnover.mean() * ppy)
        stats['annual_cost_drag'] = float(self.costs.mean() * ppy)
        return stats


def rebalance_dates(index: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    """
    Last available date of each period

    :param index: available dates
    :param frequency: pandas period alias, e.g. 'W', 'M', 'Q' or 'Y'
    :return: the last date of ``index`` in each period
    """
    index = pd.DatetimeIndex(index).sort_values()
    last = index.to_series().groupby(index.to_period(frequency)).max()
    return pd.DatetimeIndex(last.to_numpy())


def backtest(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    transaction_cost_bps: float = 10.0,
    execution_lag: int = 1,
    rebalance: Optional[str] = None,
) -> BacktestResult:
    """
    Simulate a strategy from target weights

    :param returns: simple returns of the assets (dates x assets); the return on date t is earned by the positions
        held from the close of t-1 to the close of t. Missing returns are treated as zero
    :param weights: target weights decided at the close of each date; all-NaN rows mean "no new target" and
        missing assets in other rows mean a zero weight
    :param transaction_cost_bps: one-way cost, in basis points of traded notional
    :param execution_lag: number of periods between the signal and the trade. With 1, a target computed at the
        close of t is traded at the close of t and earns returns from t+1 onwards. Use 2 to be conservative
    :param rebalance: optional pandas period alias ('W', 'M', 'Q', 'Y'); only the last target in each period is
        traded, which reduces turnover
    :return: :class:`BacktestResult`

    **Usage**

    Between rebalances, positions drift with asset returns: :math:`w_{t}^{-} = w_{t-1} (1 + r_t) / (1 + r_{p,t})`,
    where :math:`r_{p,t}` is the portfolio return net of costs.
    On rebalance dates, turnover is :math:`\\sum_i |w_{i,t}^{*} - w_{i,t}^{-}|` and costs are charged on it.
    Uninvested capital earns nothing, so strategy returns are excess returns when the asset returns are.
    """
    if execution_lag < 1:
        raise MqValueError('execution_lag must be at least 1 to avoid look-ahead bias')
    returns = returns.sort_index()
    targets = weights.sort_index().reindex(columns=returns.columns)
    targets = targets[targets.notna().any(axis=1)]
    if rebalance is not None:
        targets = targets.loc[rebalance_dates(targets.index, rebalance)]

    # map each signal date to the position of the first return date on or after it, then delay execution
    positions = returns.index.searchsorted(targets.index, side='left') + execution_lag
    n_dates, n_assets = returns.shape
    schedule = np.full((n_dates, n_assets), np.nan)
    # when several signals map to the same execution date, the latest one wins
    latest = (len(positions) - 1 - np.unique(positions[::-1], return_index=True)[1]).astype(int)
    latest = latest[positions[latest] < n_dates]
    schedule[positions[latest]] = targets.to_numpy(dtype=float)[latest]
    has_target = ~np.isnan(schedule).all(axis=1)
    schedule = np.nan_to_num(schedule)

    asset_returns = np.nan_to_num(returns.to_numpy(dtype=float))
    cost_rate = transaction_cost_bps / 1e4
    held = np.zeros((n_dates, n_assets))
    gross = np.zeros(n_dates)
    turnover = np.zeros(n_dates)
    current = np.zeros(n_assets)
    for t in range(n_dates):
        if has_target[t]:
            turnover[t] = np.abs(schedule[t] - current).sum()
            current = schedule[t].copy()
        held[t] = current
        gross[t] = current @ asset_returns[t]
        growth = 1 + gross[t] - turnover[t] * cost_rate
        current = current * (1 + asset_returns[t]) / growth if growth > 0 else np.zeros(n_assets)

    costs = turnover * cost_rate
    index = returns.index
    return BacktestResult(
        returns=pd.Series(gross - costs, index=index, name='net'),
        gross_returns=pd.Series(gross, index=index, name='gross'),
        weights=pd.DataFrame(held, index=index, columns=returns.columns),
        turnover=pd.Series(turnover, index=index, name='turnover'),
        costs=pd.Series(costs, index=index, name='costs'),
    )


def rolling_allocation(
    returns: pd.DataFrame,
    allocator: Callable[[pd.DataFrame], pd.Series],
    lookback: int = 252,
    rebalance: str = 'M',
    min_observations: Optional[int] = None,
) -> pd.DataFrame:
    """
    Walk-forward portfolio construction

    :param returns: simple returns of the assets (dates x assets)
    :param allocator: function mapping a window of returns to weights, e.g.
        ``lambda r: risk_parity(ledoit_wolf_covariance(r))``
    :param lookback: number of observations in the estimation window
    :param rebalance: pandas period alias of the rebalance frequency
    :param min_observations: minimum complete observations required per asset; defaults to ``lookback // 2``
    :return: target weights on each rebalance date, estimated only with data up to that date

    **Usage**

    Parameters are re-estimated at every rebalance using only the trailing window, so the resulting backtest is
    genuinely out of sample. Assets without enough history in the window are excluded (weight zero).
    """
    min_observations = min_observations or lookback // 2
    rows = {}
    for date in rebalance_dates(returns.index, rebalance):
        window = returns.loc[:date].tail(lookback)
        window = window.loc[:, window.notna().sum() >= min_observations].dropna()
        if window.shape[1] == 0 or len(window) < min_observations:
            continue
        rows[date] = allocator(window)
    if not rows:
        raise MqValueError('no rebalance date has enough history')
    return pd.DataFrame.from_dict(rows, orient='index').reindex(columns=returns.columns).fillna(0.0)


def walk_forward_splits(
    n_observations: int, train_size: int, test_size: int, expanding: bool = False
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Sequential train/test splits for time series

    :param n_observations: number of observations
    :param train_size: size of the (initial) training window
    :param test_size: size of each test window
    :param expanding: grow the training window instead of rolling it
    :return: iterator of (train positions, test positions); every test window follows its training window
    """
    if train_size <= 0 or test_size <= 0:
        raise MqValueError('train_size and test_size must be positive')
    start = train_size
    while start < n_observations:
        train = np.arange(0 if expanding else start - train_size, start)
        test = np.arange(start, min(start + test_size, n_observations))
        yield train, test
        start += test_size


def purged_kfold_splits(
    n_observations: int, n_splits: int = 5, label_horizon: int = 0, embargo: float = 0.01
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    K-fold cross-validation for financial labels that overlap in time (López de Prado, 2018)

    :param n_observations: number of observations, in chronological order
    :param n_splits: number of folds
    :param label_horizon: number of periods spanned by each label, e.g. 21 for a label based on the next month's
        return; training observations whose label window overlaps the test fold are purged
    :param embargo: fraction of observations dropped from training right after each test fold, to prevent leakage
        through serially correlated features
    :return: iterator of (train positions, test positions)

    **Usage**

    Standard k-fold cross-validation leaks information when labels overlap, grossly overstating out-of-sample
    performance. Purging removes the training observations whose labels overlap the test fold, and the embargo
    removes those immediately after it (López de Prado, 2018, "Advances in Financial Machine Learning", ch. 7).
    """
    if not 2 <= n_splits <= n_observations:
        raise MqValueError('n_splits must be between 2 and the number of observations')
    gap = int(np.ceil(embargo * n_observations))
    positions = np.arange(n_observations)
    for test in np.array_split(positions, n_splits):
        start, end = test[0], test[-1]
        keep = (positions + label_horizon < start) | (positions > end + gap)
        yield positions[keep], test
