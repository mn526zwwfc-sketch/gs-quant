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
    backtest,
    equal_weight,
    ledoit_wolf_covariance,
    purged_kfold_splits,
    rebalance_dates,
    risk_parity,
    rolling_allocation,
    walk_forward_splits,
)
from gs_quant.research.demo import main, run, synthetic_prices


@pytest.fixture
def returns():
    rng = np.random.default_rng(9)
    index = pd.bdate_range('2020-01-01', periods=300)
    return pd.DataFrame(rng.normal(0.0004, 0.01, (300, 3)), index=index, columns=['x', 'y', 'z'])


def _dollar_simulation(returns, targets_by_position, cost_bps):
    """Reference implementation tracking dollar holdings"""
    nav, holdings, navs = 1.0, np.zeros(returns.shape[1]), []
    for t, r in enumerate(returns.to_numpy()):
        cost = 0.0
        if t in targets_by_position:
            target = nav * targets_by_position[t]
            cost = np.abs(target - holdings).sum() * cost_bps / 1e4
            holdings = target
        nav = nav + holdings @ r - cost
        holdings = holdings * (1 + r)
        navs.append(nav)
    return pd.Series(navs, index=returns.index)


def test_rebalance_dates(returns):
    dates = rebalance_dates(returns.index, 'M')
    assert dates[0] == pd.Timestamp('2020-01-31')
    assert len(dates) == len(returns.index.to_period('M').unique())
    assert rebalance_dates(returns.index[::-1], 'Q')[0] == pd.Timestamp('2020-03-31')


def test_backtest_matches_dollar_simulation(returns):
    target = np.array([0.5, 0.3, 0.2])
    weights = pd.DataFrame(np.tile(target, (len(returns), 1)), index=returns.index, columns=returns.columns)
    result = backtest(returns, weights, transaction_cost_bps=25, rebalance='M')
    # each month-end target is traded on the next close; the last one falls after the end of the data
    positions = {returns.index.get_loc(d) + 1 for d in rebalance_dates(returns.index, 'M')} - {len(returns)}
    expected = _dollar_simulation(returns, {p: target for p in positions}, 25)
    assert np.allclose(result.equity_curve, expected, atol=1e-12)
    assert np.isclose(result.turnover.iloc[min(positions)], 1.0)  # initial purchase
    assert (result.turnover[result.turnover > 0].index.isin(returns.index[sorted(positions)])).all()
    assert np.allclose(result.returns, result.gross_returns - result.costs)


def test_backtest_has_no_look_ahead(returns):
    # a perfect-foresight signal must not earn the same-day return
    oracle = np.sign(returns)
    result = backtest(returns, oracle, transaction_cost_bps=0)
    expected = (oracle.shift(1) * returns).sum(axis=1)
    assert np.allclose(result.gross_returns.iloc[2:], expected.iloc[2:], atol=0.02)
    assert result.gross_returns.mean() < (oracle * returns).sum(axis=1).mean() / 5
    lagged = backtest(returns, oracle, transaction_cost_bps=0, execution_lag=2)
    assert lagged.weights.iloc[:2].eq(0).all().all()
    with pytest.raises(MqValueError):
        backtest(returns, oracle, execution_lag=0)


def test_backtest_signal_dates_outside_trading_calendar(returns):
    weekend = pd.Timestamp('2020-02-01')  # Saturday: executed on the close of the next trading day
    weights = pd.DataFrame({'x': [1.0], 'y': [0.0], 'z': [np.nan]}, index=[weekend])
    result = backtest(returns, weights, execution_lag=1)
    first = result.weights['x'].ne(0).idxmax()
    assert first == pd.Timestamp('2020-02-04')
    assert result.weights['z'].eq(0).all()


def test_backtest_latest_signal_wins(returns):
    weights = pd.DataFrame(
        {'x': [1.0, 0.0], 'y': [0.0, 1.0]}, index=[pd.Timestamp('2020-02-01'), pd.Timestamp('2020-02-02')]
    )
    result = backtest(returns, weights)
    assert result.weights.loc['2020-02-04', 'y'] == 1.0
    assert result.turnover.sum() == 1.0


def test_backtest_ignores_empty_rows_and_drifts(returns):
    weights = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    weights.iloc[10] = [1.0, 0.0, 0.0]
    result = backtest(returns, weights, transaction_cost_bps=0)
    assert result.turnover.sum() == 1.0
    assert np.allclose(result.weights.iloc[11:, 0], 1.0)  # a single asset keeps a weight of one
    assert np.allclose(result.equity_curve.iloc[-1], (1 + returns['x'].iloc[11:]).prod())


def test_backtest_summary(returns):
    weights = pd.DataFrame([[1 / 3] * 3], index=[returns.index[0]], columns=returns.columns)
    summary = backtest(returns, weights, rebalance='M').summary()
    assert {'sharpe_ratio', 'annual_turnover', 'annual_cost_drag'} <= set(summary.index)
    assert summary['annual_cost_drag'] > 0


def test_rolling_allocation_is_out_of_sample(returns):
    seen = []

    def allocator(window):
        seen.append(window.index[-1])
        return risk_parity(ledoit_wolf_covariance(window))

    weights = rolling_allocation(returns, allocator, lookback=60, rebalance='M')
    assert list(weights.index) == seen
    assert all(date in rebalance_dates(returns.index, 'M') for date in seen)
    assert np.allclose(weights.sum(axis=1), 1)
    shocked = returns.copy()
    shocked.iloc[200:] *= 5
    pd.testing.assert_frame_equal(
        weights.loc[: returns.index[199]],
        rolling_allocation(shocked, allocator, lookback=60, rebalance='M').loc[: returns.index[199]],
    )


def test_rolling_allocation_handles_missing_history(returns):
    partial = returns.copy()
    partial.iloc[:150, 2] = np.nan
    weights = rolling_allocation(partial, equal_weight, lookback=60, rebalance='M')
    assert np.allclose(weights.loc[: partial.index[140], 'z'], 0)
    assert np.allclose(weights.iloc[-1], 1 / 3)
    with pytest.raises(MqValueError):
        rolling_allocation(returns.iloc[:10], equal_weight, lookback=60)


def test_walk_forward_splits():
    rolling = list(walk_forward_splits(10, train_size=4, test_size=3))
    assert [(list(a), list(b)) for a, b in rolling] == [
        ([0, 1, 2, 3], [4, 5, 6]),
        ([3, 4, 5, 6], [7, 8, 9]),
    ]
    expanding = list(walk_forward_splits(10, 4, 4, expanding=True))
    assert list(expanding[-1][0]) == list(range(8)) and list(expanding[-1][1]) == [8, 9]
    with pytest.raises(MqValueError):
        list(walk_forward_splits(10, 0, 2))


def test_purged_kfold_splits():
    splits = list(purged_kfold_splits(100, n_splits=5, label_horizon=5, embargo=0.03))
    assert len(splits) == 5
    assert sorted(np.concatenate([test for _, test in splits])) == list(range(100))
    train, test = splits[2]  # test fold is 40..59
    assert list(test) == list(range(40, 60))
    assert train[train < 40].max() == 34  # labels of 35..39 overlap the fold and are purged
    assert train[train > 59].min() == 63  # embargo of 3 observations
    plain = list(purged_kfold_splits(100, n_splits=5, label_horizon=0, embargo=0))
    assert all(len(train) + len(test) == 100 for train, test in plain)
    with pytest.raises(MqValueError):
        list(purged_kfold_splits(3, n_splits=5))


def test_demo_runs_end_to_end(capsys, tmp_path):
    prices = synthetic_prices(n_assets=6, years=3, seed=1)
    table = run(prices, cost_bps=5)
    assert 'hrp' in table.columns and 'ts_momentum_voltgt' in table.columns
    assert table.loc['annualized_volatility'].notna().all()
    csv = tmp_path / 'prices.csv'
    synthetic_prices(n_assets=5, years=3, seed=2).to_csv(csv)
    main(['--csv', str(csv), '--cost-bps', '2'])
    assert 'deflated Sharpe ratio' in capsys.readouterr().out
