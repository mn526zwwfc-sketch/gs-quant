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
from gs_quant.markets.position_set import PositionSet
from gs_quant.research import GrowthStrategyConfig, LongTermGrowthStrategy
from gs_quant.research.growth_demo import main, run, synthetic_growth_universe
from gs_quant.research.growth_strategy import _cap_weights


@pytest.fixture(scope='module')
def universe():
    return synthetic_growth_universe(n_companies=50, years=7, seed=3)


@pytest.fixture(scope='module')
def config():
    return GrowthStrategyConfig(n_holdings=8, max_weight=0.2, rebalance='Q')


@pytest.fixture(scope='module')
def strategy(universe, config):
    fundamentals, prices, sectors = universe
    return LongTermGrowthStrategy(fundamentals, prices, config, sectors)


@pytest.fixture(scope='module')
def result(strategy):
    return strategy.run()


def test_config_validation():
    with pytest.raises(MqValueError):
        GrowthStrategyConfig(weighting='magic')
    with pytest.raises(MqValueError):
        GrowthStrategyConfig(score_weights={'luck': 1.0})
    with pytest.raises(MqValueError):
        GrowthStrategyConfig(buffer=0.5)
    with pytest.raises(MqValueError):
        GrowthStrategyConfig(growth_quantile=0)
    with pytest.raises(MqValueError):
        GrowthStrategyConfig(min_margin_of_safety=0.1, hold_margin_of_safety=0.2)


def test_cap_weights():
    capped = _cap_weights(pd.Series([0.7, 0.2, 0.1], index=list('abc')), 0.5)
    assert np.isclose(capped.sum(), 1) and np.isclose(capped['a'], 0.5)
    assert np.isclose(capped['b'] / capped['c'], 2)  # excess redistributed proportionally
    few = _cap_weights(pd.Series([0.9, 0.1], index=['a', 'b']), 0.2)
    assert few.tolist() == [0.2, 0.2]  # the rest stays in cash


def test_screen_eligibility_rules(strategy, config):
    table = strategy.screen(pd.Timestamp('2016-12-30'))
    eligible = table[table['eligible']]
    assert len(eligible) > 0
    assert (eligible['history_years'] >= config.min_history_years).all()
    assert (eligible['revenue_cagr_3y'] >= max(table['revenue_cagr_3y'].quantile(0.5), 0.05)).all()
    assert (eligible['f_score'] >= config.min_f_score).all()
    assert (eligible['operating_margin'] > 0).all()
    assert (eligible['margin_of_safety'] >= 0).all()
    assert (eligible['roic'] > eligible['wacc']).all()
    # the hold rules are a strictly wider superset of the entry rules
    holdable = table[table['holdable']]
    assert set(eligible.index) <= set(holdable.index)
    assert (holdable['margin_of_safety'] >= config.hold_margin_of_safety).all()
    assert (holdable['f_score'] >= config.min_f_score - 1).all()
    assert (holdable['revenue_cagr_3y'] >= table['revenue_cagr_3y'].quantile(1 - 0.5 * config.buffer)).all()
    assert table.loc[~table['holdable'], 'composite_score'].isna().all()
    assert holdable['composite_score'].is_monotonic_decreasing
    assert holdable['rank'].tolist() == list(range(1, len(holdable) + 1))


def test_screen_uses_only_published_information(universe, config):
    fundamentals, prices, sectors = universe
    date = pd.Timestamp('2016-06-30')
    base = LongTermGrowthStrategy(fundamentals, prices, config, sectors).screen(date)
    future = fundamentals.copy()
    unpublished = pd.to_datetime(future['filed']) > date
    future.loc[unpublished, 'revenue'] *= 10
    shocked_prices = prices.copy()
    shocked_prices.loc[shocked_prices.index > date] *= 3
    shocked = LongTermGrowthStrategy(future, shocked_prices, config, sectors).screen(date)
    pd.testing.assert_frame_equal(base, shocked)


def test_run_weights(result, config):
    weights = result.weights
    assert weights.index.is_monotonic_increasing
    assert (weights >= 0).all().all()
    assert (weights.sum(axis=1) <= 1 + 1e-9).all()
    assert (weights.max(axis=1) <= config.max_weight + 1e-9).all()
    assert ((weights > 0).sum(axis=1) <= config.n_holdings).all()
    for date, screen in result.screens.items():
        held = weights.loc[date][weights.loc[date] > 0].index
        assert set(held) == set(screen.index[screen['selected']])
        assert screen.loc[held, 'holdable'].all()


def test_holdings_and_position_set(strategy, result):
    holdings = result.holdings()
    assert holdings['weight'].is_monotonic_decreasing
    assert {'margin_of_safety', 'implied_growth', 'composite_score'} <= set(holdings.columns)
    position_set = strategy.to_position_set(result)
    assert isinstance(position_set, PositionSet)
    assert position_set.date == result.weights.index[-1].date()
    frame = position_set.to_frame(add_tags=True)
    assert set(frame['identifier']) == set(holdings.index)
    assert np.isclose(frame['weight'].sum(), holdings['weight'].sum())
    assert {'composite_score', 'margin_of_safety', 'implied_growth'} <= set(frame.columns)


def test_buffer_reduces_turnover(universe):
    fundamentals, prices, sectors = universe

    def name_changes(buffer, hold_margin_of_safety):
        config = GrowthStrategyConfig(
            n_holdings=6,
            buffer=buffer,
            hold_margin_of_safety=hold_margin_of_safety,
            weighting='equal',
            max_weight=0.25,
        )
        held = LongTermGrowthStrategy(fundamentals, prices, config, sectors).run().weights > 0
        return int((held != held.shift(1)).iloc[1:].sum().sum())

    strict = name_changes(1.0, 0.0)  # identical entry and exit rules
    assert name_changes(2.0, -0.15) < strict


def test_sector_limit(universe):
    fundamentals, prices, sectors = universe
    config = GrowthStrategyConfig(n_holdings=8, max_per_sector=1, weighting='inverse_volatility', max_weight=0.2)
    weights = LongTermGrowthStrategy(fundamentals, prices, config, sectors).run().weights
    per_sector = (weights > 0).T.groupby(sectors).sum().T
    assert (per_sector <= 1).all().all()


@pytest.mark.parametrize('weighting', ['equal', 'hrp'])
def test_weighting_methods(universe, weighting):
    fundamentals, prices, sectors = universe
    config = GrowthStrategyConfig(n_holdings=5, weighting=weighting, max_weight=0.3)
    weights = LongTermGrowthStrategy(fundamentals, prices, config, sectors).run(start='2016-01-01').weights
    assert weights.index[0] >= pd.Timestamp('2016-01-01')
    invested = weights.sum(axis=1)
    held = (weights > 0).sum(axis=1)
    assert np.allclose(invested[held >= 4], 1)  # fully invested once enough companies qualify
    if weighting == 'equal':
        full = weights[held == 5]
        assert np.allclose(full[full > 0].stack(), 0.2)


def test_dates_without_opportunities_go_to_cash(universe):
    fundamentals, prices, sectors = universe
    config = GrowthStrategyConfig(n_holdings=5, min_margin_of_safety=10.0)  # nothing is ever that cheap
    strategy = LongTermGrowthStrategy(fundamentals, prices, config, sectors)
    result = strategy.run()
    assert (result.weights == 0).all().all()
    assert strategy.backtest(result).returns.eq(0).all()


def test_backtest_and_market_proxy(universe, config, result, strategy):
    simulation = strategy.backtest(result, transaction_cost_bps=10)
    assert simulation.costs.sum() > 0
    assert np.isclose((simulation.returns + simulation.costs - simulation.gross_returns).abs().max(), 0)
    fundamentals, prices, sectors = universe
    market = prices.mean(axis=1)
    with_market = LongTermGrowthStrategy(fundamentals, prices, config, sectors, market_prices=market)
    assert with_market.screen(pd.Timestamp('2016-12-30'))['beta'].notna().all()


def test_run_without_data(universe, config):
    fundamentals, prices, sectors = universe
    with pytest.raises(MqValueError):
        LongTermGrowthStrategy(fundamentals, prices, config, sectors).run(end='2010-01-01')


def test_growth_demo(universe, capsys, tmp_path):
    fundamentals, prices, sectors = universe
    output = run(fundamentals, prices, sectors, GrowthStrategyConfig(n_holdings=6, max_weight=0.2), cost_bps=10)
    assert {'strategy', 'result', 'backtest', 'opportunities'} <= set(output)
    assert 'EXPORT TO gs_quant.markets' in capsys.readouterr().out
    fundamentals.to_csv(tmp_path / 'fundamentals.csv', index=False)
    prices.to_csv(tmp_path / 'prices.csv')
    sectors.rename_axis('ticker').reset_index().to_csv(tmp_path / 'sectors.csv', index=False)
    main(
        [
            '--fundamentals',
            str(tmp_path / 'fundamentals.csv'),
            '--prices',
            str(tmp_path / 'prices.csv'),
            '--sectors',
            str(tmp_path / 'sectors.csv'),
            '--holdings',
            '6',
        ]
    )
    assert 'OPPORTUNITIES' in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(['--fundamentals', str(tmp_path / 'fundamentals.csv')])
