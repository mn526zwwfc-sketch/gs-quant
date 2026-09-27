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

Walk through of the growth company valuation and the long-term strategy.

Run ``python -m gs_quant.research.growth_demo`` for a synthetic universe, or provide your own data:

    python -m gs_quant.research.growth_demo --fundamentals fundamentals.csv --prices prices.csv [--sectors sectors.csv]

* fundamentals.csv: one row per company and fiscal year, columns as in
  :func:`gs_quant.research.fundamentals.prepare_fundamentals` (e.g. built with ``parse_sec_companyfacts``)
* prices.csv: dates in the first column, one column of adjusted prices per ticker
* sectors.csv: columns ``ticker`` and ``sector``
"""

import argparse
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from gs_quant.research.backtest import backtest
from gs_quant.research.growth_strategy import GrowthStrategyConfig, LongTermGrowthStrategy
from gs_quant.research.growth_valuation import growth_dcf, value_company
from gs_quant.research.performance import information_ratio, performance_summary

SECTORS = ('Technology', 'Health Care', 'Consumer', 'Industrials', 'Communication')


def synthetic_growth_universe(
    n_companies: int = 120, years: int = 12, seed: int = 11
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Simulate fundamentals and prices for a universe of companies with heterogeneous, fading growth

    :param n_companies: number of companies
    :param years: number of fiscal years (and years of daily prices)
    :param seed: random seed
    :return: tuple (annual fundamentals, daily prices, sectors)

    **Usage**

    Each company has an initial growth rate that fades geometrically towards 2.5% (by 20% a year, in line with the
    weak persistence of growth documented by Chan, Karceski & Lakonishok, 2003), its own margin, capital intensity
    and R&D spend. Its share price equals a DCF fair value times a mean-reverting mispricing factor and a
    mean-reverting market valuation cycle.
    Prices therefore contain exploitable mispricing by construction: backtest results on this data validate the
    mechanics of the strategy, not its real-world profitability.
    """
    rng = np.random.default_rng(seed)
    tickers = [f'GRW{i:03d}' for i in range(n_companies)]
    sectors = pd.Series(rng.choice(SECTORS, n_companies), index=tickers, name='sector')
    first_year = 2010
    dates = pd.bdate_range(f'{first_year + 1}-01-01', f'{first_year + years}-12-31')
    market = np.zeros(len(dates))
    market_shocks = rng.normal(0.0, 0.009, len(dates))
    for t in range(1, len(dates)):
        market[t] = market[t - 1] * (1 - 1 / 504) + market_shocks[t]  # valuation cycle, half-life about 1.4 years
    rows, prices = [], {}
    for ticker in tickers:
        mu = rng.uniform(-0.02, 0.30)
        age = int(rng.integers(-8, 3))  # staggered life cycles: some companies only start maturing years later
        margin = rng.uniform(0.03, 0.30)
        sales_to_capital = rng.uniform(0.8, 3.0)
        gross_extra = rng.uniform(0.15, 0.40)
        rd = rng.uniform(0.05, 0.15) if sectors[ticker] in ('Technology', 'Health Care') else rng.uniform(0, 0.03)
        shares = rng.uniform(50, 500)
        dilution = rng.uniform(0.0, 0.03)
        revenue = rng.uniform(500, 5000)
        fair, ends = [], []
        for year in range(years + 1):
            expected_growth = 0.025 + (mu - 0.025) * 0.8 ** max(year + age, 0)
            growth = expected_growth + rng.normal(0, 0.03)
            revenue *= 1 + growth
            shares *= 1 + dilution
            m = margin + rng.normal(0, 0.015)
            ebit = revenue * m
            invested = revenue / sales_to_capital
            debt, cash = 0.3 * invested, 0.1 * revenue
            equity = invested - debt + cash
            net_income = ebit * 0.79 - debt * 0.05
            rows.append(
                {
                    'ticker': ticker,
                    'period_end': pd.Timestamp(f'{first_year + year}-12-31'),
                    'filed': pd.Timestamp(f'{first_year + year}-12-31') + pd.Timedelta(days=int(rng.integers(40, 80))),
                    'revenue': revenue,
                    'gross_profit': revenue * (m + gross_extra),
                    'operating_income': ebit,
                    'pretax_income': ebit - debt * 0.05,
                    'income_tax': (ebit - debt * 0.05) * 0.21,
                    'net_income': net_income,
                    'operating_cash_flow': net_income + 0.03 * revenue + rng.normal(0, 0.02) * revenue,
                    'capex': revenue * growth / sales_to_capital + 0.03 * revenue,
                    'research_development': rd * revenue,
                    'total_assets': equity + debt + 0.25 * revenue,
                    'current_assets': 0.3 * revenue + cash,
                    'current_liabilities': 0.25 * revenue,
                    'total_debt': debt,
                    'cash': cash,
                    'total_equity': equity,
                    'shares_outstanding': shares,
                }
            )
            forward = 0.025 + (mu - 0.025) * 0.8 ** np.maximum(np.arange(year + 1, year + 11) + age, 0)
            value = growth_dcf(
                revenue, forward, margin, 0.085, 0.025, 0.21, sales_to_capital, None, debt - cash, shares
            )
            fair.append(np.log(max(value.value_per_share, 1.0)))
            ends.append(pd.Timestamp(f'{first_year + year}-12-31'))
        log_fair = np.interp(dates.asi8, pd.DatetimeIndex(ends).asi8, fair)
        mispricing = np.zeros(len(dates))
        mispricing[0] = rng.normal(0, 0.3)
        shocks = rng.normal(0, 0.3 * np.sqrt(2 / 252), len(dates))
        for t in range(1, len(dates)):
            mispricing[t] = mispricing[t - 1] * (1 - 1 / 252) + shocks[t]
        beta = rng.uniform(0.7, 1.4)
        idiosyncratic = rng.normal(0, 0.01, len(dates))
        prices[ticker] = np.exp(log_fair + beta * market + mispricing + idiosyncratic)
    return pd.DataFrame(rows), pd.DataFrame(prices, index=dates), sectors


def _section(title: str):
    print(f'\n{"=" * 110}\n{title}\n{"=" * 110}')


def run(
    fundamentals: pd.DataFrame,
    prices: pd.DataFrame,
    sectors: Optional[pd.Series] = None,
    config: Optional[GrowthStrategyConfig] = None,
    cost_bps: float = 20.0,
) -> dict:
    """
    Screen, value, select and backtest, printing the results

    :param fundamentals: annual fundamentals
    :param prices: daily adjusted prices
    :param sectors: optional sector of each ticker
    :param config: strategy configuration
    :param cost_bps: one-way transaction cost in basis points
    :return: dictionary with the strategy, its result, the backtest and the latest opportunities table
    """
    pd.set_option('display.width', 220)
    pd.set_option('display.max_columns', 30)
    config = config or GrowthStrategyConfig(n_holdings=min(20, max(5, prices.shape[1] // 4)))
    strategy = LongTermGrowthStrategy(fundamentals, prices, config, sectors)
    result = strategy.run()
    latest = result.weights.index[-1]
    screen = result.screens[latest]

    _section(f'1. OPPORTUNITIES ON {latest:%Y-%m-%d}: growth companies ranked by composite score')
    columns = {
        'price': 'price',
        'expected_value': 'fair_value',
        'margin_of_safety': 'safety',
        'initial_growth': 'g_forecast',
        'implied_growth': 'g_implied',
        'revenue_cagr_3y': 'cagr_3y',
        'roic': 'roic',
        'wacc': 'wacc',
        'f_score': 'F',
        'g_score': 'G',
        'composite_score': 'score',
        'eligible': 'buy_ok',
        'selected': 'held',
    }
    eligible = screen[screen['eligible']]
    ranked = screen[screen['holdable']]
    print(
        f'Universe {len(screen)} companies: {len(eligible)} meet the entry rules, {len(ranked)} the (wider) hold '
        f'rules, {int(screen["selected"].sum())} held'
    )
    print(ranked[list(columns)].rename(columns=columns).head(15).round(3))
    if ranked.empty:
        print('No company qualifies today: the portfolio is in cash.')
        return {'strategy': strategy, 'result': result, 'opportunities': eligible}

    _section('2. VALUATION DETAIL OF THE TOP-RANKED COMPANY')
    top = ranked.index[0]
    row = ranked.loc[top]
    valuation = value_company(row, row['price'], row['beta'] if np.isfinite(row['beta']) else 1.0, config.valuation)
    print(
        f'{top}: price {row["price"]:.2f} | WACC {valuation.wacc:.2%} | cost of equity {valuation.cost_of_equity:.2%}'
    )
    print('Scenario values: ' + ', '.join(f'{k} {v:.2f}' for k, v in valuation.scenario_values.items()))
    print(
        f'Probability-weighted value {valuation.expected_value:.2f} -> margin of safety '
        f'{valuation.margin_of_safety:.1%}; residual income value {valuation.residual_income_value:.2f}'
    )
    print(
        f'Growth forecast {valuation.initial_growth:.1%} vs growth implied by the price '
        f'{valuation.implied_growth:.1%} (terminal value = {valuation.terminal_value_share:.0%} of the base DCF)'
    )

    _section('3. CURRENT PORTFOLIO')
    holdings = result.holdings()
    print(holdings[['weight', 'margin_of_safety', 'revenue_cagr_3y', 'roic', 'composite_score']].round(3))
    if sectors is not None:
        print(
            '\nSector weights:\n'
            + holdings['weight'].groupby(sectors.reindex(holdings.index)).sum().round(3).to_string()
        )

    _section(f'4. BACKTEST (net of {cost_bps:.0f} bps costs) vs equal-weighted universe')
    simulation = strategy.backtest(result, transaction_cost_bps=cost_bps)
    start = result.weights.index[0]
    benchmark_weights = pd.DataFrame(1 / prices.shape[1], index=result.weights.index, columns=prices.columns)
    benchmark = backtest(strategy.returns, benchmark_weights, transaction_cost_bps=cost_bps)
    net = pd.DataFrame({'strategy': simulation.returns, 'equal_weight_universe': benchmark.returns}).loc[start:]
    print(performance_summary(net).round(3))
    print(f'\nInformation ratio vs benchmark: {information_ratio(net["strategy"], net["equal_weight_universe"]):.2f}')
    print(f'Annual turnover: {simulation.summary()["annual_turnover"]:.2f}x')

    _section('5. EXPORT TO gs_quant.markets')
    position_set = strategy.to_position_set(result)
    print(position_set.to_frame(add_tags=True).head(10))
    print('\nWith a Marquee session: position_set.resolve(); position_set.price() or create a Basket from it.')
    return {'strategy': strategy, 'result': result, 'backtest': simulation, 'opportunities': eligible}


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fundamentals', help='CSV of annual fundamentals')
    parser.add_argument('--prices', help='CSV of daily prices: first column dates, one column per ticker')
    parser.add_argument('--sectors', help='CSV with columns ticker and sector')
    parser.add_argument('--holdings', type=int, default=None, help='number of companies to hold')
    parser.add_argument('--cost-bps', type=float, default=20.0, help='one-way transaction cost in basis points')
    parser.add_argument('--seed', type=int, default=11, help='random seed for the synthetic universe')
    args = parser.parse_args(argv)
    if bool(args.fundamentals) != bool(args.prices):
        parser.error('--fundamentals and --prices must be provided together')
    if args.fundamentals:
        fundamentals = pd.read_csv(args.fundamentals)
        prices = pd.read_csv(args.prices, index_col=0, parse_dates=True)
        sectors = pd.read_csv(args.sectors).set_index('ticker')['sector'] if args.sectors else None
    else:
        fundamentals, prices, sectors = synthetic_growth_universe(seed=args.seed)
    config = GrowthStrategyConfig(n_holdings=args.holdings) if args.holdings else None
    run(fundamentals, prices, sectors, config, args.cost_bps)


if __name__ == '__main__':
    main()
