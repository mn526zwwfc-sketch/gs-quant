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

End-to-end walk through of :mod:`gs_quant.research`.

Run ``python -m gs_quant.research.demo`` for synthetic data, or pass ``--csv prices.csv`` with a date column
followed by one column of prices per asset.
"""

import argparse
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from gs_quant.research import (
    backtest,
    black_scholes,
    bond_analytics,
    clean_prices,
    compute_returns,
    cross_sectional_momentum,
    deflated_sharpe_ratio,
    equal_weight,
    expected_shortfall,
    hampel_filter,
    hierarchical_risk_parity,
    inverse_volatility,
    ledoit_wolf_covariance,
    ledoit_wolf_shrinkage,
    low_volatility,
    minimum_variance,
    moving_average_crossover,
    performance_summary,
    risk_parity,
    rolling_allocation,
    sharpe_ratio,
    time_series_momentum,
    validate_prices,
    value_at_risk,
    volatility_scaling,
)


def synthetic_prices(n_assets: int = 12, years: int = 12, seed: int = 7) -> pd.DataFrame:
    """
    Simulate daily prices with a common factor, sector clusters, volatility regimes and persistent drifts

    :param n_assets: number of assets
    :param years: length of the history
    :param seed: random seed
    :return: price levels (dates x assets), with a few deliberate data errors to be cleaned
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2010-01-01', periods=252 * years)
    n = len(dates)
    regime_vol = np.where(np.sin(np.arange(n) / 180) > 0.6, 2.0, 1.0)
    market = rng.normal(0, 0.008, n) * regime_vol
    sectors = rng.normal(0, 0.006, (n, 3)) * regime_vol[:, None]
    drift = np.cumsum(rng.normal(0, 0.00002, (n, n_assets)), axis=0) + 0.0002
    vols = np.linspace(0.008, 0.02, n_assets)
    returns = (
        drift
        + market[:, None] * rng.uniform(0.6, 1.4, n_assets)
        + sectors[:, np.arange(n_assets) % 3]
        + rng.normal(0, 1, (n, n_assets)) * vols * regime_vol[:, None]
    )
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=[f'ASSET_{i + 1:02d}' for i in range(n_assets)]
    )
    prices.iloc[n // 6, 0] = 0.0  # bad print
    prices.iloc[n // 4, 1 % n_assets] *= 1.5  # fat-finger spike
    prices.iloc[n // 3 : n // 3 + 10, 2 % n_assets] = prices.iloc[n // 3 - 1, 2 % n_assets]  # stale quotes
    prices.iloc[n // 2 : n // 2 + 3, 3 % n_assets] = np.nan  # missing data
    return prices


def _section(title: str):
    print(f'\n{"=" * 100}\n{title}\n{"=" * 100}')


def run(prices: pd.DataFrame, cost_bps: float = 10.0) -> pd.DataFrame:
    """
    Run the full workflow and print the results

    :param prices: raw price levels (dates x assets)
    :param cost_bps: one-way transaction cost in basis points
    :return: performance table of all strategies
    """
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 20)

    _section('1. DATA MANAGEMENT: audit, clean and robust outlier filtering')
    report = validate_prices(prices)
    issues = report.to_frame()
    print(issues[issues.sum(axis=1) > 0])
    prices = clean_prices(prices, max_fill=5)
    returns = hampel_filter(compute_returns(prices), window=63, n_sigmas=8.0)
    print(f'\nClean returns: {returns.shape[0]} dates x {returns.shape[1]} assets')

    _section('2. RISK: shrinkage covariance and tail risk')
    recent = returns.tail(252)
    print(f'Ledoit-Wolf shrinkage intensity (last year): {ledoit_wolf_shrinkage(recent):.3f}')
    ew = returns.mean(axis=1)
    print(f'Equal-weight daily VaR 99% historical:     {value_at_risk(ew, 0.99):.4%}')
    print(f'Equal-weight daily VaR 99% Cornish-Fisher: {value_at_risk(ew, 0.99, "cornish_fisher"):.4%}')
    print(f'Equal-weight daily ES 97.5% historical:    {expected_shortfall(ew, 0.975):.4%}')

    _section('3. PORTFOLIO CONSTRUCTION: walk-forward, monthly rebalance, Ledoit-Wolf covariance')
    allocators = {
        'equal_weight': equal_weight,
        'inverse_vol': lambda r: inverse_volatility(ledoit_wolf_covariance(r)),
        'min_variance': lambda r: minimum_variance(ledoit_wolf_covariance(r), max_weight=0.25),
        'risk_parity': lambda r: risk_parity(ledoit_wolf_covariance(r)),
        'hrp': lambda r: hierarchical_risk_parity(ledoit_wolf_covariance(r)),
    }
    results, latest = {}, {}
    for name, allocator in allocators.items():
        weights = rolling_allocation(returns, allocator, lookback=252, rebalance='M')
        results[name] = backtest(returns, weights, transaction_cost_bps=cost_bps)
        latest[name] = weights.iloc[-1]
    print(f'Latest weights ({returns.index[-1]:%Y-%m-%d}):')
    print(pd.DataFrame(latest).round(3))

    _section('4. STRATEGIES: momentum, trend, low volatility, volatility targeting')
    tsmom = time_series_momentum(prices)
    strategies = {
        'ts_momentum': (tsmom, 'M'),
        'ts_momentum_voltgt': (volatility_scaling(tsmom, prices, target_volatility=0.10), 'M'),
        'xs_momentum_12_1': (cross_sectional_momentum(prices, quantile=0.25), 'M'),
        'ma_crossover_50_200': (moving_average_crossover(prices), 'W'),
        'low_volatility': (low_volatility(prices, quantile=0.25), 'M'),
    }
    for name, (weights, frequency) in strategies.items():
        results[name] = backtest(returns, weights, transaction_cost_bps=cost_bps, rebalance=frequency)
        print(f'{name:>20}: annual turnover {results[name].summary()["annual_turnover"]:.2f}x')

    _section('5. EVALUATION: net-of-cost performance and multiple-testing adjustment')
    start = max(result.weights.abs().sum(axis=1).ne(0).idxmax() for result in results.values())
    net = pd.DataFrame({name: result.returns.loc[start:] for name, result in results.items()})
    table = performance_summary(net)
    print(table.round(3))
    trials = [sharpe_ratio(net[c]) for c in net]
    best = net.columns[int(np.argmax(trials))]
    dsr = deflated_sharpe_ratio(net[best], trials)
    print(
        f'\nBest strategy: {best} (Sharpe {max(trials):.2f}); deflated Sharpe ratio over {len(trials)} trials: '
        f'{dsr:.3f} -> {"significant" if dsr > 0.95 else "NOT significant"} at 95%'
    )

    _section('6. VALUATION: bonds and options')
    bond = bond_analytics(yield_to_maturity=0.045, coupon_rate=0.05, maturity=10, frequency=2)
    print(
        f'10y 5% bond at 4.5% yield: price {bond.price:.3f}, modified duration {bond.modified_duration:.3f}, '
        f'convexity {bond.convexity:.2f}, DV01 {bond.dv01:.4f}'
    )
    option = black_scholes(spot=100, strike=105, expiry=0.5, rate=0.04, volatility=0.25)
    print(
        f'6m 105 call on 100 spot, 25% vol: price {option.price:.3f}, delta {option.delta:.3f}, '
        f'gamma {option.gamma:.4f}, vega {option.vega:.3f}'
    )
    return table


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--csv', help='CSV of prices: first column dates, then one column per asset')
    parser.add_argument('--cost-bps', type=float, default=10.0, help='one-way transaction cost in basis points')
    parser.add_argument('--seed', type=int, default=7, help='random seed for synthetic data')
    args = parser.parse_args(argv)
    prices = pd.read_csv(args.csv, index_col=0, parse_dates=True) if args.csv else synthetic_prices(seed=args.seed)
    run(prices, args.cost_bps)


if __name__ == '__main__':
    main()
