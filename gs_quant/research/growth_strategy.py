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

Long-term "quality growth at a reasonable price" strategy.

Buys fast-growing companies whose growth is profitable, well financed and not yet priced in, combining four
families of evidence from the academic literature:

* Growth: sustained revenue and earnings growth, with stability of growth (Chan, Karceski & Lakonishok, 2003;
  Mohanram, 2005).
* Quality: returns on capital above the cost of capital, gross profitability, cash-backed earnings and strong
  fundamentals (Novy-Marx, 2013; Sloan, 1996; Piotroski, 2000; Mohanram, 2005; Asness, Frazzini & Pedersen, 2019).
* Valuation: a margin of safety relative to a probability-weighted growth DCF, a market-implied growth below the
  realistic forecast, and the residual-income value-to-price ratio (Graham & Dodd, 1934; Rappaport & Mauboussin,
  2001; Frankel & Lee, 1998). Avoids overpaying for glamour stocks (Lakonishok, Shleifer & Vishny, 1994).
* Momentum: 12-1 month price momentum to avoid value traps and time entries (Jegadeesh & Titman, 1993;
  Asness, Moskowitz & Pedersen, 2013).

Portfolio rules follow the evidence on implementation: point-in-time data, a holding buffer that cuts turnover
(Novy-Marx & Velikov, 2016), risk-based weights with a position cap, and optional sector limits. Holdings can be
exported as a :class:`gs_quant.markets.position_set.PositionSet` to be priced, analysed or turned into a basket on
Marquee.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from gs_quant.errors import MqValueError
from gs_quant.research.backtest import BacktestResult, backtest, rebalance_dates, weights_frame
from gs_quant.research.fundamentals import compute_fundamental_metrics, mohanram_g_score, point_in_time_metrics
from gs_quant.research.growth_valuation import ValuationAssumptions, value_company
from gs_quant.research.portfolio import equal_weight, hierarchical_risk_parity, inverse_volatility, risk_parity
from gs_quant.research.risk import ledoit_wolf_covariance

# metric -> +1 if higher is better, -1 if lower is better
SCORE_COMPONENTS = {
    'growth': {'revenue_cagr_3y': 1, 'revenue_growth': 1, 'eps_growth': 1, 'growth_volatility': -1},
    'quality': {
        'roic_spread': 1,
        'gross_profitability': 1,
        'fcf_margin': 1,
        'accruals': -1,
        'f_score': 1,
        'g_score': 1,
    },
    'valuation': {'margin_of_safety': 1, 'expectations_gap': 1, 'rim_to_price': 1},
    'momentum': {'momentum_12_1': 1},
}

WEIGHTING_METHODS = ('equal', 'inverse_volatility', 'risk_parity', 'hrp')


@dataclass
class GrowthStrategyConfig:
    """
    Parameters of :class:`LongTermGrowthStrategy`

    :param n_holdings: number of companies held
    :param buffer: hysteresis for existing holdings, which are kept while they rank within ``n_holdings * buffer``,
        stay within the fastest-growing ``growth_quantile * buffer`` of the universe and keep an F-score of at least
        ``min_f_score - 1`` (when ``buffer > 1``). Wider exit than entry rules cut turnover and trading costs
        (Novy-Marx & Velikov, 2016)
    :param rebalance: pandas period alias of the rebalance frequency ('Q', 'M', 'Y', ...)
    :param min_history_years: minimum number of annual reports
    :param min_market_cap: minimum market capitalization, in the currency of the prices
    :param growth_quantile: only the fastest-growing fraction of the universe (by 3-year revenue CAGR) is eligible
    :param min_revenue_growth: minimum 3-year revenue CAGR
    :param min_f_score: minimum Piotroski F-score
    :param require_value_creation: require ROIC above WACC
    :param min_margin_of_safety: minimum expected value / price - 1 to buy
    :param hold_margin_of_safety: holdings are sold once the margin of safety falls below this level, e.g. -0.10
        sells when the price exceeds the estimated value by about 11%
    :param max_per_sector: maximum number of holdings per sector (requires sectors)
    :param score_weights: weight of each score family in the composite score
    :param weighting: 'equal', 'inverse_volatility', 'risk_parity' or 'hrp'
    :param max_weight: cap on each position; when fewer than ``1 / max_weight`` companies qualify, the remainder is
        held in cash rather than concentrated in the few that do
    :param risk_lookback: trading days used for betas, momentum and covariances
    :param extra_lag_days: safety delay added to filing dates
    :param valuation: assumptions of the valuation model
    """

    n_holdings: int = 25
    buffer: float = 1.5
    rebalance: str = 'Q'
    min_history_years: int = 3
    min_market_cap: float = 0.0
    growth_quantile: float = 0.5
    min_revenue_growth: float = 0.05
    min_f_score: int = 5
    require_value_creation: bool = True
    min_margin_of_safety: float = 0.0
    hold_margin_of_safety: float = -0.10
    max_per_sector: Optional[int] = None
    score_weights: dict[str, float] = field(
        default_factory=lambda: {'growth': 0.25, 'quality': 0.30, 'valuation': 0.30, 'momentum': 0.15}
    )
    weighting: str = 'risk_parity'
    max_weight: float = 0.10
    risk_lookback: int = 252
    extra_lag_days: int = 0
    valuation: ValuationAssumptions = field(default_factory=ValuationAssumptions)

    def __post_init__(self):
        if self.weighting not in WEIGHTING_METHODS:
            raise MqValueError(f'weighting must be one of {WEIGHTING_METHODS}')
        if unknown := set(self.score_weights) - set(SCORE_COMPONENTS):
            raise MqValueError(f'unknown score families: {sorted(unknown)}')
        if self.n_holdings < 1 or self.buffer < 1:
            raise MqValueError('n_holdings must be positive and buffer at least 1')
        if not 0 < self.growth_quantile <= 1:
            raise MqValueError('growth_quantile must be in (0, 1]')
        if self.hold_margin_of_safety > self.min_margin_of_safety:
            raise MqValueError('hold_margin_of_safety cannot exceed min_margin_of_safety')


@dataclass
class GrowthStrategyResult:
    """
    Output of :meth:`LongTermGrowthStrategy.run`: target weights and the full screen on each rebalance date
    """

    weights: pd.DataFrame
    screens: dict[pd.Timestamp, pd.DataFrame]

    def holdings(self, date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        Selected companies on a rebalance date (the latest by default), with their weights and key statistics

        :param date: rebalance date
        :return: one row per holding, sorted by weight
        """
        date = self.weights.index[-1] if date is None else pd.Timestamp(date)
        screen = self.screens[date]
        held = self.weights.loc[date]
        held = held[held > 0]
        table = screen.loc[held.index].assign(weight=held)
        return table.sort_values('weight', ascending=False)


def _centered_rank(values: pd.Series) -> pd.Series:
    valid = values.replace([np.inf, -np.inf], np.nan)
    n = valid.notna().sum()
    if n < 2:
        return pd.Series(0.0, index=values.index).where(valid.notna())
    return (valid.rank() - 1) / (n - 1) - 0.5


def _family_score(table: pd.DataFrame, components: dict[str, int]) -> pd.Series:
    ranks = pd.DataFrame({metric: _centered_rank(table[metric] * sign) for metric, sign in components.items()})
    return ranks.mean(axis=1)


def _cap_weights(weights: pd.Series, cap: float) -> pd.Series:
    """
    Cap weights at ``cap`` and redistribute the excess proportionally to uncapped names. When there are fewer than
    ``1 / cap`` names every position is capped and the remainder is left in cash
    """
    weights = weights / weights.sum()
    if cap * len(weights) <= 1:
        return pd.Series(cap, index=weights.index)
    for _ in range(len(weights)):
        over = weights > cap + 1e-12
        if not over.any():
            break
        excess = (weights[over] - cap).sum()
        weights[over] = cap
        free = ~over & (weights < cap)
        weights[free] += excess * weights[free] / weights[free].sum()
    return weights


class LongTermGrowthStrategy:
    """
    Quality growth at a reasonable price, rebalanced periodically on point-in-time fundamentals

    :param fundamentals: annual fundamentals (see :func:`gs_quant.research.fundamentals.prepare_fundamentals`)
    :param prices: daily share prices (dates x tickers), adjusted for splits and dividends
    :param config: :class:`GrowthStrategyConfig`
    :param sectors: optional sector of each ticker, used for G-score peer medians and sector limits
    :param market_prices: optional market index level used for betas; equal-weighted universe when omitted

    **Examples**

    >>> strategy = LongTermGrowthStrategy(fundamentals, prices, GrowthStrategyConfig(n_holdings=20))
    >>> result = strategy.run()
    >>> result.holdings()                       # current portfolio with valuation and scores
    >>> strategy.backtest(result).summary()     # net-of-cost historical simulation
    >>> strategy.to_position_set(result)        # gs_quant.markets PositionSet for Marquee
    """

    def __init__(
        self,
        fundamentals: pd.DataFrame,
        prices: pd.DataFrame,
        config: Optional[GrowthStrategyConfig] = None,
        sectors: Optional[pd.Series] = None,
        market_prices: Optional[pd.Series] = None,
    ):
        self.config = config or GrowthStrategyConfig()
        self.metrics = compute_fundamental_metrics(fundamentals)
        self.prices = prices.sort_index()
        self.returns = (self.prices / self.prices.shift(1) - 1).iloc[1:]
        self.sectors = sectors
        market = market_prices.reindex(self.prices.index) if market_prices is not None else None
        self.market_returns = (
            (market / market.shift(1) - 1).iloc[1:] if market is not None else self.returns.mean(axis=1)
        )

    def _market_data(self, date: pd.Timestamp) -> pd.DataFrame:
        lookback = self.config.risk_lookback
        window = self.returns.loc[:date].tail(lookback)
        market = self.market_returns.reindex(window.index)
        valid = window.notna().sum() >= lookback // 2
        centered = window.sub(window.mean())
        m = market - market.mean()
        beta = (centered.mul(m, axis=0).sum() / (m.pow(2).sum() or np.nan)).where(valid)
        history = self.prices.loc[:date]
        # 12-1 momentum: return from one year ago to one month ago
        momentum = history.iloc[-22] / history.iloc[-1 - lookback] - 1 if len(history) > lookback else np.nan
        return pd.DataFrame({'price': history.iloc[-1], 'beta': beta, 'momentum_12_1': momentum})

    def screen(self, date: pd.Timestamp) -> pd.DataFrame:
        """
        Evaluate the whole universe on a date: metrics, valuation, scores and eligibility

        :param date: evaluation date; only information published up to this date is used
        :return: one row per company, sorted by composite score
        """
        c = self.config
        date = pd.Timestamp(date)
        known = point_in_time_metrics(self.metrics, [date], extra_lag_days=c.extra_lag_days)
        if known.empty:
            return pd.DataFrame()
        table = known.xs(date, level='date').join(self._market_data(date), how='inner')
        table['market_cap'] = table['price'] * table['shares_outstanding']
        groups = self.sectors.reindex(table.index) if self.sectors is not None else None
        table['g_score'] = mohanram_g_score(table, groups)

        def fundamentals_pass(growth_quantile: float, min_f_score: int) -> pd.Series:
            growth_cut = table['revenue_cagr_3y'].quantile(1 - min(growth_quantile, 1.0))
            return (
                (table['history_years'] >= c.min_history_years)
                & (table['market_cap'] >= c.min_market_cap)
                & (table['revenue_cagr_3y'] >= max(growth_cut, c.min_revenue_growth))
                & (table['operating_margin'] > 0)
                & (table['f_score'] >= min_f_score)
                & table['price'].gt(0)
            )

        entry = fundamentals_pass(c.growth_quantile, c.min_f_score)
        hold = fundamentals_pass(c.growth_quantile * c.buffer, c.min_f_score - (1 if c.buffer > 1 else 0))
        valuations = {}
        for ticker in table.index[hold]:
            row = table.loc[ticker]
            beta = row['beta'] if np.isfinite(row['beta']) else 1.0
            valuations[ticker] = value_company(row, row['price'], beta, c.valuation).to_series()
        valuation = pd.DataFrame.from_dict(valuations, orient='index').drop(columns='price', errors='ignore')
        table = table.join(valuation, how='left') if not valuation.empty else table.assign(margin_of_safety=np.nan)
        for column in ('wacc', 'expectations_gap', 'residual_income_value', 'expected_value'):
            if column not in table:
                table[column] = np.nan
        table['roic_spread'] = table['roic'] - table['wacc']
        table['rim_to_price'] = table['residual_income_value'] / table['price']

        creates_value = table['roic_spread'] > 0 if c.require_value_creation else pd.Series(True, index=table.index)
        table['eligible'] = entry & (table['margin_of_safety'] >= c.min_margin_of_safety) & creates_value
        table['holdable'] = hold & (table['margin_of_safety'] >= c.hold_margin_of_safety) & creates_value

        # scores and ranks are computed among all holdable companies, a superset of the eligible ones
        candidates = table[table['holdable']]
        scores = pd.DataFrame(index=table.index)
        for family in c.score_weights:
            scores[f'{family}_score'] = _family_score(candidates, SCORE_COMPONENTS[family])
        total = sum(c.score_weights.values())
        table = table.join(scores)
        table['composite_score'] = sum(
            table[f'{family}_score'].fillna(0) * weight / total for family, weight in c.score_weights.items()
        ).where(table['holdable'])
        table['rank'] = table['composite_score'].rank(ascending=False, method='first')
        return table.sort_values(['composite_score', 'revenue_cagr_3y'], ascending=[False, False])

    def _select(self, table: pd.DataFrame, previous: list[str]) -> list[str]:
        c = self.config
        holdable = table[table['holdable']]
        kept = [t for t in previous if t in holdable.index and holdable.loc[t, 'rank'] <= c.n_holdings * c.buffer]
        kept = sorted(kept, key=lambda t: holdable.loc[t, 'rank'])
        entries = table[table['eligible']].sort_values('rank').index
        selected, per_sector = [], {}

        def add(ticker: str) -> bool:
            sector = self.sectors.get(ticker) if self.sectors is not None else None
            if c.max_per_sector and sector is not None and per_sector.get(sector, 0) >= c.max_per_sector:
                return False
            selected.append(ticker)
            per_sector[sector] = per_sector.get(sector, 0) + 1
            return True

        for ticker in kept:
            if len(selected) < c.n_holdings:
                add(ticker)
        for ticker in entries:
            if len(selected) >= c.n_holdings:
                break
            if ticker not in selected:
                add(ticker)
        return selected

    def _weights(self, selected: list[str], date: pd.Timestamp) -> pd.Series:
        c = self.config
        if not selected:
            return pd.Series(dtype=float)
        window = self.returns.loc[:date, selected].tail(c.risk_lookback).dropna(axis=1, thresh=c.risk_lookback // 2)
        window = window.dropna()
        if c.weighting == 'equal' or window.shape[1] < len(selected) or len(window) < 20:
            weights = equal_weight(selected)
        else:
            cov = ledoit_wolf_covariance(window)
            allocate = {'inverse_volatility': inverse_volatility, 'risk_parity': risk_parity}
            weights = allocate.get(c.weighting, hierarchical_risk_parity)(cov)
        return _cap_weights(weights.astype(float), c.max_weight)

    def run(self, start: Optional[str] = None, end: Optional[str] = None) -> GrowthStrategyResult:
        """
        Run the screen, selection and weighting on every rebalance date

        :param start: first rebalance date considered; defaults to one risk lookback after the first price
        :param end: last rebalance date considered
        :return: :class:`GrowthStrategyResult`
        """
        c = self.config
        index = self.prices.index
        first = index[min(c.risk_lookback, len(index) - 1)]
        dates = rebalance_dates(index[index >= max(first, pd.Timestamp(start) if start else first)], c.rebalance)
        if end is not None:
            dates = dates[dates <= pd.Timestamp(end)]
        screens, rows, previous = {}, {}, []
        for date in dates:
            table = self.screen(date)
            if table.empty:
                continue
            selected = self._select(table, previous)
            weights = self._weights(selected, date)
            table['selected'] = table.index.isin(selected)
            screens[date] = table
            rows[date] = weights
            previous = selected
        if not rows:
            raise MqValueError('no rebalance date had enough fundamentals and price history')
        weights = weights_frame(rows, self.prices.columns)
        return GrowthStrategyResult(weights=weights, screens=screens)

    def backtest(self, result: GrowthStrategyResult, transaction_cost_bps: float = 20.0) -> BacktestResult:
        """
        Net-of-cost simulation of the strategy's target weights

        :param result: output of :meth:`run`
        :param transaction_cost_bps: one-way transaction cost
        :return: :class:`gs_quant.research.backtest.BacktestResult`
        """
        return backtest(self.returns, result.weights, transaction_cost_bps=transaction_cost_bps)

    @staticmethod
    def to_position_set(result: GrowthStrategyResult, date: Optional[pd.Timestamp] = None):
        """
        Export holdings as a :class:`gs_quant.markets.position_set.PositionSet`

        :param result: output of :meth:`run`
        :param date: rebalance date; the latest by default
        :return: position set with weights and tags (composite score, margin of safety, implied growth)

        **Usage**

        With a Marquee session the position set can be resolved to assets, priced, analysed with risk models or
        used to create a basket (see :mod:`gs_quant.markets.position_set` and :mod:`gs_quant.markets.baskets`).
        """
        from gs_quant.markets.position_set import Position, PositionSet

        holdings = result.holdings(date)
        as_of = (result.weights.index[-1] if date is None else pd.Timestamp(date)).date()
        positions = [
            Position(
                identifier=ticker,
                weight=float(row['weight']),
                tags=[
                    {'composite_score': f'{row["composite_score"]:.4f}'},
                    {'margin_of_safety': f'{row["margin_of_safety"]:.4f}'},
                    {'implied_growth': f'{row["implied_growth"]:.4f}'},
                ],
            )
            for ticker, row in holdings.iterrows()
        ]
        return PositionSet(positions=positions, date=as_of)
