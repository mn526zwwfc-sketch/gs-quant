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

Systematic trading strategies documented in peer-reviewed research.

Every function takes prices (dates x assets) and returns target weights (dates x assets) computed with the data
available at each date's close. Feed them to :func:`gs_quant.research.backtest.backtest`, which delays execution
to avoid look-ahead bias.

* Time-series momentum / trend following (Moskowitz, Ooi & Pedersen, 2012; Hurst, Ooi & Pedersen, 2017).
* Cross-sectional momentum, 12-1 months (Jegadeesh & Titman, 1993; Carhart, 1997).
* Short-term reversal with the contrarian weights of Lo & MacKinlay (1990) (Jegadeesh, 1990; Lehmann, 1990).
* Moving-average trading rules (Brock, Lakonishok & LeBaron, 1992; Faber, 2007).
* Low-volatility anomaly (Ang, Hodrick, Xing & Zhang, 2006; Baker, Bradley & Wurgler, 2011;
  Frazzini & Pedersen, 2014).
* Volatility targeting / volatility-managed portfolios (Moreira & Muir, 2017; Harvey et al., 2018).
* Signal combination across styles (Asness, Moskowitz & Pedersen, 2013).
"""

from typing import Optional

import numpy as np
import pandas as pd

from gs_quant.errors import MqValueError
from gs_quant.research.data_management import cross_sectional_rank, cross_sectional_zscore


def _check_quantile(quantile: float):
    if not 0 < quantile <= 0.5:
        raise MqValueError('quantile must be in (0, 0.5]')


def _normalize_by_count(signal: pd.DataFrame) -> pd.DataFrame:
    """Split capital equally across the assets that have a signal on each date"""
    count = signal.notna().sum(axis=1).replace(0, np.nan)
    return signal.div(count, axis=0)


def _long_short_quantiles(score: pd.DataFrame, quantile: float, long_only: bool, ascending: bool) -> pd.DataFrame:
    """Equal-weight the top quantile long (sum 1) and, unless long only, the bottom quantile short (sum -1)"""
    _check_quantile(quantile)
    pct = score.rank(axis=1, pct=True, ascending=ascending)

    def _equal_weight(selected: pd.DataFrame) -> pd.DataFrame:
        selected = selected.astype(float)
        return selected.div(selected.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    weights = _equal_weight(pct > 1 - quantile)
    if not long_only:
        weights = weights - _equal_weight(pct <= quantile)
    return weights.where(score.notna().any(axis=1), np.nan, axis=0)


def ex_ante_volatility(prices: pd.DataFrame, center_of_mass: float = 60, periods_per_year: float = 261) -> pd.DataFrame:
    """
    Exponentially weighted ex-ante volatility, as in Moskowitz, Ooi & Pedersen (2012)

    :param prices: price levels (dates x assets)
    :param center_of_mass: center of mass of the exponential weights, in observations (60 days in the paper)
    :param periods_per_year: annualization factor (261 in the paper)
    :return: annualized volatility estimate known at each date
    """
    returns = prices / prices.shift(1) - 1
    return np.sqrt(returns.ewm(com=center_of_mass, min_periods=center_of_mass).var() * periods_per_year)


def time_series_momentum(
    prices: pd.DataFrame,
    lookback: int = 252,
    volatility_target: float = 0.40,
    center_of_mass: float = 60,
    periods_per_year: float = 261,
) -> pd.DataFrame:
    """
    Time-series momentum (Moskowitz, Ooi & Pedersen, 2012)

    :param prices: price levels (dates x assets), typically futures or total return indices
    :param lookback: observations used to measure the trend (12 months in the paper)
    :param volatility_target: annualized volatility targeted for each position (40% in the paper)
    :param center_of_mass: center of mass of the ex-ante volatility estimate
    :param periods_per_year: annualization factor of the volatility estimate
    :return: target weights

    **Usage**

    Each asset is held long if its own past return over the lookback is positive and short otherwise, with a size
    inversely proportional to its ex-ante volatility: :math:`w_{i,t} = \\frac{1}{S_t}
    \\mathrm{sign}(r_{i,t-L,t}) \\frac{\\sigma_{tgt}}{\\sigma_{i,t}}`, where :math:`S_t` is the number of assets.
    The strategy has been profitable in 58 liquid futures for over a century (Hurst, Ooi & Pedersen, 2017) and
    tends to perform well in extended market downturns.
    """
    trend = np.sign(prices / prices.shift(lookback) - 1)
    vol = ex_ante_volatility(prices, center_of_mass, periods_per_year)
    raw = trend * volatility_target / vol.replace(0, np.nan)
    return _normalize_by_count(raw)


def cross_sectional_momentum(
    prices: pd.DataFrame, lookback: int = 252, skip: int = 21, quantile: float = 0.2, long_only: bool = False
) -> pd.DataFrame:
    """
    Cross-sectional (relative strength) momentum (Jegadeesh & Titman, 1993)

    :param prices: price levels (dates x assets)
    :param lookback: formation period in observations (12 months)
    :param skip: most recent observations excluded from the formation period (1 month), to avoid the short-term
        reversal effect
    :param quantile: fraction of the universe held on each side
    :param long_only: hold only the winners
    :return: target weights; winners sum to 1 and losers to -1

    **Usage**

    Ranks assets on their return from :math:`t - L` to :math:`t - S` ("12-1 momentum"), buys past winners and sells
    past losers. The premium is one of the most robust anomalies across markets and asset classes (Carhart, 1997;
    Asness, Moskowitz & Pedersen, 2013), but it suffers occasional crashes after market rebounds
    (Daniel & Moskowitz, 2016), which volatility scaling mitigates (Barroso & Santa-Clara, 2015).
    """
    if skip >= lookback:
        raise MqValueError('skip must be shorter than lookback')
    score = prices.shift(skip) / prices.shift(lookback) - 1
    return _long_short_quantiles(score, quantile, long_only, ascending=True)


def short_term_reversal(prices: pd.DataFrame, lookback: int = 21) -> pd.DataFrame:
    """
    Short-term reversal with Lo & MacKinlay (1990) contrarian weights

    :param prices: price levels (dates x assets)
    :param lookback: observations over which past returns are measured (1 month or 1 week in the literature)
    :return: dollar-neutral target weights with long side summing to 1

    **Usage**

    :math:`w_{i,t} \\propto -(r_{i,t} - \\bar{r}_t)`: buy recent losers and sell recent winners, proportionally to
    their deviation from the cross-sectional mean (Jegadeesh, 1990; Lehmann, 1990). Turnover is high, so the
    strategy is very sensitive to transaction costs.
    """
    past = prices / prices.shift(lookback) - 1
    raw = -past.sub(past.mean(axis=1), axis=0)
    gross = raw.abs().sum(axis=1).replace(0, np.nan)
    return raw.div(gross / 2, axis=0)


def moving_average_crossover(
    prices: pd.DataFrame, fast: int = 50, slow: int = 200, long_short: bool = False
) -> pd.DataFrame:
    """
    Moving-average crossover trading rule (Brock, Lakonishok & LeBaron, 1992)

    :param prices: price levels (dates x assets)
    :param fast: window of the short moving average
    :param slow: window of the long moving average
    :param long_short: short the asset (instead of holding cash) when the fast average is below the slow one
    :return: target weights, capital split equally across assets

    **Usage**

    Long when the fast moving average is above the slow one. With ``fast=1`` and ``slow=200`` this is the
    price-above-200-day-average rule popularized by Faber (2007), which historically reduced drawdowns.
    """
    if fast >= slow:
        raise MqValueError('fast window must be shorter than slow window')
    fast_ma = prices.rolling(fast, min_periods=fast).mean()
    slow_ma = prices.rolling(slow, min_periods=slow).mean()
    signal = (fast_ma > slow_ma).astype(float)
    if long_short:
        signal = signal * 2 - 1
    return _normalize_by_count(signal.where(slow_ma.notna()))


def low_volatility(
    prices: pd.DataFrame, lookback: int = 252, quantile: float = 0.2, long_only: bool = True
) -> pd.DataFrame:
    """
    Low-volatility anomaly (Ang, Hodrick, Xing & Zhang, 2006; Blitz & van Vliet, 2007)

    :param prices: price levels (dates x assets)
    :param lookback: observations used to estimate volatility
    :param quantile: fraction of the universe held on each side
    :param long_only: hold only the low-volatility assets; otherwise also short the high-volatility ones
    :return: target weights

    **Usage**

    Low-risk stocks have historically earned higher risk-adjusted returns than high-risk stocks, contrary to the
    CAPM. Leading explanations are leverage constraints (Frazzini & Pedersen, 2014) and benchmarking of
    institutional investors (Baker, Bradley & Wurgler, 2011).
    """
    returns = prices / prices.shift(1) - 1
    vol = returns.rolling(lookback, min_periods=lookback).std()
    return _long_short_quantiles(vol, quantile, long_only, ascending=False)


def volatility_scaling(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    target_volatility: float = 0.10,
    lookback: int = 63,
    max_leverage: float = 2.0,
    periods_per_year: float = 252,
) -> pd.DataFrame:
    """
    Scale a strategy to a constant volatility target (Moreira & Muir, 2017; Harvey et al., 2018)

    :param weights: target weights of the unscaled strategy, on the same (e.g. daily) dates as the prices
    :param prices: price levels of the assets
    :param target_volatility: annualized volatility target
    :param lookback: observations used to estimate the strategy's realized volatility
    :param max_leverage: cap on the scaling factor
    :param periods_per_year: annualization factor
    :return: scaled target weights

    **Usage**

    Volatility clusters while expected returns do not rise proportionally, so reducing exposure when recent
    volatility is high improves Sharpe ratios and cuts tail risk, particularly for equity and momentum strategies
    (Moreira & Muir, 2017; Barroso & Santa-Clara, 2015). The realized volatility of the strategy is estimated with
    the weights held on each date, so the scaling factor at :math:`t` only uses data up to :math:`t`.
    """
    aligned = prices.reindex(index=weights.index, columns=weights.columns)
    returns = aligned / aligned.shift(1) - 1
    strategy = (weights.shift(1) * returns).sum(axis=1, min_count=1)
    realized = strategy.rolling(lookback, min_periods=lookback // 2).std() * np.sqrt(periods_per_year)
    scale = (target_volatility / realized.replace(0, np.nan)).clip(upper=max_leverage)
    return weights.mul(scale, axis=0)


def combine_signals(signals: dict[str, pd.DataFrame], weights: Optional[dict[str, float]] = None) -> pd.DataFrame:
    """
    Combine signals from several styles into a composite score

    :param signals: signal name to scores (dates x assets); higher means more attractive
    :param weights: relative weight of each signal; equal weights when omitted
    :return: weighted average of the cross-sectionally ranked signals, standardized on each date

    **Usage**

    Styles such as value and momentum are negatively correlated, so their combination is far more stable than
    either alone (Asness, Moskowitz & Pedersen, 2013, "Value and Momentum Everywhere", Journal of Finance 68(3)).
    Rank transformation makes signals with different units comparable and robust to outliers.
    """
    if not signals:
        raise MqValueError('at least one signal is required')
    weights = weights or {name: 1.0 for name in signals}
    total = sum(weights[name] for name in signals)
    combined = sum(cross_sectional_rank(signal).fillna(0) * weights[name] / total for name, signal in signals.items())
    available = sum(signal.notna().astype(int) for signal in signals.values()) > 0
    combined = combined.where(available)
    # dates where all assets score the same carry no preference: zero rather than undefined
    return cross_sectional_zscore(combined).fillna(0.0).where(combined.notna())
