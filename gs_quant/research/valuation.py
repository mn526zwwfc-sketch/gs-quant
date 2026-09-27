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

Valuation of securities with textbook closed-form models.

* Fixed income: price, yield, Macaulay and modified duration, convexity and DV01 of fixed-coupon bonds
  (Macaulay, 1938; Fabozzi, 2012).
* Options: Black-Scholes-Merton price and Greeks with a continuous dividend yield (Black & Scholes, 1973;
  Merton, 1973) and implied volatility (Hull, 2018).
* Equities: dividend discount model (Gordon, 1962), discounted cash flows with a terminal value and the CAPM
  cost of equity (Sharpe, 1964; Lintner, 1965).

These are local, dependency-free building blocks. For production pricing with market data, use the instruments
and risk measures in :mod:`gs_quant.instrument` and :mod:`gs_quant.risk`.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

import numpy as np
from scipy import optimize, stats

from gs_quant.errors import MqValueError


class OptionType(Enum):
    CALL = 'call'
    PUT = 'put'


@dataclass(frozen=True)
class BondAnalytics:
    """
    Price and interest rate sensitivities of a bond, per 100 of face value when ``face=100``
    """

    price: float
    macaulay_duration: float
    modified_duration: float
    convexity: float
    dv01: float


@dataclass(frozen=True)
class OptionAnalytics:
    """
    Black-Scholes-Merton price and Greeks

    ``vega`` and ``rho`` are per unit change (multiply by 0.01 for a one point move) and ``theta`` is per year.
    """

    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def bond_cash_flows(
    coupon_rate: float, maturity: float, face: float = 100.0, frequency: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cash flow schedule of a fixed-coupon bullet bond

    :param coupon_rate: annual coupon rate, e.g. 0.05
    :param maturity: time to maturity in years; the first coupon period may be shorter than the others
    :param face: face value repaid at maturity
    :param frequency: coupons per year
    :return: tuple of payment times in years and payment amounts
    """
    if maturity <= 0 or frequency <= 0:
        raise MqValueError('maturity and frequency must be positive')
    n = int(np.ceil(maturity * frequency - 1e-9))
    times = maturity - np.arange(n - 1, -1, -1) / frequency
    flows = np.full(n, face * coupon_rate / frequency)
    flows[-1] += face
    return times, flows


def bond_price(
    yield_to_maturity: float, coupon_rate: float, maturity: float, face: float = 100.0, frequency: int = 2
) -> float:
    """
    Price of a fixed-coupon bond from its yield

    :param yield_to_maturity: annual yield compounded ``frequency`` times per year
    :param coupon_rate: annual coupon rate
    :param maturity: time to maturity in years
    :param face: face value
    :param frequency: coupons (and compounding periods) per year
    :return: full (dirty) price :math:`\\sum_k CF_k (1 + y/f)^{-f t_k}`
    """
    times, flows = bond_cash_flows(coupon_rate, maturity, face, frequency)
    return float(np.sum(flows * (1 + yield_to_maturity / frequency) ** (-frequency * times)))


def bond_yield(price: float, coupon_rate: float, maturity: float, face: float = 100.0, frequency: int = 2) -> float:
    """
    Yield to maturity implied by a bond price

    :param price: full (dirty) price
    :param coupon_rate: annual coupon rate
    :param maturity: time to maturity in years
    :param face: face value
    :param frequency: coupons (and compounding periods) per year
    :return: annual yield compounded ``frequency`` times per year
    """
    if price <= 0:
        raise MqValueError('price must be positive')
    lower = -0.99 * frequency
    return float(
        optimize.brentq(
            lambda y: bond_price(y, coupon_rate, maturity, face, frequency) - price, lower, 10.0, xtol=1e-14
        )
    )


def bond_analytics(
    yield_to_maturity: float, coupon_rate: float, maturity: float, face: float = 100.0, frequency: int = 2
) -> BondAnalytics:
    """
    Price, duration, convexity and DV01 of a fixed-coupon bond

    :param yield_to_maturity: annual yield compounded ``frequency`` times per year
    :param coupon_rate: annual coupon rate
    :param maturity: time to maturity in years
    :param face: face value
    :param frequency: coupons (and compounding periods) per year
    :return: :class:`BondAnalytics`

    **Usage**

    The price change for a yield move :math:`\\Delta y` is approximated by
    :math:`\\Delta P / P \\approx -D_{mod} \\Delta y + \\frac{1}{2} C \\Delta y^2` (Fabozzi, 2012).
    """
    times, flows = bond_cash_flows(coupon_rate, maturity, face, frequency)
    base = 1 + yield_to_maturity / frequency
    pv = flows * base ** (-frequency * times)
    price = float(pv.sum())
    macaulay = float(np.sum(times * pv) / price)
    modified = macaulay / base
    convexity = float(np.sum(pv * times * (times + 1 / frequency)) / (price * base**2))
    return BondAnalytics(price, macaulay, modified, convexity, modified * price * 1e-4)


def _check_option_inputs(spot: float, strike: float, expiry: float, volatility: float):
    if spot <= 0 or strike <= 0:
        raise MqValueError('spot and strike must be positive')
    if expiry <= 0 or volatility <= 0:
        raise MqValueError('expiry and volatility must be positive')


def black_scholes(
    spot: float,
    strike: float,
    expiry: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    option_type: OptionType = OptionType.CALL,
) -> OptionAnalytics:
    """
    Black-Scholes-Merton price and Greeks of a European option

    :param spot: price of the underlying
    :param strike: strike price
    :param expiry: time to expiry in years
    :param rate: continuously compounded risk-free rate
    :param volatility: annualized volatility of the underlying
    :param dividend_yield: continuously compounded dividend yield (Merton, 1973)
    :param option_type: call or put
    :return: :class:`OptionAnalytics`
    """
    _check_option_inputs(spot, strike, expiry, volatility)
    sqrt_t = np.sqrt(expiry)
    d1 = (np.log(spot / strike) + (rate - dividend_yield + 0.5 * volatility**2) * expiry) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    df_r, df_q = np.exp(-rate * expiry), np.exp(-dividend_yield * expiry)
    pdf_d1 = stats.norm.pdf(d1)
    gamma = df_q * pdf_d1 / (spot * volatility * sqrt_t)
    vega = spot * df_q * pdf_d1 * sqrt_t
    decay = -spot * df_q * pdf_d1 * volatility / (2 * sqrt_t)
    if OptionType(option_type) == OptionType.CALL:
        n_d1, n_d2 = stats.norm.cdf(d1), stats.norm.cdf(d2)
        price = spot * df_q * n_d1 - strike * df_r * n_d2
        delta = df_q * n_d1
        theta = decay - rate * strike * df_r * n_d2 + dividend_yield * spot * df_q * n_d1
        rho = strike * expiry * df_r * n_d2
    else:
        n_d1, n_d2 = stats.norm.cdf(-d1), stats.norm.cdf(-d2)
        price = strike * df_r * n_d2 - spot * df_q * n_d1
        delta = -df_q * n_d1
        theta = decay + rate * strike * df_r * n_d2 - dividend_yield * spot * df_q * n_d1
        rho = -strike * expiry * df_r * n_d2
    return OptionAnalytics(*(float(v) for v in (price, delta, gamma, vega, theta, rho)))


def implied_volatility(
    price: float,
    spot: float,
    strike: float,
    expiry: float,
    rate: float,
    dividend_yield: float = 0.0,
    option_type: OptionType = OptionType.CALL,
) -> float:
    """
    Black-Scholes-Merton implied volatility

    :param price: observed option premium
    :param spot: price of the underlying
    :param strike: strike price
    :param expiry: time to expiry in years
    :param rate: continuously compounded risk-free rate
    :param dividend_yield: continuously compounded dividend yield
    :param option_type: call or put
    :return: the volatility that reproduces the observed premium
    """
    _check_option_inputs(spot, strike, expiry, 1.0)
    forward_spot, pv_strike = spot * np.exp(-dividend_yield * expiry), strike * np.exp(-rate * expiry)
    if OptionType(option_type) == OptionType.CALL:
        lower, upper = max(forward_spot - pv_strike, 0.0), forward_spot
    else:
        lower, upper = max(pv_strike - forward_spot, 0.0), pv_strike
    if not lower < price < upper:
        raise MqValueError(f'price {price} is outside the no-arbitrage bounds ({lower:.6f}, {upper:.6f})')

    def objective(vol: float) -> float:
        return black_scholes(spot, strike, expiry, rate, vol, dividend_yield, option_type).price - price

    return float(optimize.brentq(objective, 1e-6, 10.0, xtol=1e-12))


def gordon_growth_value(next_dividend: float, required_return: float, growth: float) -> float:
    """
    Value of a stock whose dividends grow at a constant rate (Gordon, 1962)

    :param next_dividend: dividend expected over the next year
    :param required_return: cost of equity, e.g. from :func:`capm_expected_return`
    :param growth: perpetual dividend growth rate, which must be lower than the required return
    :return: :math:`D_1 / (k - g)`
    """
    if required_return <= growth:
        raise MqValueError('the required return must exceed the growth rate')
    return next_dividend / (required_return - growth)


def discounted_cash_flow(
    cash_flows: Sequence[float], discount_rate: float, terminal_growth: Optional[float] = None
) -> float:
    """
    Present value of annual cash flows with an optional Gordon growth terminal value

    :param cash_flows: cash flows at the end of years 1, 2, ..., n
    :param discount_rate: annual discount rate, e.g. the weighted average cost of capital
    :param terminal_growth: perpetual growth rate after year n; no terminal value when omitted
    :return: :math:`\\sum_t CF_t / (1 + k)^t + TV_n / (1 + k)^n` with :math:`TV_n = CF_n (1 + g) / (k - g)`
    """
    flows = np.asarray(cash_flows, dtype=float)
    if flows.size == 0:
        raise MqValueError('at least one cash flow is required')
    discount = (1 + discount_rate) ** -np.arange(1, flows.size + 1)
    value = float(np.sum(flows * discount))
    if terminal_growth is not None:
        value += gordon_growth_value(flows[-1] * (1 + terminal_growth), discount_rate, terminal_growth) * discount[-1]
    return value


def capm_expected_return(beta: float, risk_free: float, market_return: float) -> float:
    """
    Expected return under the Capital Asset Pricing Model (Sharpe, 1964; Lintner, 1965)

    :param beta: sensitivity of the asset to the market portfolio
    :param risk_free: risk-free rate
    :param market_return: expected return of the market portfolio
    :return: :math:`r_f + \\beta (E[r_m] - r_f)`
    """
    return risk_free + beta * (market_return - risk_free)
