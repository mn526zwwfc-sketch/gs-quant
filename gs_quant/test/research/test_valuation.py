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
import pytest

from gs_quant.errors import MqValueError
from gs_quant.research import (
    OptionType,
    black_scholes,
    bond_analytics,
    bond_cash_flows,
    bond_price,
    bond_yield,
    capm_expected_return,
    discounted_cash_flow,
    gordon_growth_value,
    implied_volatility,
)


def test_bond_cash_flows():
    times, flows = bond_cash_flows(0.06, 2, 100, 2)
    assert np.allclose(times, [0.5, 1, 1.5, 2])
    assert np.allclose(flows, [3, 3, 3, 103])
    times, _ = bond_cash_flows(0.06, 1.75, 100, 2)
    assert np.allclose(times, [0.25, 0.75, 1.25, 1.75])
    with pytest.raises(MqValueError):
        bond_cash_flows(0.05, 0, 100, 2)


def test_bond_price_par_and_zero():
    assert np.isclose(bond_price(0.05, 0.05, 10), 100)
    assert np.isclose(bond_price(0.04, 0.0, 5, frequency=1), 100 / 1.04**5)
    assert bond_price(0.06, 0.05, 10) < 100 < bond_price(0.04, 0.05, 10)


def test_bond_yield_round_trip():
    for y in (-0.005, 0.0, 0.03, 0.12):
        price = bond_price(y, 0.04, 7.5, frequency=4)
        assert np.isclose(bond_yield(price, 0.04, 7.5, frequency=4), y, atol=1e-10)
    with pytest.raises(MqValueError):
        bond_yield(-1, 0.04, 5)


def test_bond_analytics_matches_finite_differences():
    y, h = 0.045, 1e-5
    result = bond_analytics(y, 0.05, 10)
    up, down = bond_price(y + h, 0.05, 10), bond_price(y - h, 0.05, 10)
    assert np.isclose(result.price, bond_price(y, 0.05, 10))
    assert np.isclose(result.modified_duration, -(up - down) / (2 * h) / result.price, rtol=1e-6)
    assert np.isclose(result.convexity, (up + down - 2 * result.price) / h**2 / result.price, rtol=1e-4)
    assert np.isclose(result.macaulay_duration, result.modified_duration * (1 + y / 2))
    assert np.isclose(result.dv01, result.modified_duration * result.price * 1e-4)


def test_zero_coupon_duration_equals_maturity():
    result = bond_analytics(0.03, 0.0, 8, frequency=1)
    assert np.isclose(result.macaulay_duration, 8)
    assert np.isclose(result.convexity, 8 * 9 / 1.03**2)


def test_black_scholes_reference_values():
    # textbook example from Hull, "Options, Futures and Other Derivatives": S=42, K=40, r=10%, sigma=20%, T=0.5
    call = black_scholes(42, 40, 0.5, 0.10, 0.20)
    put = black_scholes(42, 40, 0.5, 0.10, 0.20, option_type=OptionType.PUT)
    assert np.isclose(call.price, 4.76, atol=5e-3)
    assert np.isclose(put.price, 0.81, atol=5e-3)
    assert np.isclose(black_scholes(100, 100, 1, 0.05, 0.2).price, 10.450583572185565)


def test_put_call_parity_with_dividends():
    s, k, t, r, v, q = 100, 95, 0.75, 0.03, 0.3, 0.02
    call = black_scholes(s, k, t, r, v, q)
    put = black_scholes(s, k, t, r, v, q, 'put')
    assert np.isclose(call.price - put.price, s * np.exp(-q * t) - k * np.exp(-r * t))
    assert np.isclose(call.delta - put.delta, np.exp(-q * t))
    assert np.isclose(call.gamma, put.gamma) and np.isclose(call.vega, put.vega)


@pytest.mark.parametrize('option_type', [OptionType.CALL, OptionType.PUT])
def test_greeks_match_finite_differences(option_type):
    base = dict(spot=100.0, strike=110.0, expiry=0.8, rate=0.04, volatility=0.25, dividend_yield=0.01)

    def price(**bump):
        return black_scholes(**{**base, **bump}, option_type=option_type).price

    greeks = black_scholes(**base, option_type=option_type)
    h = 1e-4
    assert np.isclose(greeks.delta, (price(spot=100 + h) - price(spot=100 - h)) / (2 * h), rtol=1e-6)
    assert np.isclose(greeks.gamma, (price(spot=100 + 1e-2) + price(spot=100 - 1e-2) - 2 * greeks.price) / 1e-4, 1e-4)
    assert np.isclose(greeks.vega, (price(volatility=0.25 + h) - price(volatility=0.25 - h)) / (2 * h), rtol=1e-6)
    assert np.isclose(greeks.theta, -(price(expiry=0.8 + h) - price(expiry=0.8 - h)) / (2 * h), rtol=1e-5)
    assert np.isclose(greeks.rho, (price(rate=0.04 + h) - price(rate=0.04 - h)) / (2 * h), rtol=1e-6)


def test_black_scholes_input_validation():
    with pytest.raises(MqValueError):
        black_scholes(100, 100, 0, 0.05, 0.2)
    with pytest.raises(MqValueError):
        black_scholes(-1, 100, 1, 0.05, 0.2)


@pytest.mark.parametrize('option_type', ['call', 'put'])
def test_implied_volatility_round_trip(option_type):
    price = black_scholes(100, 90, 0.5, 0.02, 0.37, 0.01, option_type).price
    assert np.isclose(implied_volatility(price, 100, 90, 0.5, 0.02, 0.01, option_type), 0.37, atol=1e-8)


def test_implied_volatility_arbitrage_bounds():
    with pytest.raises(MqValueError):
        implied_volatility(0.5, 100, 90, 0.5, 0.02)  # below intrinsic value
    with pytest.raises(MqValueError):
        implied_volatility(150, 100, 90, 0.5, 0.02)  # above the spot


def test_equity_valuation():
    assert np.isclose(gordon_growth_value(2.0, 0.08, 0.03), 40)
    with pytest.raises(MqValueError):
        gordon_growth_value(2.0, 0.03, 0.03)
    assert np.isclose(discounted_cash_flow([100, 100], 0.1), 100 / 1.1 + 100 / 1.21)
    with_terminal = discounted_cash_flow([100, 100], 0.1, terminal_growth=0.02)
    assert np.isclose(with_terminal, 100 / 1.1 + 100 / 1.21 + 100 * 1.02 / 0.08 / 1.21)
    # a level perpetuity valued through the terminal value equals C / r
    assert np.isclose(discounted_cash_flow([5.0], 0.05, terminal_growth=0.0), 100)
    with pytest.raises(MqValueError):
        discounted_cash_flow([], 0.1)


def test_capm_expected_return():
    assert np.isclose(capm_expected_return(1.2, 0.02, 0.07), 0.08)
    assert capm_expected_return(0.0, 0.02, 0.07) == 0.02
