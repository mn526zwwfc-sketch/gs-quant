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
    Scenario,
    ValuationAssumptions,
    blume_adjusted_beta,
    cost_of_capital,
    estimate_initial_growth,
    growth_dcf,
    growth_fade_path,
    residual_income_value,
    reverse_dcf_growth,
    value_company,
)


@pytest.fixture
def metrics():
    return pd.Series(
        {
            'revenue': 1000.0,
            'operating_margin': 0.20,
            'operating_margin_3y': 0.18,
            'revenue_growth': 0.25,
            'revenue_cagr_3y': 0.22,
            'tax_rate': 0.21,
            'invested_capital': 500.0,
            'total_debt': 200.0,
            'net_debt': 100.0,
            'shares_outstanding': 100.0,
            'book_value_per_share': 4.0,
            'roe': 0.30,
        }
    )


def test_growth_fade_path():
    path = growth_fade_path(0.30, 0.02, horizon=10, hold_years=2)
    assert len(path) == 10
    assert path[0] == path[1] == 0.30
    assert np.isclose(path[-1], 0.02)
    assert np.all(np.diff(path[1:]) < 0)
    assert np.allclose(np.diff(path[2:]), np.diff(path[2:])[0])  # linear fade
    assert np.allclose(growth_fade_path(0.1, 0.02, 5, hold_years=5), 0.1)
    with pytest.raises(MqValueError):
        growth_fade_path(0.1, 0.02, 5, hold_years=6)


def test_growth_dcf_matches_growing_perpetuity():
    revenue, margin, tax, s2c, wacc, g = 100.0, 0.2, 0.25, 2.0, 0.08, 0.03
    roic = margin * (1 - tax) * s2c  # return on new capital consistent with the sales-to-capital ratio
    result = growth_dcf(revenue, [g] * 10, margin, wacc, g, tax, s2c, terminal_roic=roic)
    nopat_1 = revenue * (1 + g) * margin * (1 - tax)
    assert np.isclose(result.enterprise_value, nopat_1 * (1 - g / roic) / (wacc - g))
    assert np.allclose(result.projections['free_cash_flow'], result.projections['nopat'] * (1 - g / roic))


def test_growth_dcf_no_growth_and_value_driver():
    flat = growth_dcf(100.0, [0.0] * 5, 0.2, 0.1, 0.0, tax_rate=0.0)
    assert np.isclose(flat.enterprise_value, 20 / 0.1)  # perpetuity of NOPAT, no reinvestment
    # when new capital earns exactly its cost, the terminal value is NOPAT_{T+1} / WACC whatever the growth
    for g in (0.0, 0.02, 0.04):
        result = growth_dcf(100.0, [0.1] * 5, 0.2, 0.08, g, 0.2, 1.5)
        nopat_next = result.projections['revenue'].iloc[-1] * (1 + g) * 0.2 * 0.8
        assert np.isclose(result.pv_terminal, nopat_next / 0.08 * result.projections['discount_factor'].iloc[-1])


def test_growth_dcf_capital_structure_and_losses():
    base = growth_dcf(100.0, [0.1] * 5, [0.1, 0.05, 0.0, -0.05, 0.1], 0.09, 0.02, 0.25, 2.0, net_debt=50, shares=10)
    assert np.isclose(base.equity_value, base.enterprise_value - 50)
    assert np.isclose(base.value_per_share, base.equity_value / 10)
    losses = base.projections.loc[4]
    assert np.isclose(losses['nopat'], losses['revenue'] * -0.05)  # no tax credit on operating losses
    assert np.isclose(base.terminal_value_share, base.pv_terminal / base.enterprise_value)
    assert list(base.projections.index) == [1, 2, 3, 4, 5]


def test_growth_is_worth_more_with_higher_returns_on_capital():
    efficient = growth_dcf(100.0, growth_fade_path(0.3, 0.02, 10), 0.2, 0.08, 0.02, sales_to_capital=4.0)
    capital_hungry = growth_dcf(100.0, growth_fade_path(0.3, 0.02, 10), 0.2, 0.08, 0.02, sales_to_capital=0.5)
    assert efficient.enterprise_value > capital_hungry.enterprise_value


def test_growth_dcf_validation():
    with pytest.raises(MqValueError):
        growth_dcf(100.0, [], 0.2, 0.08, 0.02)
    with pytest.raises(MqValueError):
        growth_dcf(100.0, [0.1], 0.2, 0.02, 0.03)
    with pytest.raises(MqValueError):
        growth_dcf(100.0, [0.1, 0.1], [0.2, 0.2, 0.2], 0.08, 0.02)
    with pytest.raises(MqValueError):
        growth_dcf(100.0, [0.1], 0.2, 0.08, 0.02, sales_to_capital=0)


def test_reverse_dcf_round_trip():
    path = growth_fade_path(0.18, 0.025, 10, 2)
    price = growth_dcf(500.0, path, 0.25, 0.085, 0.025, 0.21, 2.0, net_debt=100, shares=20).value_per_share
    implied = reverse_dcf_growth(price, 500.0, 0.25, 0.085, 0.025, 10, 2, 0.21, 2.0, net_debt=100, shares=20)
    assert np.isclose(implied, 0.18, atol=1e-7)
    assert np.isnan(reverse_dcf_growth(price * 1000, 500.0, 0.25, 0.085, 0.025, shares=20))


def test_residual_income_value():
    assert np.isclose(residual_income_value(10.0, 0.09, 0.09), 10.0)  # no abnormal earnings: worth book value
    one_year = residual_income_value(10.0, 0.2, 0.1, horizon=1)
    assert np.isclose(one_year, 10.0)  # ROE fades to the cost of equity within the single year
    two_years = residual_income_value(10.0, 0.2, 0.1, horizon=2, payout_ratio=1.0)
    assert np.isclose(two_years, 10 + (0.15 - 0.1) * 10 / 1.1)
    assert residual_income_value(10.0, 0.3, 0.1) > residual_income_value(10.0, 0.2, 0.1)
    assert np.isnan(residual_income_value(-1.0, 0.2, 0.1))
    assert np.isnan(residual_income_value(10.0, np.nan, 0.1))


def test_cost_of_capital():
    assert np.isclose(blume_adjusted_beta(1.5), 1.335)
    assert np.isclose(blume_adjusted_beta(1.0), 1.0)
    ke, wacc = cost_of_capital(1.2, 0.04, 0.05, market_equity=800, debt=200, pre_tax_cost_of_debt=0.06, tax_rate=0.25)
    assert np.isclose(ke, 0.10)
    assert np.isclose(wacc, 0.8 * 0.10 + 0.2 * 0.06 * 0.75)
    assert np.isclose(cost_of_capital(1.0, 0.04, 0.05, 100, -50, 0.06, 0.25)[1], 0.09)
    with pytest.raises(MqValueError):
        cost_of_capital(1.0, 0.04, 0.05, 0, 0, 0.06, 0.25)


def test_estimate_initial_growth():
    assert np.isclose(estimate_initial_growth(0.3, 0.24, 0.02, persistence=0.5), 0.02 + 0.5 * (0.26 - 0.02))
    assert estimate_initial_growth(2.0, 1.5, 0.02) == 0.35
    assert estimate_initial_growth(-0.8, -0.5, 0.02) == -0.10
    assert np.isclose(estimate_initial_growth(np.nan, 0.12, 0.02, persistence=1.0), 0.12)
    assert estimate_initial_growth(np.nan, np.nan, 0.02) == 0.02


def test_valuation_assumptions_validation():
    with pytest.raises(MqValueError):
        ValuationAssumptions(scenarios={'only': Scenario(0.5, 1.0)})
    with pytest.raises(MqValueError):
        ValuationAssumptions(terminal_growth=0.2)


def test_value_company(metrics):
    assumptions = ValuationAssumptions()
    price = 60.0
    valuation = value_company(metrics, price, beta=1.2, assumptions=assumptions)
    values = valuation.scenario_values
    assert values['bear'] < values['base'] < values['bull']
    expected = sum(values[k] * s.probability for k, s in assumptions.scenarios.items())
    assert np.isclose(valuation.expected_value, expected)
    assert np.isclose(valuation.margin_of_safety, expected / price - 1)
    assert np.isclose(valuation.cost_of_equity, 0.04 + blume_adjusted_beta(1.2) * 0.05)
    assert np.isclose(valuation.initial_growth, 0.025 + 0.6 * ((2 * 0.22 + 0.25) / 3 - 0.025))
    assert np.isfinite(valuation.residual_income_value) and 0 < valuation.terminal_value_share < 1
    assert np.isclose(valuation.expectations_gap, valuation.initial_growth - valuation.implied_growth)

    # the base case valued at the implied growth reproduces the price
    margins = np.linspace(0.20, 0.18, assumptions.horizon + 1)[1:]
    path = growth_fade_path(valuation.implied_growth, 0.025, assumptions.horizon, assumptions.hold_years)
    repriced = growth_dcf(1000.0, path, margins, valuation.wacc, 0.025, 0.21, 2.0, None, 100.0, 100.0)
    assert np.isclose(repriced.value_per_share, price, rtol=1e-6)

    flat = pd.Series(valuation.to_series())
    assert {'value_bear', 'value_base', 'value_bull', 'expectations_gap', 'margin_of_safety'} <= set(flat.index)
    with pytest.raises(MqValueError):
        value_company(metrics, 0.0)


def test_value_company_defaults_for_missing_data(metrics):
    sparse = metrics.drop(['operating_margin_3y', 'invested_capital', 'revenue_cagr_3y', 'roe'])
    valuation = value_company(sparse, 50.0, assumptions=ValuationAssumptions(shrink_beta=False))
    assert np.isclose(valuation.cost_of_equity, 0.09)
    assert np.isnan(valuation.residual_income_value)
    assert np.isclose(valuation.initial_growth, 0.025 + 0.6 * (0.25 - 0.025))
    cheaper = value_company(metrics, 30.0)
    dearer = value_company(metrics, 90.0)
    assert cheaper.margin_of_safety > dearer.margin_of_safety
    assert cheaper.implied_growth < dearer.implied_growth
