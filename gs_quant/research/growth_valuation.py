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

Valuation of growth companies.

Valuing fast growers is mostly about how long the growth lasts and what it costs. The models here follow the
academic and professional consensus:

* Growth fades: sales growth shows little persistence beyond chance (Chan, Karceski & Lakonishok, 2003) and
  investors who extrapolate past growth overpay for "glamour" stocks (Lakonishok, Shleifer & Vishny, 1994;
  La Porta, 1996). Observed growth is shrunk towards the long-run rate and faded over the explicit horizon.
* Growth is not free: every unit of new revenue requires investment, set by the sales-to-capital ratio
  (Damodaran, 2012), so growth only creates value when the return on new capital exceeds its cost
  (Koller, Goedhart & Wessels, 2020, "Valuation", the value driver formula).
* Competition erodes excess returns: margins revert towards their history and the terminal return on new capital
  defaults to the cost of capital (Fama & French, 2000; Koller et al., 2020).
* Expectations investing: the reverse DCF solves for the growth the market price implies, to compare with a
  realistic forecast (Rappaport & Mauboussin, 2001).
* Cross-check with the residual income model (Ohlson, 1995; Frankel & Lee, 1998) and a probability-weighted
  bear/base/bull valuation with a margin of safety (Graham & Dodd, 1934; Damodaran, 2012).
"""

from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import optimize

from gs_quant.errors import MqValueError

FloatOrPath = Union[float, Sequence[float], np.ndarray]


@dataclass(frozen=True)
class DCFResult:
    """
    Output of :func:`growth_dcf`; values are in the currency of the inputs
    """

    enterprise_value: float
    equity_value: float
    value_per_share: float
    pv_explicit: float
    pv_terminal: float
    projections: pd.DataFrame

    @property
    def terminal_value_share(self) -> float:
        """Fraction of enterprise value coming from the terminal value"""
        return self.pv_terminal / self.enterprise_value if self.enterprise_value else np.nan


@dataclass(frozen=True)
class Scenario:
    """
    A valuation scenario: probability, multiplier on the initial growth rate and shift of the target margin
    """

    probability: float
    growth_multiplier: float
    margin_shift: float = 0.0


def _default_scenarios() -> dict[str, Scenario]:
    return {
        'bear': Scenario(0.25, 0.5, -0.03),
        'base': Scenario(0.50, 1.0, 0.0),
        'bull': Scenario(0.25, 1.5, 0.02),
    }


@dataclass(frozen=True)
class ValuationAssumptions:
    """
    Market and model assumptions for :func:`value_company`

    :param risk_free: long-term government bond yield
    :param equity_risk_premium: expected excess return of equities; implied premia have averaged 4-6% for the US
        (Damodaran, 2012)
    :param terminal_growth: perpetual growth after the horizon; must not exceed long-run nominal GDP growth
    :param horizon: years of explicit forecast
    :param hold_years: years during which the initial growth is sustained before fading
    :param growth_persistence: fraction of the gap between observed and terminal growth that is expected to persist
        (Chan, Karceski & Lakonishok, 2003, find very little persistence)
    :param max_initial_growth: cap on the initial growth forecast
    :param min_initial_growth: floor on the initial growth forecast
    :param terminal_roic: return on new capital after the horizon; the cost of capital when omitted, i.e.
        growth then adds no value (Koller et al., 2020)
    :param default_sales_to_capital: revenue per unit of invested capital when it cannot be computed
    :param cost_of_debt_spread: credit spread over the risk-free rate for the pre-tax cost of debt
    :param payout_ratio: dividend payout used by the residual income model
    :param shrink_beta: apply the Blume (1971) adjustment towards one
    :param scenarios: bear/base/bull scenarios; probabilities must sum to one
    """

    risk_free: float = 0.04
    equity_risk_premium: float = 0.05
    terminal_growth: float = 0.025
    horizon: int = 10
    hold_years: int = 2
    growth_persistence: float = 0.6
    max_initial_growth: float = 0.35
    min_initial_growth: float = -0.10
    terminal_roic: Optional[float] = None
    default_sales_to_capital: float = 1.5
    cost_of_debt_spread: float = 0.015
    payout_ratio: float = 0.25
    shrink_beta: bool = True
    scenarios: dict[str, Scenario] = field(default_factory=_default_scenarios)

    def __post_init__(self):
        if not np.isclose(sum(s.probability for s in self.scenarios.values()), 1):
            raise MqValueError('scenario probabilities must sum to one')
        if self.terminal_growth >= self.risk_free + self.equity_risk_premium:
            raise MqValueError('terminal growth must be below the cost of capital')


@dataclass(frozen=True)
class CompanyValuation:
    """
    Output of :func:`value_company`; per-share values are in the currency of the price
    """

    price: float
    cost_of_equity: float
    wacc: float
    initial_growth: float
    implied_growth: float
    scenario_values: dict[str, float]
    expected_value: float
    margin_of_safety: float
    residual_income_value: float
    terminal_value_share: float

    @property
    def expectations_gap(self) -> float:
        """Forecast growth minus the growth implied by the price; positive means the market expects too little"""
        return self.initial_growth - self.implied_growth

    def to_series(self) -> pd.Series:
        """Flatten the valuation into a single series"""
        data = {k: v for k, v in asdict(self).items() if k != 'scenario_values'}
        data.update({f'value_{k}': v for k, v in self.scenario_values.items()})
        data['expectations_gap'] = self.expectations_gap
        return pd.Series(data)


def growth_fade_path(
    initial_growth: float, terminal_growth: float, horizon: int = 10, hold_years: int = 0
) -> np.ndarray:
    """
    Growth rates that hold for a while and then fade linearly to the terminal rate

    :param initial_growth: growth rate in the first years
    :param terminal_growth: growth rate reached in the last year of the horizon
    :param horizon: number of years
    :param hold_years: years during which the initial growth is sustained
    :return: array of ``horizon`` annual growth rates

    **Usage**

    Multi-stage growth models (Fuller & Hsia, 1984; Damodaran, 2012) capture the empirical fact that high growth
    rates revert towards the economy-wide rate within a few years (Chan, Karceski & Lakonishok, 2003).
    """
    if horizon < 1 or not 0 <= hold_years <= horizon:
        raise MqValueError('horizon must be positive and hold_years between 0 and horizon')
    years = np.arange(1, horizon + 1)
    fade = horizon - hold_years
    fraction = np.clip((years - hold_years) / fade, 0, 1) if fade else np.zeros(horizon)
    return initial_growth + (terminal_growth - initial_growth) * fraction


def _path(value: FloatOrPath, horizon: int, name: str) -> np.ndarray:
    path = np.broadcast_to(np.asarray(value, dtype=float), (horizon,)) if np.ndim(value) == 0 else np.asarray(value)
    if path.shape != (horizon,):
        raise MqValueError(f'{name} must be a scalar or have one value per year of the growth path')
    return path.astype(float)


def growth_dcf(
    revenue: float,
    growth: Sequence[float],
    operating_margin: FloatOrPath,
    wacc: float,
    terminal_growth: float,
    tax_rate: float = 0.21,
    sales_to_capital: float = 1.5,
    terminal_roic: Optional[float] = None,
    net_debt: float = 0.0,
    shares: float = 1.0,
) -> DCFResult:
    """
    Free cash flow to the firm DCF for a growing company

    :param revenue: revenue of the last fiscal year
    :param growth: revenue growth for each year of the explicit forecast, e.g. from :func:`growth_fade_path`
    :param operating_margin: EBIT margin, constant or one value per year
    :param wacc: weighted average cost of capital
    :param terminal_growth: perpetual growth after the explicit forecast
    :param tax_rate: tax rate on positive operating income
    :param sales_to_capital: additional revenue generated per unit of new invested capital
    :param terminal_roic: return on new invested capital after the forecast; defaults to ``wacc``
    :param net_debt: debt minus cash, subtracted from the enterprise value
    :param shares: number of shares
    :return: :class:`DCFResult`

    **Usage**

    For each year, :math:`NOPAT_t = R_t m_t - \\tau \\max(R_t m_t, 0)`, the investment needed to grow is
    :math:`(R_{t+1} - R_t) / (S/C)` and :math:`FCF_t = NOPAT_t - \\text{reinvestment}_t` (Damodaran, 2012). The
    terminal value uses the value driver formula
    :math:`TV_T = NOPAT_{T+1} (1 - g / RONIC) / (WACC - g)` (Koller, Goedhart & Wessels, 2020), so terminal growth
    only adds value when the return on new capital exceeds the cost of capital.
    """
    growth = np.asarray(growth, dtype=float)
    horizon = len(growth)
    if horizon == 0:
        raise MqValueError('growth path must not be empty')
    if wacc <= terminal_growth:
        raise MqValueError('wacc must exceed terminal growth')
    if sales_to_capital <= 0 or shares <= 0:
        raise MqValueError('sales_to_capital and shares must be positive')
    terminal_roic = wacc if terminal_roic is None else terminal_roic
    margins = _path(operating_margin, horizon, 'operating_margin')

    revenues = revenue * np.cumprod(1 + growth)
    ebit = revenues * margins
    nopat = ebit - tax_rate * np.maximum(ebit, 0)
    next_revenue = np.append(revenues[1:], revenues[-1] * (1 + terminal_growth))
    reinvestment = (next_revenue - revenues) / sales_to_capital
    fcf = nopat - reinvestment
    discount = (1 + wacc) ** -np.arange(1, horizon + 1)

    terminal_ebit = next_revenue[-1] * margins[-1]
    terminal_nopat = terminal_ebit - tax_rate * max(terminal_ebit, 0)
    terminal_value = terminal_nopat * (1 - terminal_growth / terminal_roic) / (wacc - terminal_growth)
    pv_explicit = float(fcf @ discount)
    pv_terminal = float(terminal_value * discount[-1])
    enterprise_value = pv_explicit + pv_terminal
    equity_value = enterprise_value - net_debt
    projections = pd.DataFrame(
        {
            'revenue': revenues,
            'growth': growth,
            'operating_margin': margins,
            'nopat': nopat,
            'reinvestment': reinvestment,
            'free_cash_flow': fcf,
            'discount_factor': discount,
        },
        index=pd.RangeIndex(1, horizon + 1, name='year'),
    )
    return DCFResult(enterprise_value, equity_value, equity_value / shares, pv_explicit, pv_terminal, projections)


def reverse_dcf_growth(
    price: float,
    revenue: float,
    operating_margin: FloatOrPath,
    wacc: float,
    terminal_growth: float,
    horizon: int = 10,
    hold_years: int = 2,
    tax_rate: float = 0.21,
    sales_to_capital: float = 1.5,
    terminal_roic: Optional[float] = None,
    net_debt: float = 0.0,
    shares: float = 1.0,
    bounds: tuple[float, float] = (-0.5, 2.0),
) -> float:
    """
    Initial growth rate implied by the market price (reverse DCF)

    :param price: current share price
    :param revenue: revenue of the last fiscal year
    :param operating_margin: EBIT margin, constant or one value per year
    :param wacc: weighted average cost of capital
    :param terminal_growth: perpetual growth after the explicit forecast
    :param horizon: years of explicit forecast
    :param hold_years: years during which the initial growth is sustained before fading
    :param tax_rate: tax rate on positive operating income
    :param sales_to_capital: additional revenue generated per unit of new invested capital
    :param terminal_roic: return on new invested capital after the forecast; defaults to ``wacc``
    :param net_debt: debt minus cash
    :param shares: number of shares
    :param bounds: search interval for the growth rate
    :return: implied initial growth, or NaN when no growth rate within the bounds justifies the price

    **Usage**

    Rather than forecasting, "expectations investing" asks what the price already assumes and whether that is
    plausible (Rappaport & Mauboussin, 2001). A price implying 40% annual growth for years is a warning sign; one
    implying less growth than the company can realistically deliver is an opportunity.
    """

    def gap(initial: float) -> float:
        path = growth_fade_path(initial, terminal_growth, horizon, hold_years)
        return (
            growth_dcf(
                revenue,
                path,
                operating_margin,
                wacc,
                terminal_growth,
                tax_rate,
                sales_to_capital,
                terminal_roic,
                net_debt,
                shares,
            ).value_per_share
            - price
        )

    low, high = gap(bounds[0]), gap(bounds[1])
    if np.sign(low) == np.sign(high):
        return np.nan
    return float(optimize.brentq(gap, *bounds, xtol=1e-8))


def residual_income_value(
    book_value_per_share: float,
    roe: float,
    cost_of_equity: float,
    horizon: int = 10,
    payout_ratio: float = 0.25,
) -> float:
    """
    Residual income (abnormal earnings) valuation with ROE fading to the cost of equity

    :param book_value_per_share: current book value of equity per share
    :param roe: current return on equity
    :param cost_of_equity: required return on equity
    :param horizon: years over which ROE converges to the cost of equity
    :param payout_ratio: fraction of earnings paid out; the rest is added to book value (clean surplus)
    :return: :math:`B_0 + \\sum_t (ROE_t - r) B_{t-1} / (1 + r)^t`

    **Usage**

    Value equals book value plus the present value of earnings above the cost of equity (Ohlson, 1995). Anchoring
    on book value makes the estimate less sensitive to the terminal value than a DCF, and the ratio of this value
    to the price predicts returns (Frankel & Lee, 1998).
    """
    if book_value_per_share <= 0 or not np.isfinite(roe):
        return np.nan
    roes = roe + (cost_of_equity - roe) * np.arange(1, horizon + 1) / horizon
    book, value = book_value_per_share, book_value_per_share
    for year, roe_t in enumerate(roes, start=1):
        value += (roe_t - cost_of_equity) * book / (1 + cost_of_equity) ** year
        book *= 1 + roe_t * (1 - payout_ratio)
    return float(value)


def blume_adjusted_beta(beta: float) -> float:
    """
    Blume (1971) adjustment of an estimated beta towards the market beta of one

    :param beta: raw regression beta
    :return: :math:`0.67 \\beta + 0.33`
    """
    return 0.67 * beta + 0.33


def cost_of_capital(
    beta: float,
    risk_free: float,
    equity_risk_premium: float,
    market_equity: float,
    debt: float,
    pre_tax_cost_of_debt: float,
    tax_rate: float,
) -> tuple[float, float]:
    """
    Cost of equity (CAPM) and weighted average cost of capital

    :param beta: equity beta
    :param risk_free: risk-free rate
    :param equity_risk_premium: expected market excess return
    :param market_equity: market value of equity
    :param debt: value of debt
    :param pre_tax_cost_of_debt: yield on the company's debt
    :param tax_rate: marginal tax rate (interest is tax deductible)
    :return: tuple (cost of equity, WACC)
    """
    cost_of_equity = risk_free + beta * equity_risk_premium
    total = market_equity + max(debt, 0.0)
    if total <= 0:
        raise MqValueError('the firm must have positive capital')
    wacc = (market_equity * cost_of_equity + max(debt, 0.0) * pre_tax_cost_of_debt * (1 - tax_rate)) / total
    return cost_of_equity, wacc


def estimate_initial_growth(
    revenue_growth: float,
    revenue_cagr_3y: float,
    terminal_growth: float,
    persistence: float = 0.6,
    floor: float = -0.10,
    cap: float = 0.35,
) -> float:
    """
    Forecast of next year's revenue growth from recent history, shrunk towards the long-run rate

    :param revenue_growth: last year's revenue growth
    :param revenue_cagr_3y: three-year compound annual revenue growth
    :param terminal_growth: long-run growth rate towards which the forecast is shrunk
    :param persistence: fraction of the excess growth expected to persist
    :param floor: minimum forecast
    :param cap: maximum forecast
    :return: :math:`g_\\infty + \\phi (\\bar g - g_\\infty)`, where :math:`\\bar g` weighs the 3-year CAGR 2/3 and
        the last year 1/3
    """
    observed = [(v, w) for v, w in ((revenue_cagr_3y, 2.0), (revenue_growth, 1.0)) if np.isfinite(v)]
    if not observed:
        return terminal_growth
    blended = sum(v * w for v, w in observed) / sum(w for _, w in observed)
    return float(np.clip(terminal_growth + persistence * (blended - terminal_growth), floor, cap))


def value_company(
    metrics: pd.Series, price: float, beta: float = 1.0, assumptions: ValuationAssumptions = ValuationAssumptions()
) -> CompanyValuation:
    """
    Full valuation of a company from its latest fundamentals

    :param metrics: one row of :func:`gs_quant.research.fundamentals.compute_fundamental_metrics`
    :param price: current share price
    :param beta: raw equity beta, e.g. from a regression on market returns
    :param assumptions: :class:`ValuationAssumptions`
    :return: :class:`CompanyValuation`

    **Usage**

    1. Cost of capital from the CAPM (Blume-adjusted beta) and market-value weights.
    2. Initial growth from :func:`estimate_initial_growth`, held for ``hold_years`` and faded to the terminal rate.
    3. Operating margin moving linearly from its current level to its three-year average (profitability mean
       reversion, Fama & French, 2000), plus a shift in each scenario.
    4. Bear/base/bull DCF values weighted by their probabilities, and the margin of safety
       :math:`E[V] / P - 1` (Graham & Dodd, 1934).
    5. Reverse DCF implied growth and a residual income cross-check.
    """
    a = assumptions
    if price <= 0:
        raise MqValueError('price must be positive')
    shares = float(metrics['shares_outstanding'])
    tax_rate = float(metrics.get('tax_rate', 0.21))
    beta = blume_adjusted_beta(beta) if a.shrink_beta else beta
    debt = float(np.nan_to_num(metrics.get('total_debt', 0.0)))
    cost_of_equity, wacc = cost_of_capital(
        beta, a.risk_free, a.equity_risk_premium, price * shares, debt, a.risk_free + a.cost_of_debt_spread, tax_rate
    )
    wacc = max(wacc, a.terminal_growth + 0.01)
    invested = metrics.get('invested_capital', np.nan)
    sales_to_capital = metrics['revenue'] / invested if invested and np.isfinite(invested) else np.nan
    sales_to_capital = float(np.clip(sales_to_capital, 0.2, 10)) if np.isfinite(sales_to_capital) else None
    sales_to_capital = sales_to_capital or a.default_sales_to_capital
    initial_growth = estimate_initial_growth(
        metrics.get('revenue_growth', np.nan),
        metrics.get('revenue_cagr_3y', np.nan),
        a.terminal_growth,
        a.growth_persistence,
        a.min_initial_growth,
        a.max_initial_growth,
    )
    current_margin = float(metrics['operating_margin'])
    target_margin = metrics.get('operating_margin_3y', np.nan)
    target_margin = float(target_margin) if np.isfinite(target_margin) else current_margin
    common = dict(
        wacc=wacc,
        terminal_growth=a.terminal_growth,
        tax_rate=tax_rate,
        sales_to_capital=sales_to_capital,
        terminal_roic=a.terminal_roic,
        net_debt=float(np.nan_to_num(metrics.get('net_debt', 0.0))),
        shares=shares,
    )
    values, terminal_share = {}, np.nan
    for name, scenario in a.scenarios.items():
        margins = np.linspace(current_margin, target_margin + scenario.margin_shift, a.horizon + 1)[1:]
        growth = growth_fade_path(
            initial_growth * scenario.growth_multiplier, a.terminal_growth, a.horizon, a.hold_years
        )
        result = growth_dcf(metrics['revenue'], growth, margins, **common)
        values[name] = result.value_per_share
        if name == 'base':
            terminal_share = result.terminal_value_share
    expected = sum(values[name] * s.probability for name, s in a.scenarios.items())
    base_margins = np.linspace(current_margin, target_margin, a.horizon + 1)[1:]
    implied = reverse_dcf_growth(
        price, metrics['revenue'], base_margins, horizon=a.horizon, hold_years=a.hold_years, **common
    )
    rim = residual_income_value(
        float(metrics.get('book_value_per_share', np.nan)),
        float(metrics.get('roe', np.nan)),
        cost_of_equity,
        a.horizon,
        a.payout_ratio,
    )
    return CompanyValuation(
        price=float(price),
        cost_of_equity=float(cost_of_equity),
        wacc=float(wacc),
        initial_growth=initial_growth,
        implied_growth=implied,
        scenario_values=values,
        expected_value=float(expected),
        margin_of_safety=float(expected / price - 1),
        residual_income_value=rim,
        terminal_value_share=float(terminal_share),
    )
