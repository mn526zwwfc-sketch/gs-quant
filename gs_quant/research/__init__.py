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

Academic quantitative finance toolkit.

Local (no Marquee session required) implementations of data management, risk, valuation, portfolio construction,
trading strategies and backtesting methods from peer-reviewed research. Each function documents the paper it is
based on; see README.md in this package for an overview and the full bibliography.
"""

from .backtest import (
    BacktestResult,
    backtest,
    purged_kfold_splits,
    rebalance_dates,
    rolling_allocation,
    walk_forward_splits,
    weights_frame,
)
from .data_management import (
    DataQualityReport,
    ReturnType,
    aggregate_returns,
    align,
    clean_prices,
    compute_returns,
    cross_sectional_rank,
    cross_sectional_zscore,
    detect_outliers,
    hampel_filter,
    lag_signal,
    point_in_time,
    validate_prices,
    winsorize,
)
from .factors import (
    FactorRegressionResult,
    FamaMacBethResult,
    factor_regression,
    fama_macbeth,
    newey_west_lags,
)
from .fundamentals import (
    compute_fundamental_metrics,
    fetch_sec_companyfacts,
    fetch_sec_fundamentals,
    mohanram_g_score,
    parse_sec_companyfacts,
    point_in_time_metrics,
    prepare_fundamentals,
)
from .growth_strategy import (
    GrowthStrategyConfig,
    GrowthStrategyResult,
    LongTermGrowthStrategy,
)
from .growth_valuation import (
    CompanyValuation,
    DCFResult,
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
from .performance import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    deflated_sharpe_ratio,
    drawdown,
    expected_maximum_sharpe,
    infer_periods_per_year,
    information_ratio,
    max_drawdown,
    minimum_track_record_length,
    omega_ratio,
    performance_summary,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    sortino_ratio,
)
from .portfolio import (
    BlackLittermanResult,
    black_litterman,
    efficient_frontier,
    equal_weight,
    hierarchical_risk_parity,
    inverse_volatility,
    kelly_fraction,
    kelly_weights,
    maximum_sharpe,
    mean_variance,
    minimum_variance,
    risk_parity,
)
from .risk import (
    VaRMethod,
    covariance_to_correlation,
    denoise_correlation,
    denoised_covariance,
    ewma_covariance,
    expected_shortfall,
    ledoit_wolf_covariance,
    ledoit_wolf_shrinkage,
    portfolio_volatility,
    risk_contributions,
    sample_covariance,
    value_at_risk,
)
from .signals import (
    combine_signals,
    cross_sectional_momentum,
    ex_ante_volatility,
    low_volatility,
    moving_average_crossover,
    short_term_reversal,
    time_series_momentum,
    volatility_scaling,
)
from .valuation import (
    BondAnalytics,
    OptionAnalytics,
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
