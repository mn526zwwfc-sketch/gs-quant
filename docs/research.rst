Research Package
================

Local implementations of data management, risk, valuation, portfolio construction, systematic strategies and
backtesting methods from peer-reviewed research. No Marquee session is required.

Data Management
---------------

.. currentmodule:: gs_quant.research.data_management

.. autosummary::
   :toctree: functions

   validate_prices
   clean_prices
   detect_outliers
   hampel_filter
   winsorize
   cross_sectional_zscore
   cross_sectional_rank
   compute_returns
   aggregate_returns
   align
   point_in_time
   lag_signal

.. autosummary::
   :toctree: classes

   DataQualityReport
   ReturnType

Risk
----

.. currentmodule:: gs_quant.research.risk

.. autosummary::
   :toctree: functions

   sample_covariance
   ewma_covariance
   ledoit_wolf_covariance
   ledoit_wolf_shrinkage
   covariance_to_correlation
   denoise_correlation
   denoised_covariance
   portfolio_volatility
   risk_contributions
   value_at_risk
   expected_shortfall

Performance
-----------

.. currentmodule:: gs_quant.research.performance

.. autosummary::
   :toctree: functions

   annualized_return
   annualized_volatility
   sharpe_ratio
   sortino_ratio
   drawdown
   max_drawdown
   calmar_ratio
   omega_ratio
   information_ratio
   probabilistic_sharpe_ratio
   expected_maximum_sharpe
   deflated_sharpe_ratio
   minimum_track_record_length
   performance_summary
   infer_periods_per_year

Valuation
---------

.. currentmodule:: gs_quant.research.valuation

.. autosummary::
   :toctree: functions

   bond_cash_flows
   bond_price
   bond_yield
   bond_analytics
   black_scholes
   implied_volatility
   gordon_growth_value
   discounted_cash_flow
   capm_expected_return

Factor Models
-------------

.. currentmodule:: gs_quant.research.factors

.. autosummary::
   :toctree: functions

   factor_regression
   fama_macbeth
   newey_west_lags

Portfolio Construction
----------------------

.. currentmodule:: gs_quant.research.portfolio

.. autosummary::
   :toctree: functions

   equal_weight
   inverse_volatility
   minimum_variance
   mean_variance
   maximum_sharpe
   risk_parity
   hierarchical_risk_parity
   kelly_weights
   kelly_fraction
   black_litterman
   efficient_frontier

Strategies
----------

.. currentmodule:: gs_quant.research.signals

.. autosummary::
   :toctree: functions

   time_series_momentum
   cross_sectional_momentum
   short_term_reversal
   moving_average_crossover
   low_volatility
   volatility_scaling
   combine_signals
   ex_ante_volatility

Backtesting
-----------

.. currentmodule:: gs_quant.research.backtest

.. autosummary::
   :toctree: functions

   backtest
   rolling_allocation
   rebalance_dates
   walk_forward_splits
   purged_kfold_splits
