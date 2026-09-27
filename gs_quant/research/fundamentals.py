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

Point-in-time company fundamentals and the accounting signals used in academic research.

Fundamentals are provided as a long table with one row per company and fiscal year (see
:func:`prepare_fundamentals` for the schema). Sources include SEC EDGAR (:func:`parse_sec_companyfacts`, free,
with filing dates) or, for Marquee users, :func:`gs_quant.timeseries.measures_factset.factset_fundamentals`.

Signals implemented:

* Growth: revenue growth, 3-year CAGR, EPS growth and the variability of growth (Chan, Karceski & Lakonishok, 2003).
* Profitability: ROIC (Koller, Goedhart & Wessels, 2020), gross profitability (Novy-Marx, 2013), ROA, ROE.
* Earnings quality: accruals (Sloan, 1996), free cash flow margin.
* Investment: asset growth (Cooper, Gulen & Schill, 2008).
* Composite scores: Piotroski (2000) F-score and Mohanram (2005) G-score, the latter designed specifically to separate
  winners from losers among growth (low book-to-market) stocks.
"""

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from gs_quant.errors import MqValueError

_logger = logging.getLogger(__name__)

REQUIRED_FIELDS = (
    'revenue',
    'operating_income',
    'net_income',
    'total_assets',
    'total_equity',
    'shares_outstanding',
)
OPTIONAL_FIELDS = (
    'gross_profit',
    'pretax_income',
    'income_tax',
    'interest_expense',
    'operating_cash_flow',
    'capex',
    'research_development',
    'advertising',
    'current_assets',
    'current_liabilities',
    'total_debt',
    'cash',
)

# SEC XBRL tags for each field, in order of preference (companies switch tags over time)
SEC_TAGS = {
    'revenue': (
        'Revenues',
        'RevenueFromContractWithCustomerExcludingAssessedTax',
        'RevenueFromContractWithCustomerIncludingAssessedTax',
        'SalesRevenueNet',
    ),
    'gross_profit': ('GrossProfit',),
    'operating_income': ('OperatingIncomeLoss',),
    'net_income': ('NetIncomeLoss', 'ProfitLoss'),
    'pretax_income': (
        'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
        'IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments',
    ),
    'income_tax': ('IncomeTaxExpenseBenefit',),
    'interest_expense': ('InterestExpense',),
    'operating_cash_flow': (
        'NetCashProvidedByUsedInOperatingActivities',
        'NetCashProvidedByUsedInOperatingActivitiesContinuingOperations',
    ),
    'capex': ('PaymentsToAcquirePropertyPlantAndEquipment',),
    'research_development': ('ResearchAndDevelopmentExpense',),
    'advertising': ('AdvertisingExpense',),
    'total_assets': ('Assets',),
    'current_assets': ('AssetsCurrent',),
    'current_liabilities': ('LiabilitiesCurrent',),
    'total_debt': ('LongTermDebt', 'LongTermDebtNoncurrent'),
    'cash': ('CashAndCashEquivalentsAtCarryingValue',),
    'total_equity': ('StockholdersEquity', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'),
    'shares_outstanding': ('WeightedAverageNumberOfDilutedSharesOutstanding', 'CommonStockSharesOutstanding'),
}
SEC_ANNUAL_FORMS = ('10-K', '10-K/A', '20-F', '20-F/A', '40-F', '40-F/A')


def prepare_fundamentals(data: pd.DataFrame, default_filing_lag_days: int = 90) -> pd.DataFrame:
    """
    Validate and normalize a table of annual fundamentals

    :param data: one row per company and fiscal year with columns ``ticker``, ``period_end`` (fiscal year end),
        optionally ``filed`` (date the figures became public) and the accounting fields: required
        ``revenue``, ``operating_income``, ``net_income``, ``total_assets``, ``total_equity``,
        ``shares_outstanding``; optional ``gross_profit``, ``pretax_income``, ``income_tax``, ``interest_expense``,
        ``operating_cash_flow``, ``capex`` (positive outflow), ``research_development``, ``advertising``,
        ``current_assets``, ``current_liabilities``, ``total_debt``, ``cash``
    :param default_filing_lag_days: publication delay assumed when ``filed`` is missing (US large filers must file
        the 10-K within 60 to 90 days of year end)
    :return: normalized table sorted by ticker and period end

    **Usage**

    When a fiscal year appears several times (restatements, comparatives in later reports), the first reported
    figures are kept: later revisions were not known to investors at the time, and using them introduces look-ahead
    bias.
    """
    missing = [c for c in ('ticker', 'period_end', *REQUIRED_FIELDS) if c not in data.columns]
    if missing:
        raise MqValueError(f'fundamentals are missing required columns: {missing}')
    df = data.copy()
    df['period_end'] = pd.to_datetime(df['period_end'])
    lagged = df['period_end'] + pd.Timedelta(days=default_filing_lag_days)
    df['filed'] = pd.to_datetime(df['filed']).fillna(lagged) if 'filed' in df.columns else lagged
    for field in OPTIONAL_FIELDS:
        if field not in df.columns:
            df[field] = np.nan
    numeric = [*REQUIRED_FIELDS, *OPTIONAL_FIELDS]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors='coerce')
    df = df.sort_values('filed').drop_duplicates(['ticker', 'period_end'], keep='first')
    return df.sort_values(['ticker', 'period_end']).reset_index(drop=True)


def _average(current: pd.Series, previous: pd.Series) -> pd.Series:
    return pd.concat([current, previous], axis=1).mean(axis=1)


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.where(denominator > 0)


def compute_fundamental_metrics(fundamentals: pd.DataFrame, default_tax_rate: float = 0.21) -> pd.DataFrame:
    """
    Growth, profitability, quality and investment metrics for each company and fiscal year

    :param fundamentals: output of :func:`prepare_fundamentals` (annual data)
    :param default_tax_rate: tax rate used when the effective rate cannot be computed
    :return: the input table with the metric columns appended

    **Usage**

    Key definitions (averages use the current and previous fiscal year):

    * ``roic`` = EBIT (1 - tax) / average (equity + debt - cash)
    * ``operating_margin_3y`` = average EBIT margin of the last three fiscal years
    * ``gross_profitability`` = gross profit / total assets (Novy-Marx, 2013)
    * ``accruals`` = (net income - operating cash flow) / average assets (Sloan, 1996)
    * ``asset_growth`` = total assets / previous total assets - 1 (Cooper, Gulen & Schill, 2008)
    * ``f_score`` = Piotroski (2000) score in 0..9; missing signals count as zero
    """
    df = prepare_fundamentals(fundamentals)
    by = df.groupby('ticker', sort=False)

    def prev(column: str, periods: int = 1) -> pd.Series:
        return by[column].shift(periods)

    df['history_years'] = by.cumcount() + 1
    df['revenue_growth'] = _ratio(df['revenue'], prev('revenue')) - 1
    growth_3y = _ratio(df['revenue'], prev('revenue', 3))
    df['revenue_cagr_3y'] = growth_3y.where(growth_3y > 0) ** (1 / 3) - 1
    df['eps'] = _ratio(df['net_income'], df['shares_outstanding'])
    prev_eps = by['eps'].shift(1)
    df['eps_growth'] = (df['eps'] - prev_eps) / prev_eps.abs().replace(0, np.nan)
    df['growth_volatility'] = (
        df.groupby('ticker', sort=False)['revenue_growth']
        .rolling(5, min_periods=3)
        .std()
        .reset_index(level=0, drop=True)
    )

    df['operating_margin'] = _ratio(df['operating_income'], df['revenue'])
    df['operating_margin_3y'] = (
        df.groupby('ticker', sort=False)['operating_margin']
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df['gross_margin'] = _ratio(df['gross_profit'], df['revenue'])
    df['gross_profitability'] = _ratio(df['gross_profit'], df['total_assets'])
    tax_rate = (df['income_tax'] / df['pretax_income'].where(df['pretax_income'] > 0)).clip(0, 0.5)
    df['tax_rate'] = tax_rate.fillna(default_tax_rate)
    df['nopat'] = df['operating_income'] * (1 - df['tax_rate'])
    invested = df['total_equity'] + df['total_debt'].fillna(0) - df['cash'].fillna(0)
    df['invested_capital'] = invested.where(invested > 0)
    df['roic'] = _ratio(df['nopat'], _average(df['invested_capital'], prev('invested_capital')))
    avg_assets = _average(df['total_assets'], prev('total_assets'))
    df['roa'] = _ratio(df['net_income'], avg_assets)
    df['roe'] = _ratio(
        df['net_income'], _average(df['total_equity'], prev('total_equity')).where(df['total_equity'] > 0)
    )
    df['cfo_to_assets'] = _ratio(df['operating_cash_flow'], avg_assets)
    df['accruals'] = _ratio(df['net_income'] - df['operating_cash_flow'], avg_assets)
    df['free_cash_flow'] = df['operating_cash_flow'] - df['capex']
    df['fcf_margin'] = _ratio(df['free_cash_flow'], df['revenue'])
    df['asset_growth'] = _ratio(df['total_assets'], prev('total_assets')) - 1
    df['asset_turnover'] = _ratio(df['revenue'], prev('total_assets'))
    df['leverage'] = _ratio(df['total_debt'].fillna(0), avg_assets)
    df['current_ratio'] = _ratio(df['current_assets'], df['current_liabilities'])
    df['roa_volatility'] = (
        df.groupby('ticker', sort=False)['roa'].rolling(5, min_periods=3).std().reset_index(level=0, drop=True)
    )
    df['rd_intensity'] = _ratio(df['research_development'], df['total_assets'])
    df['capex_intensity'] = _ratio(df['capex'], df['total_assets'])
    df['advertising_intensity'] = _ratio(df['advertising'], df['total_assets'])
    df['net_debt'] = df['total_debt'].fillna(0) - df['cash'].fillna(0)
    df['book_value_per_share'] = _ratio(df['total_equity'], df['shares_outstanding'])

    by = df.groupby('ticker', sort=False)
    signals = pd.DataFrame(
        {
            'roa_positive': df['roa'] > 0,
            'cfo_positive': df['operating_cash_flow'] > 0,
            'roa_improving': df['roa'] > by['roa'].shift(1),
            'cash_exceeds_earnings': df['cfo_to_assets'] > df['roa'],
            'leverage_not_rising': df['leverage'] <= by['leverage'].shift(1),
            'liquidity_improving': df['current_ratio'] > by['current_ratio'].shift(1),
            'no_dilution': df['shares_outstanding'] <= by['shares_outstanding'].shift(1),
            'margin_improving': df['gross_margin'] > by['gross_margin'].shift(1),
            'turnover_improving': df['asset_turnover'] > by['asset_turnover'].shift(1),
        }
    )
    df['f_score'] = signals.sum(axis=1).astype(int)
    return df


def mohanram_g_score(cross_section: pd.DataFrame, groups: Optional[pd.Series] = None) -> pd.Series:
    """
    Mohanram (2005) G-score for growth stocks

    :param cross_section: one row per company on a given date, with the columns of
        :func:`compute_fundamental_metrics`
    :param groups: optional industry of each company (same index); medians are computed within industries,
        as in the paper, when provided
    :return: score in 0..8 (0..7 without advertising data); missing signals count as zero

    **Usage**

    Eight binary signals comparing each company with the median of its peers: profitability (ROA and cash flow ROA
    above median, cash flow above earnings), stability (earnings and sales growth variability below median) and
    conservative accounting (R&D, capital expenditure and advertising intensity above median). Mohanram (2005),
    "Separating Winners from Losers among Low Book-to-Market Stocks using Financial Statement Analysis", Review of
    Accounting Studies 10, shows that long high-G / short low-G growth stocks earned significant excess returns.
    """
    columns = [
        'roa',
        'cfo_to_assets',
        'roa_volatility',
        'growth_volatility',
        'rd_intensity',
        'capex_intensity',
        'advertising_intensity',
    ]
    data = cross_section[columns]
    medians = data.groupby(groups).transform('median') if groups is not None else data.median()
    above = data.gt(medians)
    below = data.lt(medians)
    signals = pd.DataFrame(
        {
            'roa': above['roa'],
            'cfo': above['cfo_to_assets'],
            'cash_exceeds_earnings': cross_section['cfo_to_assets'] > cross_section['roa'],
            'stable_earnings': below['roa_volatility'],
            'stable_growth': below['growth_volatility'],
            'rd': above['rd_intensity'],
            'capex': above['capex_intensity'],
            'advertising': above['advertising_intensity'],
        }
    )
    return signals.sum(axis=1).astype(int)


def point_in_time_metrics(
    metrics: pd.DataFrame, dates: Sequence, tickers: Optional[Sequence[str]] = None, extra_lag_days: int = 0
) -> pd.DataFrame:
    """
    Latest published fundamentals of every company as of each date

    :param metrics: output of :func:`compute_fundamental_metrics`
    :param dates: evaluation dates
    :param tickers: companies to include; all by default
    :param extra_lag_days: additional safety delay after the filing date
    :return: frame indexed by (date, ticker) with the fiscal year known on that date; companies without any
        published report yet are absent
    """
    available = metrics.assign(available=metrics['filed'] + pd.Timedelta(days=extra_lag_days))
    if tickers is not None:
        available = available[available['ticker'].isin(tickers)]
    available = available.sort_values('available')
    grid = pd.MultiIndex.from_product(
        [pd.DatetimeIndex(sorted(dates)), available['ticker'].unique()], names=['date', 'ticker']
    ).to_frame(index=False)
    merged = pd.merge_asof(
        grid.sort_values('date'), available, left_on='date', right_on='available', by='ticker', direction='backward'
    )
    merged = merged.dropna(subset=['period_end'])
    return merged.set_index(['date', 'ticker']).sort_index()


def _sec_field(facts: dict, tags: Sequence[str], duration: bool) -> pd.DataFrame:
    frames = []
    for priority, tag in enumerate(tags):
        concept = facts.get('facts', {}).get('us-gaap', {}).get(tag)
        if not concept:
            continue
        for unit, entries in concept.get('units', {}).items():
            if unit not in ('USD', 'shares'):
                continue
            df = pd.DataFrame(entries)
            if df.empty or 'form' not in df.columns:
                continue
            df = df[df['form'].isin(SEC_ANNUAL_FORMS)].copy()
            df['end'] = pd.to_datetime(df['end'])
            df['filed'] = pd.to_datetime(df['filed'])
            if duration:
                if 'start' not in df.columns:
                    continue
                days = (df['end'] - pd.to_datetime(df['start'])).dt.days
                df = df[days.between(330, 400)]
            elif 'start' in df.columns:
                df = df[df['start'].isna()]
            df['priority'] = priority
            frames.append(df[['end', 'val', 'filed', 'priority']])
    if not frames:
        return pd.DataFrame(columns=['end', 'val', 'filed'])
    combined = pd.concat(frames)
    # the first report of each fiscal year, from the most preferred tag
    combined = combined.sort_values(['priority', 'filed']).drop_duplicates('end', keep='first')
    return combined[['end', 'val', 'filed']]


def parse_sec_companyfacts(facts: dict, ticker: Optional[str] = None) -> pd.DataFrame:
    """
    Convert an SEC EDGAR ``companyfacts`` document into the annual fundamentals schema

    :param facts: JSON document from ``https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json``
    :param ticker: identifier stored in the ``ticker`` column; defaults to the entity name
    :return: annual fundamentals, as first reported in 10-K/20-F/40-F filings, with their filing dates

    **Usage**

    Fiscal years are defined by the annual (330 to 400 days) revenue and net income facts. Balance sheet items are
    matched on the fiscal year end date. Each row's ``filed`` date is the latest of its fields' first filing dates,
    so the row is only used once all of it was public.
    """
    duration_fields = {
        'revenue',
        'gross_profit',
        'operating_income',
        'net_income',
        'pretax_income',
        'income_tax',
        'interest_expense',
        'operating_cash_flow',
        'capex',
        'research_development',
        'advertising',
    }
    fields = {name: _sec_field(facts, tags, name in duration_fields) for name, tags in SEC_TAGS.items()}
    fields['shares_outstanding'] = pd.concat(
        [
            _sec_field(facts, SEC_TAGS['shares_outstanding'][:1], duration=True),
            _sec_field(facts, SEC_TAGS['shares_outstanding'][1:], duration=False),
        ]
    ).drop_duplicates('end', keep='first')
    period_ends = sorted(set(fields['revenue']['end']) | set(fields['net_income']['end']))
    if not period_ends:
        raise MqValueError('no annual revenue or net income facts found')
    table = pd.DataFrame({'period_end': pd.DatetimeIndex(period_ends)})
    filed = []
    for name, values in fields.items():
        values = values.set_index('end')
        table[name] = table['period_end'].map(values['val']).astype(float)
        filed.append(pd.to_datetime(table['period_end'].map(values['filed'])))
    table['filed'] = pd.concat(filed, axis=1).max(axis=1)
    table.insert(0, 'ticker', ticker or facts.get('entityName', str(facts.get('cik', 'UNKNOWN'))))
    return table


def fetch_sec_companyfacts(cik: int, user_agent: str, timeout: float = 30.0) -> dict:
    """
    Download an SEC EDGAR ``companyfacts`` document

    :param cik: SEC Central Index Key of the company, e.g. 320193 for Apple
    :param user_agent: identification required by the SEC fair access policy, e.g. "Company Name email@domain.com"
    :param timeout: request timeout in seconds
    :return: parsed JSON document, to be passed to :func:`parse_sec_companyfacts`
    """
    url = f'https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json'
    return _sec_get(url, user_agent, timeout)


def _sec_get(url: str, user_agent: str, timeout: float) -> dict:
    import requests

    if not user_agent or '@' not in user_agent:
        raise MqValueError('the SEC requires a user agent with a contact e-mail, e.g. "Name email@domain.com"')
    response = requests.get(url, headers={'User-Agent': user_agent}, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_sec_fundamentals(
    tickers: Sequence[str], user_agent: str, pause: float = 0.15, timeout: float = 30.0
) -> pd.DataFrame:
    """
    Download and parse the annual fundamentals of several US-listed companies from SEC EDGAR

    :param tickers: exchange tickers, e.g. ['AAPL', 'MSFT', 'NVDA']
    :param user_agent: identification required by the SEC fair access policy, e.g. "Company Name email@domain.com"
    :param pause: seconds between requests; the SEC allows at most 10 requests per second
    :param timeout: request timeout in seconds
    :return: annual fundamentals of all companies found, ready for :func:`prepare_fundamentals`

    **Usage**

    Tickers are mapped to SEC identifiers with ``https://www.sec.gov/files/company_tickers.json``. Companies that
    are not found, or that do not report annual US GAAP facts (e.g. foreign filers using IFRS), are skipped with a
    warning.
    """
    import time

    mapping = _sec_get('https://www.sec.gov/files/company_tickers.json', user_agent, timeout)
    ciks = {entry['ticker'].upper(): int(entry['cik_str']) for entry in mapping.values()}
    frames, skipped = [], []
    for ticker in tickers:
        cik = ciks.get(ticker.upper())
        if cik is None:
            skipped.append(ticker)
            continue
        try:
            frames.append(parse_sec_companyfacts(fetch_sec_companyfacts(cik, user_agent, timeout), ticker.upper()))
        except MqValueError:
            skipped.append(ticker)
        time.sleep(pause)
    if skipped:
        _logger.warning('no SEC annual US GAAP fundamentals for: %s', ', '.join(skipped))
    if not frames:
        raise MqValueError('none of the tickers could be downloaded')
    return pd.concat(frames, ignore_index=True)
