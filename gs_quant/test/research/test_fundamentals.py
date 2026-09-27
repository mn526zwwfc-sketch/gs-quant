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

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from gs_quant.errors import MqValueError
from gs_quant.research import (
    compute_fundamental_metrics,
    fetch_sec_companyfacts,
    fetch_sec_fundamentals,
    mohanram_g_score,
    parse_sec_companyfacts,
    point_in_time_metrics,
    prepare_fundamentals,
)


def _company(ticker, years, growth=0.1, margin=0.2, improving=True, start=2015):
    rows, revenue = [], 100.0
    for i in range(years):
        revenue *= 1 + growth
        drift = i if improving else -i
        rows.append(
            {
                'ticker': ticker,
                'period_end': f'{start + i}-12-31',
                'revenue': revenue,
                'gross_profit': revenue * (0.5 + 0.01 * drift),
                'operating_income': revenue * margin,
                'pretax_income': revenue * margin,
                'income_tax': revenue * margin * 0.25,
                'net_income': revenue * margin * 0.75 * (1 + 0.05 * drift),
                'operating_cash_flow': revenue * 0.2,
                'capex': revenue * 0.05,
                'research_development': revenue * 0.1,
                'total_assets': 150.0 + 5 * i,
                'current_assets': 50.0 + 2 * drift,
                'current_liabilities': 30.0,
                'total_debt': 40.0 - 2 * drift,
                'cash': 10.0,
                'total_equity': 90.0,
                'shares_outstanding': 100.0 - drift,
            }
        )
    return rows


@pytest.fixture
def fundamentals():
    return pd.DataFrame(_company('GROW', 6, growth=0.2) + _company('SLOW', 6, growth=0.02, improving=False))


def test_prepare_fundamentals_schema():
    raw = pd.DataFrame(_company('A', 2))
    prepared = prepare_fundamentals(raw.drop(columns=['capex', 'cash']), default_filing_lag_days=60)
    assert (prepared['filed'] == prepared['period_end'] + pd.Timedelta(days=60)).all()
    assert prepared['capex'].isna().all() and 'advertising' in prepared.columns
    with pytest.raises(MqValueError, match='revenue'):
        prepare_fundamentals(raw.drop(columns=['revenue']))


def test_prepare_fundamentals_keeps_first_report():
    raw = pd.DataFrame(_company('A', 1))
    original = raw.assign(filed='2016-02-15')
    restated = raw.assign(filed='2017-02-15', revenue=999.0)
    prepared = prepare_fundamentals(pd.concat([restated, original]))
    assert len(prepared) == 1
    assert prepared['revenue'].iloc[0] == raw['revenue'].iloc[0]


def test_growth_and_profitability_metrics(fundamentals):
    m = compute_fundamental_metrics(fundamentals).set_index(['ticker', 'period_end'])
    grow = m.loc['GROW']
    assert np.allclose(grow['revenue_growth'].dropna(), 0.2)
    assert np.allclose(grow['revenue_cagr_3y'].dropna(), 0.2)
    assert grow['revenue_cagr_3y'].isna().sum() == 3
    assert list(grow['history_years']) == [1, 2, 3, 4, 5, 6]
    last, prev = grow.iloc[-1], grow.iloc[-2]
    invested = 90 + (40 - 2 * np.arange(6)) - 10
    assert np.isclose(last['tax_rate'], 0.25)
    assert np.isclose(last['roic'], last['operating_income'] * 0.75 / ((invested[-1] + invested[-2]) / 2))
    avg_assets = (last['total_assets'] + prev['total_assets']) / 2
    assert np.isclose(last['accruals'], (last['net_income'] - last['operating_cash_flow']) / avg_assets)
    assert np.isclose(last['gross_profitability'], last['gross_profit'] / last['total_assets'])
    assert np.isclose(last['fcf_margin'], 0.15)
    assert np.isclose(last['asset_growth'], last['total_assets'] / prev['total_assets'] - 1)
    assert np.isclose(last['operating_margin_3y'], 0.2)
    assert np.isclose(grow['growth_volatility'].iloc[-1], 0, atol=1e-12)
    assert np.isnan(grow['growth_volatility'].iloc[2])  # needs three growth observations


def test_piotroski_f_score(fundamentals):
    m = compute_fundamental_metrics(fundamentals).groupby('ticker')['f_score'].last()
    assert m['GROW'] >= 8  # improving profitability, liquidity, margins, less debt, buybacks
    assert m['SLOW'] < m['GROW']
    assert m.between(0, 9).all()


def test_default_tax_rate_and_invalid_denominators():
    raw = pd.DataFrame(_company('A', 3)).assign(pretax_income=-1.0, total_equity=-5.0)
    m = compute_fundamental_metrics(raw, default_tax_rate=0.3)
    assert np.allclose(m['tax_rate'], 0.3)
    assert m['roe'].isna().all()  # undefined with negative equity
    assert (m['book_value_per_share'] < 0).all()


def test_mohanram_g_score():
    cross = pd.DataFrame(
        {
            'roa': [0.2, 0.1, 0.0],
            'cfo_to_assets': [0.3, 0.05, 0.1],
            'roa_volatility': [0.01, 0.05, 0.1],
            'growth_volatility': [0.02, 0.05, 0.1],
            'rd_intensity': [0.1, 0.05, 0.0],
            'capex_intensity': [0.1, 0.05, 0.0],
            'advertising_intensity': [np.nan, np.nan, np.nan],
        },
        index=['best', 'mid', 'worst'],
    )
    score = mohanram_g_score(cross)
    assert score['best'] == 7  # every signal except advertising, which is missing
    assert score['worst'] == 1  # only cash flow above earnings
    grouped = mohanram_g_score(cross, pd.Series(['x', 'x', 'y'], index=cross.index))
    assert grouped['worst'] == 1  # alone in its group: never above or below its own median


def test_point_in_time_metrics(fundamentals):
    m = compute_fundamental_metrics(
        fundamentals.assign(filed=lambda d: pd.to_datetime(d['period_end']) + pd.Timedelta(days=45))
    )
    dates = pd.to_datetime(['2016-02-13', '2016-02-15', '2030-01-01'])
    pit = point_in_time_metrics(m, dates)
    # fiscal 2015 is filed on 2016-02-14: invisible the day before
    assert (pd.Timestamp('2016-02-13'), 'GROW') not in pit.index
    assert pit.loc[('2016-02-15', 'GROW'), 'period_end'] == pd.Timestamp('2015-12-31')
    assert pit.loc[('2030-01-01', 'SLOW'), 'period_end'] == pd.Timestamp('2020-12-31')
    lagged = point_in_time_metrics(m, dates, tickers=['GROW'], extra_lag_days=30)
    assert ('2016-02-15', 'GROW') not in lagged.index
    assert lagged.index.get_level_values('ticker').unique().tolist() == ['GROW']
    early = point_in_time_metrics(m, pd.to_datetime(['2010-01-01']))
    assert early.empty


def _fact(end, val, filed, form='10-K', start=None):
    entry = {'end': end, 'val': val, 'filed': filed, 'form': form, 'fy': int(end[:4]), 'fp': 'FY'}
    if start:
        entry['start'] = start
    return entry


@pytest.fixture
def companyfacts():
    usd = 'USD'
    return {
        'cik': 1234,
        'entityName': 'Example Corp',
        'facts': {
            'us-gaap': {
                'Revenues': {
                    'units': {
                        usd: [
                            _fact('2020-12-31', 1000, '2021-02-20', start='2020-01-01'),
                            _fact('2020-12-31', 1100, '2022-02-20', start='2020-01-01'),  # restated comparative
                            _fact('2021-03-31', 300, '2021-05-01', form='10-Q', start='2021-01-01'),
                        ]
                    }
                },
                'RevenueFromContractWithCustomerExcludingAssessedTax': {
                    'units': {
                        usd: [
                            _fact('2020-12-31', 999, '2021-02-20', start='2020-01-01'),  # lower priority tag
                            _fact('2021-12-31', 1200, '2022-02-20', start='2021-01-01'),
                            _fact('2021-12-31', 320, '2022-02-20', start='2021-10-01'),  # Q4 inside the 10-K
                        ]
                    }
                },
                'NetIncomeLoss': {
                    'units': {
                        usd: [
                            _fact('2020-12-31', 100, '2021-02-20', start='2020-01-01'),
                            _fact('2021-12-31', 130, '2022-02-20', start='2021-01-01'),
                        ]
                    }
                },
                'Assets': {
                    'units': {usd: [_fact('2020-12-31', 5000, '2021-02-20'), _fact('2021-12-31', 5500, '2022-03-01')]}
                },
                'WeightedAverageNumberOfDilutedSharesOutstanding': {
                    'units': {'shares': [_fact('2020-12-31', 50, '2021-02-20', start='2020-01-01')]}
                },
                'CommonStockSharesOutstanding': {
                    'units': {'shares': [_fact('2020-12-31', 60, '2021-02-20'), _fact('2021-12-31', 49, '2022-02-20')]}
                },
                'OperatingIncomeLoss': {'units': {'EUR': [_fact('2021-12-31', 1, '2022-02-20', start='2021-01-01')]}},
            }
        },
    }


def test_parse_sec_companyfacts(companyfacts):
    table = parse_sec_companyfacts(companyfacts, ticker='EXM')
    assert table['period_end'].tolist() == [pd.Timestamp('2020-12-31'), pd.Timestamp('2021-12-31')]
    assert table['ticker'].unique().tolist() == ['EXM']
    assert table['revenue'].tolist() == [1000, 1200]  # first report, preferred tag, annual duration only
    assert table['net_income'].tolist() == [100, 130]
    assert table['total_assets'].tolist() == [5000, 5500]
    assert table['shares_outstanding'].tolist() == [50, 49]  # weighted diluted first, then period-end count
    assert table['operating_income'].isna().all()  # non-USD units are ignored
    assert table['filed'].tolist() == [pd.Timestamp('2021-02-20'), pd.Timestamp('2022-03-01')]
    assert parse_sec_companyfacts(companyfacts)['ticker'].iloc[0] == 'Example Corp'
    prepared = prepare_fundamentals(
        table.drop(columns=['operating_income']).assign(operating_income=1.0, total_equity=1.0)
    )
    assert len(prepared) == 2


def test_parse_sec_companyfacts_without_annual_data():
    with pytest.raises(MqValueError):
        parse_sec_companyfacts({'facts': {'us-gaap': {}}})


def test_fetch_sec_companyfacts():
    with pytest.raises(MqValueError):
        fetch_sec_companyfacts(320193, user_agent='no contact')
    response = mock.Mock()
    response.json.return_value = {'cik': 320193}
    with mock.patch('requests.get', return_value=response) as get:
        assert fetch_sec_companyfacts(320193, 'Research me@example.com') == {'cik': 320193}
    url = get.call_args.args[0]
    assert url.endswith('CIK0000320193.json')
    assert get.call_args.kwargs['headers'] == {'User-Agent': 'Research me@example.com'}
    response.raise_for_status.assert_called_once()


def test_fetch_sec_fundamentals(companyfacts, caplog):
    tickers = {'0': {'cik_str': 1234, 'ticker': 'EXM', 'title': 'Example Corp'}, '1': {'cik_str': 99, 'ticker': 'IFRS'}}

    def get(url, headers, timeout):
        response = mock.Mock()
        if url.endswith('company_tickers.json'):
            response.json.return_value = tickers
        elif url.endswith('CIK0000001234.json'):
            response.json.return_value = companyfacts
        else:
            response.json.return_value = {'facts': {'ifrs-full': {}}}  # no US GAAP facts
        return response

    with mock.patch('requests.get', side_effect=get) as patched:
        table = fetch_sec_fundamentals(['exm', 'IFRS', 'MISSING'], 'Research me@example.com', pause=0)
    assert table['ticker'].unique().tolist() == ['EXM']
    assert len(table) == 2
    assert patched.call_count == 3  # ticker map + two companies (the unknown ticker is never requested)
    assert 'IFRS, MISSING' in caplog.text
    with mock.patch('requests.get', side_effect=get):
        with pytest.raises(MqValueError):
            fetch_sec_fundamentals(['MISSING'], 'Research me@example.com', pause=0)
