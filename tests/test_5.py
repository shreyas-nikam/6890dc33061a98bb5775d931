import pytest
import pandas as pd
from definition_5e8acd5481f64cc98cf1a2cfe43d3f85 import calculate_financial_ratios

@pytest.fixture
def sample_dataframe():
    data = {
        'Net Income': [100000, 50000, 250000],
        'Total Assets': [1000000, 500000, 2500000],
        'Total Debt': [500000, 250000, 1250000],
        'Shareholders\' Equity': [500000, 250000, 1250000],
        'Current Assets': [200000, 100000, 500000],
        'Current Liabilities': [100000, 50000, 250000],
        'EBITDA': [150000, 75000, 375000],
        'Interest Expense': [15000, 7500, 37500]
    }
    return pd.DataFrame(data)


def test_calculate_financial_ratios_valid_input(sample_dataframe):
    df = calculate_financial_ratios(sample_dataframe)
    assert isinstance(df, pd.DataFrame)
    assert 'ROA' in df.columns
    assert 'Debt-to-Equity' in df.columns
    assert 'Current Ratio' in df.columns
    assert 'Cash Flow Coverage' in df.columns

    assert all(df['ROA'] == sample_dataframe['Net Income'] / sample_dataframe['Total Assets'])
    assert all(df['Debt-to-Equity'] == sample_dataframe['Total Debt'] / sample_dataframe['Shareholders\' Equity'])
    assert all(df['Current Ratio'] == sample_dataframe['Current Assets'] / sample_dataframe['Current Liabilities'])
    assert all(df['Cash Flow Coverage'] == sample_dataframe['EBITDA'] / sample_dataframe['Interest Expense'])


def test_calculate_financial_ratios_empty_dataframe():
    df = pd.DataFrame()
    result_df = calculate_financial_ratios(df)
    assert isinstance(result_df, pd.DataFrame)
    assert result_df.empty


def test_calculate_financial_ratios_zero_equity(sample_dataframe):
    sample_dataframe.loc[0, 'Shareholders\' Equity'] = 0
    df = calculate_financial_ratios(sample_dataframe)
    assert isinstance(df, pd.DataFrame)
    assert 'Debt-to-Equity' in df.columns
    assert df['Debt-to-Equity'][0] == float('inf')  # Check for handling division by zero


def test_calculate_financial_ratios_zero_liabilities(sample_dataframe):
    sample_dataframe.loc[0, 'Current Liabilities'] = 0
    df = calculate_financial_ratios(sample_dataframe)
    assert isinstance(df, pd.DataFrame)
    assert 'Current Ratio' in df.columns
    assert df['Current Ratio'][0] == float('inf')

def test_calculate_financial_ratios_zero_interest_expense(sample_dataframe):
    sample_dataframe.loc[0, 'Interest Expense'] = 0
    df = calculate_financial_ratios(sample_dataframe)
    assert isinstance(df, pd.DataFrame)
    assert 'Cash Flow Coverage' in df.columns
    assert df['Cash Flow Coverage'][0] == float('inf')
