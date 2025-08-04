import pytest
import pandas as pd
from definition_01a568b6ccfc413ea880b22de2c9dcd1 import calculate_financial_ratios

@pytest.fixture
def sample_dataframe():
    data = {
        'Net Income': [100, 200, 300, 400, 500],
        'Total Assets': [1000, 2000, 3000, 4000, 5000],
        'Total Debt': [500, 1000, 1500, 2000, 2500],
        'Shareholders\' Equity': [500, 1000, 1500, 2000, 2500],
        'Current Assets': [200, 400, 600, 800, 1000],
        'Current Liabilities': [100, 200, 300, 400, 500],
        'EBITDA': [150, 300, 450, 600, 750],
        'Interest Expense': [50, 100, 150, 200, 250]
    }
    return pd.DataFrame(data)

def test_calculate_financial_ratios_empty_dataframe():
    df = pd.DataFrame()
    result_df = calculate_financial_ratios(df)
    assert isinstance(result_df, pd.DataFrame)
    assert result_df.empty

def test_calculate_financial_ratios_valid_data(sample_dataframe):
    df = sample_dataframe.copy()
    result_df = calculate_financial_ratios(df)

    assert 'ROA' in result_df.columns
    assert 'Debt-to-Equity' in result_df.columns
    assert 'Current Ratio' in result_df.columns
    assert 'Cash Flow Coverage' in result_df.columns

    assert all(result_df['ROA'] == df['Net Income'] / df['Total Assets'])
    assert all(result_df['Debt-to-Equity'] == df['Total Debt'] / df['Shareholders\' Equity'])
    assert all(result_df['Current Ratio'] == df['Current Assets'] / df['Current Liabilities'])
    assert all(result_df['Cash Flow Coverage'] == df['EBITDA'] / df['Interest Expense'])

def test_calculate_financial_ratios_zero_equity(sample_dataframe):
    df = sample_dataframe.copy()
    df['Shareholders\' Equity'] = 0
    result_df = calculate_financial_ratios(df)
    assert all(result_df['Debt-to-Equity'] == float('inf'))

def test_calculate_financial_ratios_zero_liabilities(sample_dataframe):
    df = sample_dataframe.copy()
    df['Current Liabilities'] = 0
    result_df = calculate_financial_ratios(df)
    assert all(result_df['Current Ratio'] == float('inf'))

def test_calculate_financial_ratios_zero_interest_expense(sample_dataframe):
    df = sample_dataframe.copy()
    df['Interest Expense'] = 0
    result_df = calculate_financial_ratios(df)
    assert all(result_df['Cash Flow Coverage'] == float('inf'))
