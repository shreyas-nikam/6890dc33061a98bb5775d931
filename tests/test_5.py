import pytest
import pandas as pd
import numpy as np
from definition_56a89734f25348c185a31e1219720623 import calculate_financial_ratios

@pytest.fixture
def sample_dataframe():
    data = {
        'Net Income': [100, 200, 300, 400],
        'Total Assets': [1000, 2000, 3000, 4000],
        'Total Debt': [500, 1000, 1500, 2000],
        'Shareholders\' Equity': [500, 1000, 1500, 2000],
        'Current Assets': [600, 1200, 1800, 2400],
        'Current Liabilities': [300, 600, 900, 1200],
        'EBITDA': [150, 300, 450, 600],
        'Interest Expense': [50, 100, 150, 200]
    }
    return pd.DataFrame(data)

def test_calculate_financial_ratios_empty_dataframe():
    df = pd.DataFrame()
    with pytest.raises(ValueError):
        calculate_financial_ratios(df)

def test_calculate_financial_ratios_valid_data(sample_dataframe):
    result_df = calculate_financial_ratios(sample_dataframe)
    assert isinstance(result_df, pd.DataFrame)
    assert 'ROA' in result_df.columns
    assert 'Debt-to-Equity Ratio' in result_df.columns
    assert 'Current Ratio' in result_df.columns
    assert 'Cash Flow Coverage Ratio' in result_df.columns

    assert np.allclose(result_df['ROA'], [0.1, 0.1, 0.1, 0.1])
    assert np.allclose(result_df['Debt-to-Equity Ratio'], [1.0, 1.0, 1.0, 1.0])
    assert np.allclose(result_df['Current Ratio'], [2.0, 2.0, 2.0, 2.0])
    assert np.allclose(result_df['Cash Flow Coverage Ratio'], [3.0, 3.0, 3.0, 3.0])

def test_calculate_financial_ratios_zero_equity(sample_dataframe):
    sample_dataframe['Shareholders\' Equity'] = 0
    result_df = calculate_financial_ratios(sample_dataframe)
    assert np.all(np.isinf(result_df['Debt-to-Equity Ratio']))
    
def test_calculate_financial_ratios_zero_assets(sample_dataframe):
    sample_dataframe['Total Assets'] = 0
    result_df = calculate_financial_ratios(sample_dataframe)
    assert np.all(np.isinf(result_df['ROA']))
