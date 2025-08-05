import pytest
import pandas as pd
import numpy as np
from definition_b4bfe559c3f140ad819da59bc66b9c78 import winsorize_outlier_values

@pytest.fixture
def sample_dataframe():
    data = {'col1': [1, 2, 3, 4, 5, 100], 'col2': [6, 7, 8, 9, 10, -50]}
    return pd.DataFrame(data)

def test_winsorize_no_outliers(sample_dataframe):
    df = sample_dataframe.copy()
    columns = ['col1']
    lower_quantile = 0.05
    upper_quantile = 0.95
    result = winsorize_outlier_values(df, columns, lower_quantile, upper_quantile)
    expected = pd.DataFrame({'col1': [1, 2, 3, 4, 5, 100], 'col2': [6, 7, 8, 9, 10, -50]})
    pd.testing.assert_frame_equal(result, expected)

def test_winsorize_with_outliers(sample_dataframe):
    df = sample_dataframe.copy()
    columns = ['col1', 'col2']
    lower_quantile = 0.1
    upper_quantile = 0.9
    result = winsorize_outlier_values(df, columns, lower_quantile, upper_quantile)
    expected = pd.DataFrame({'col1': [1, 2, 3, 4, 5, 5], 'col2': [6, 7, 8, 9, 10, 6]})
    pd.testing.assert_frame_equal(result, expected)

def test_winsorize_empty_dataframe():
    df = pd.DataFrame()
    columns = ['col1']
    lower_quantile = 0.1
    upper_quantile = 0.9
    result = winsorize_outlier_values(df, columns, lower_quantile, upper_quantile)
    assert isinstance(result, pd.DataFrame)
    assert result.empty

def test_winsorize_empty_columns(sample_dataframe):
    df = sample_dataframe.copy()
    columns = []
    lower_quantile = 0.1
    upper_quantile = 0.9
    result = winsorize_outlier_values(df, columns, lower_quantile, upper_quantile)
    pd.testing.assert_frame_equal(result, df)

def test_winsorize_invalid_quantile(sample_dataframe):
    df = sample_dataframe.copy()
    columns = ['col1']
    lower_quantile = 1.1
    upper_quantile = 0.9
    with pytest.raises(ValueError):
        winsorize_outlier_values(df, columns, lower_quantile, upper_quantile)
