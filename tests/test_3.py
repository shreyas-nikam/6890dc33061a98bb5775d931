import pytest
import pandas as pd
import numpy as np
from definition_e44c18eddde44852a99f6a97b4f504d5 import winsorize_outlier_values


def test_winsorize_outlier_values_empty_dataframe():
    df = pd.DataFrame()
    columns = ['col1']
    limits = (0.05, 0.95)
    expected_df = pd.DataFrame()
    result_df = winsorize_outlier_values(df.copy(), columns, limits)
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_winsorize_outlier_values_no_outliers():
    df = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
    columns = ['col1']
    limits = (0.05, 0.95)
    expected_df = df.copy()
    result_df = winsorize_outlier_values(df.copy(), columns, limits)
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_winsorize_outlier_values_with_outliers():
    df = pd.DataFrame({'col1': [1, 2, 3, 4, 100, -100]})
    columns = ['col1']
    limits = (0.1, 0.9)
    expected_df = pd.DataFrame({'col1': [1, 2, 3, 4, 4, 1]})
    result_df = winsorize_outlier_values(df.copy(), columns, limits)

    # Winsorization may lead to slight floating-point differences.  Comparing dataframes with tolerance
    pd.testing.assert_frame_equal(result_df, expected_df, atol=1e-5)


def test_winsorize_outlier_values_multiple_columns():
    df = pd.DataFrame({'col1': [1, 2, 100], 'col2': [-100, 2, 3]})
    columns = ['col1', 'col2']
    limits = (0.1, 0.9)
    expected_df = pd.DataFrame({'col1': [1, 2, 2], 'col2': [2, 2, 3]})
    result_df = winsorize_outlier_values(df.copy(), columns, limits)
    pd.testing.assert_frame_equal(result_df, expected_df, atol=1e-5)



def test_winsorize_outlier_values_invalid_limits():
    df = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
    columns = ['col1']
    limits = (0.95, 0.05)
    with pytest.raises(ValueError):
        winsorize_outlier_values(df.copy(), columns, limits)

