import pytest
import pandas as pd
from definition_87e0a0b5e30e4586bcff06fcee9e44fb import winsorize_outlier_values


@pytest.fixture
def sample_dataframe():
    data = {'col1': [1, 2, 3, 4, 100],
            'col2': [5, 6, 7, 8, -100],
            'col3': [9, 10, 11, 12, 13]}
    return pd.DataFrame(data)


def test_winsorize_outlier_values_basic(sample_dataframe):
    df = sample_dataframe.copy()
    columns = ['col1', 'col2']
    limits = (0.1, 0.9)  # 10th and 90th percentile
    result_df = winsorize_outlier_values(df, columns, limits)

    col1_min = df['col1'].quantile(0.1)
    col1_max = df['col1'].quantile(0.9)
    col2_min = df['col2'].quantile(0.1)
    col2_max = df['col2'].quantile(0.9)

    assert result_df['col1'].max() <= col1_max
    assert result_df['col2'].min() >= col2_min


def test_winsorize_outlier_values_no_outliers(sample_dataframe):
    df = sample_dataframe.copy()
    columns = ['col3']
    limits = (0.1, 0.9)
    original_df = df.copy()
    result_df = winsorize_outlier_values(df, columns, limits)
    pd.testing.assert_frame_equal(original_df,result_df)

def test_winsorize_outlier_values_empty_columns(sample_dataframe):
    df = sample_dataframe.copy()
    columns = []
    limits = (0.1, 0.9)
    original_df = df.copy()
    result_df = winsorize_outlier_values(df, columns, limits)
    pd.testing.assert_frame_equal(original_df,result_df)

def test_winsorize_outlier_values_invalid_limits(sample_dataframe):
     df = sample_dataframe.copy()
     columns = ['col1', 'col2']
     limits = (0.9, 0.1)
     with pytest.raises(ValueError):
        winsorize_outlier_values(df, columns, limits)

def test_winsorize_outlier_values_single_column(sample_dataframe):
    df = sample_dataframe.copy()
    columns = ['col1']
    limits = (0.1, 0.9)
    result_df = winsorize_outlier_values(df, columns, limits)

    col1_min = df['col1'].quantile(0.1)
    col1_max = df['col1'].quantile(0.9)

    assert result_df['col1'].max() <= col1_max