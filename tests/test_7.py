import pytest
import pandas as pd
import numpy as np
from definition_455a9e6ce0d3411e9fe72062bae326c7 import transform_skewed_variables

@pytest.fixture
def sample_dataframe():
    data = {'col1': [1, 2, 3, 4, 5],
            'col2': [10, 20, 30, 40, 50],
            'col3': [100, 200, 300, 400, 500]}
    return pd.DataFrame(data)

def test_transform_skewed_variables_empty_columns(sample_dataframe):
    df = sample_dataframe.copy()
    transformed_df = transform_skewed_variables(df, [], 5)
    pd.testing.assert_frame_equal(transformed_df, df)

def test_transform_skewed_variables_single_column(sample_dataframe):
    df = sample_dataframe.copy()
    transformed_df = transform_skewed_variables(df, ['col1'], 3)
    assert 'col1' in transformed_df.columns
    assert transformed_df['col1'].dtype == 'int64'
    assert transformed_df['col1'].nunique() <= 3

def test_transform_skewed_variables_multiple_columns(sample_dataframe):
    df = sample_dataframe.copy()
    transformed_df = transform_skewed_variables(df, ['col1', 'col2'], 4)
    assert 'col1' in transformed_df.columns
    assert 'col2' in transformed_df.columns
    assert transformed_df['col1'].dtype == 'int64'
    assert transformed_df['col2'].dtype == 'int64'
    assert transformed_df['col1'].nunique() <= 4
    assert transformed_df['col2'].nunique() <= 4

def test_transform_skewed_variables_invalid_n_bins(sample_dataframe):
    df = sample_dataframe.copy()
    with pytest.raises(ValueError):
        transform_skewed_variables(df, ['col1'], 0)

def test_transform_skewed_variables_non_numeric_column(sample_dataframe):
    df = sample_dataframe.copy()
    df['col4'] = ['a', 'b', 'c', 'd', 'e']
    transformed_df = transform_skewed_variables(df, ['col4'], 3)
    pd.testing.assert_frame_equal(transformed_df, df)
