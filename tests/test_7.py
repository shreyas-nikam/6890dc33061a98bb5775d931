import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from definition_187fd05159f64d9c80e3206abc1461e5 import transform_skewed_variables

@pytest.fixture
def sample_dataframe():
    data = {'col1': [1, 2, 3, 4, 5, 100], 'col2': [6, 7, 8, 9, 10, 200], 'col3': [11, 12, 13, 14, 15, 300]}
    return pd.DataFrame(data)

def test_transform_skewed_variables_basic(sample_dataframe):
    df = transform_skewed_variables(sample_dataframe.copy(), ['col1'], 3)
    assert df['col1'].nunique() == 3
    assert df['col1'].dtype == 'int64'

def test_transform_skewed_variables_multiple_columns(sample_dataframe):
    df = transform_skewed_variables(sample_dataframe.copy(), ['col1', 'col2'], 4)
    assert df['col1'].nunique() == 4
    assert df['col2'].nunique() == 4

def test_transform_skewed_variables_no_columns(sample_dataframe):
    df = transform_skewed_variables(sample_dataframe.copy(), [], 5)
    assert df.equals(sample_dataframe)

def test_transform_skewed_variables_large_n_bins(sample_dataframe):
    df = transform_skewed_variables(sample_dataframe.copy(), ['col1'], 10)
    assert df['col1'].nunique() == len(sample_dataframe['col1'].unique())

def test_transform_skewed_variables_empty_dataframe():
    df = pd.DataFrame()
    df_transformed = transform_skewed_variables(df.copy(), ['col1'], 3)
    assert df_transformed.equals(df)
