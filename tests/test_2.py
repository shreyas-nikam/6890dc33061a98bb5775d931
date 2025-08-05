import pytest
import pandas as pd
import numpy as np
from io import StringIO
from definition_75b07e1d8b154dcdb129e0f92a02bde0 import impute_missing_values

@pytest.fixture
def sample_dataframe():
    data = {'col1': [1.0, 2.0, np.nan, 4.0],
            'col2': [5.0, np.nan, 7.0, 8.0],
            'col3': [9.0, 10.0, 11.0, 12.0]}
    return pd.DataFrame(data)


def test_impute_median(sample_dataframe):
    df = sample_dataframe.copy()
    columns = ['col1', 'col2']
    imputed_df = impute_missing_values(df, columns, 'median')
    assert imputed_df['col1'].isnull().sum() == 0
    assert imputed_df['col2'].isnull().sum() == 0
    assert imputed_df['col1'][2] == 2.5  # Median of [1, 2, 4]
    assert imputed_df['col2'][1] == 6.0  # Median of [5, 7, 8]

def test_impute_mean(sample_dataframe):
    df = sample_dataframe.copy()
    columns = ['col1']
    imputed_df = impute_missing_values(df, columns, 'mean')
    assert imputed_df['col1'].isnull().sum() == 0
    assert imputed_df['col1'][2] == 7/3  # Mean of [1, 2, 4]

def test_empty_columns_list(sample_dataframe):
    df = sample_dataframe.copy()
    columns = []
    imputed_df = impute_missing_values(df, columns, 'median')
    pd.testing.assert_frame_equal(imputed_df, df)

def test_invalid_strategy(sample_dataframe):
    df = sample_dataframe.copy()
    columns = ['col1']
    with pytest.raises(ValueError):
        impute_missing_values(df, columns, 'invalid_strategy')

def test_empty_dataframe():
    df = pd.DataFrame()
    columns = ['col1']
    imputed_df = impute_missing_values(df, columns, 'median')
    assert imputed_df.empty
