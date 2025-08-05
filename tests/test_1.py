import pytest
import pandas as pd
import numpy as np
from definition_38a6ccd524864cd0af957318118b320f import perform_data_quality_checks

@pytest.fixture
def sample_dataframe():
    data = {'col1': [1, 2, np.nan, 4, 5],
            'col2': [6, 7, 8, 9, 1000],
            'col3': ['a', 'b', 'c', 'd', 'e']}
    return pd.DataFrame(data)

def test_no_missing_values(sample_dataframe):
    df = sample_dataframe.copy()
    df = df.dropna()
    result_df = perform_data_quality_checks(df)
    assert result_df.isna().sum().sum() == 0

def test_missing_values_handled(sample_dataframe):
     df = sample_dataframe.copy()
     result_df = perform_data_quality_checks(df)
     assert result_df['col1'].isna().sum() == 0

def test_outliers_handled(sample_dataframe):
    df = sample_dataframe.copy()
    result_df = perform_data_quality_checks(df)
    assert result_df['col2'].max() <= 3 * result_df['col2'].std()

def test_empty_dataframe():
    df = pd.DataFrame()
    result_df = perform_data_quality_checks(df)
    assert result_df.empty

def test_non_numeric_data_type(sample_dataframe):
    df = sample_dataframe.copy()
    with pytest.raises(TypeError):
        perform_data_quality_checks(df)

