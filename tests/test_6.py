import pytest
import pandas as pd
import numpy as np
from definition_03070cc6b02c4d229ef77f2073d87704 import create_management_experience_score

@pytest.fixture
def sample_dataframe():
    data = {'existing_column': [1, 2, 3, 4, 5]}
    return pd.DataFrame(data)

def test_create_management_experience_score_empty_df():
    df = pd.DataFrame()
    result_df = create_management_experience_score(df)
    assert 'management_experience_score' in result_df.columns
    assert len(result_df) == 0


def test_create_management_experience_score_adds_column(sample_dataframe):
    df = sample_dataframe.copy()
    result_df = create_management_experience_score(df)
    assert 'management_experience_score' in result_df.columns
    assert len(result_df) == len(df)

def test_create_management_experience_score_correct_type(sample_dataframe):
     df = sample_dataframe.copy()
     result_df = create_management_experience_score(df)
     assert result_df['management_experience_score'].dtype == np.dtype('int64')

def test_create_management_experience_score_values_within_range(sample_dataframe):
    df = sample_dataframe.copy()
    result_df = create_management_experience_score(df)
    assert all(0 <= val <= 10 for val in result_df['management_experience_score'])
