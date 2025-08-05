import pytest
import pandas as pd
import numpy as np
from definition_fd61d1f1dc7f49ca8c4a4a10b15b8c86 import create_management_experience_score

@pytest.fixture
def sample_dataframe():
    # Create a sample DataFrame for testing
    data = {'LIMIT_BAL': [20000, 30000, 40000, 50000, 60000],
            'EDUCATION': [1, 2, 3, 1, 2],
            'MARRIAGE': [1, 2, 1, 2, 1],
            'AGE': [25, 30, 35, 40, 45],
            'PAY_0': [2, 0, -1, 0, 1],
            'PAY_2': [2, 0, -1, 0, 0],
            'PAY_3': [2, 0, -1, 0, 0],
            'PAY_4': [2, 0, -1, 0, 0],
            'PAY_5': [2, 0, -1, 0, 0],
            'PAY_6': [2, 0, -1, 0, 0]}
    df = pd.DataFrame(data)
    return df

def test_create_management_experience_score_exists():
    assert callable(create_management_experience_score)

def test_create_management_experience_score_adds_column(sample_dataframe):
    df = sample_dataframe.copy()
    df_with_score = create_management_experience_score(df)
    assert 'management_experience_score' in df_with_score.columns

def test_create_management_experience_score_values_are_numeric(sample_dataframe):
    df = sample_dataframe.copy()
    df_with_score = create_management_experience_score(df)
    assert pd.api.types.is_numeric_dtype(df_with_score['management_experience_score'])

def test_create_management_experience_score_handles_empty_dataframe():
    empty_df = pd.DataFrame()
    df_with_score = create_management_experience_score(empty_df)
    assert 'management_experience_score' in df_with_score.columns
    assert len(df_with_score) == 0
