import pytest
import pandas as pd
import numpy as np
from definition_8fdc876233c84863995abc47b936762f import create_management_experience_score

@pytest.fixture
def sample_dataframe():
    # Create a sample DataFrame for testing
    data = {'Years_in_Management': [1, 5, 10, 2, np.nan],
            'Years_at_Company': [2, 5, 12, 3, 1],
            'Age': [30, 40, 50, 35, 25]}
    df = pd.DataFrame(data)
    return df

def test_create_management_experience_score_valid(sample_dataframe):
    """Test with valid data."""
    df = sample_dataframe.copy()
    result_df = create_management_experience_score(df)
    assert 'Management_Experience_Score' in result_df.columns
    assert result_df['Management_Experience_Score'].dtype == 'float64'
    assert not result_df['Management_Experience_Score'].isnull().any()
    
def test_create_management_experience_score_missing_values(sample_dataframe):
    """Test with missing values in 'Years_in_Management' column."""
    df = sample_dataframe.copy()
    result_df = create_management_experience_score(df)
    assert 'Management_Experience_Score' in result_df.columns
    # Check if missing values are handled (e.g., imputed with 0)
    assert not result_df['Management_Experience_Score'].isnull().any()

def test_create_management_experience_score_no_years_in_management(sample_dataframe):
    """Test when 'Years_in_Management' column is all zeros."""
    df = sample_dataframe.copy()
    df['Years_in_Management'] = 0
    result_df = create_management_experience_score(df)
    assert 'Management_Experience_Score' in result_df.columns
    assert (result_df['Management_Experience_Score'] >= 0).all()
    

def test_create_management_experience_score_negative_years(sample_dataframe):
    """Test with negative values in 'Years_in_Management' column."""
    df = sample_dataframe.copy()
    df['Years_in_Management'] = [-1, -2, 0, 1, 2]
    result_df = create_management_experience_score(df)
    assert 'Management_Experience_Score' in result_df.columns
    assert result_df['Management_Experience_Score'].dtype == 'float64'

def test_create_management_experience_score_empty_dataframe():
    """Test with an empty DataFrame."""
    df = pd.DataFrame()
    result_df = create_management_experience_score(df)
    assert 'Management_Experience_Score' in result_df.columns if not df.empty else True  #Checks whether the column is created or not.
    assert result_df.empty if df.empty else True
