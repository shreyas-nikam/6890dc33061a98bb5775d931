import pytest
import pandas as pd
from definition_3933a5ce9d5b464689ca9913aa9d217a import impute_missing_values

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'A': [1, 2, None, 4, 5],
        'B': [6, None, 8, 9, 10],
        'C': [11, 12, 13, 14, None]
    })

def test_impute_median(sample_dataframe):
    imputed_df = impute_missing_values(sample_dataframe.copy(), strategy='median')
    assert imputed_df['A'].isna().sum() == 0
    assert imputed_df['B'].isna().sum() == 0
    assert imputed_df['C'].isna().sum() == 0
    assert imputed_df['A'][2] == 3.0
    assert imputed_df['B'][1] == 8.5
    assert imputed_df['C'][4] == 13.0

def test_impute_mean(sample_dataframe):
    imputed_df = impute_missing_values(sample_dataframe.copy(), strategy='mean')
    assert imputed_df['A'].isna().sum() == 0
    assert imputed_df['B'].isna().sum() == 0
    assert imputed_df['C'].isna().sum() == 0
    assert imputed_df['A'][2] == 3.0
    assert imputed_df['B'][1] == 8.25
    assert imputed_df['C'][4] == 12.5
    
def test_empty_dataframe():
    df = pd.DataFrame()
    imputed_df = impute_missing_values(df.copy(), strategy='median')
    assert imputed_df.empty

def test_no_missing_values(sample_dataframe):
    df = sample_dataframe.dropna()
    imputed_df = impute_missing_values(df.copy(), strategy='median')
    pd.testing.assert_frame_equal(df, imputed_df)

def test_invalid_strategy(sample_dataframe):
    with pytest.raises(ValueError):
        impute_missing_values(sample_dataframe.copy(), strategy='invalid')

