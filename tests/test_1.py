import pytest
import pandas as pd
from definition_12bb630a23564091a9571f6468b08fde import perform_data_quality_checks

@pytest.fixture
def sample_dataframe():
    data = {'col1': [1, 2, None, 4, 5],
            'col2': [6, 7, 8, 9, 1000],  # Outlier
            'col3': ['a', 'b', 'c', 'd', 'e']}
    return pd.DataFrame(data)


def test_perform_data_quality_checks_no_issues(sample_dataframe):
    df = sample_dataframe.copy()
    df['col1'] = [1,2,3,4,5]
    df['col2'] = [6,7,8,9,10]
    result = perform_data_quality_checks(df)
    assert isinstance(result, pd.DataFrame)


def test_perform_data_quality_checks_missing_values(sample_dataframe):
    df = sample_dataframe.copy()
    result = perform_data_quality_checks(df)
    assert isinstance(result, pd.DataFrame)


def test_perform_data_quality_checks_outliers(sample_dataframe):
    df = sample_dataframe.copy()
    result = perform_data_quality_checks(df)
    assert isinstance(result, pd.DataFrame)


def test_perform_data_quality_checks_empty_dataframe():
    df = pd.DataFrame()
    result = perform_data_quality_checks(df)
    assert isinstance(result, pd.DataFrame)

def test_perform_data_quality_checks_mixed_datatypes(sample_dataframe):
        df = sample_dataframe.copy()
        result = perform_data_quality_checks(df)
        assert isinstance(result, pd.DataFrame)
