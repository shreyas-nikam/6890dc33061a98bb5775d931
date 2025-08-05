import pytest
import pandas as pd
from definition_94d3f4e75c12400a9ea71d9702f3bde9 import calculate_vif

def create_sample_dataframe(data):
    return pd.DataFrame(data)

@pytest.fixture
def sample_dataframe():
    data = {'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'feature3': [5, 4, 3, 2, 1]}
    return create_sample_dataframe(data)

def test_calculate_vif_no_multicollinearity(sample_dataframe):
    df = sample_dataframe[['feature1', 'feature3']]
    vif_values = calculate_vif(df)
    assert all(v < 5 for v in vif_values.values)

def test_calculate_vif_perfect_multicollinearity(sample_dataframe):
    df = sample_dataframe[['feature1', 'feature2']]
    vif_values = calculate_vif(df)
    assert vif_values['feature1'] == float('inf') or vif_values['feature2'] == float('inf')

def test_calculate_vif_with_multicollinearity(sample_dataframe):
    vif_values = calculate_vif(sample_dataframe)
    assert vif_values['feature2'] > 5

def test_calculate_vif_empty_dataframe():
    df = pd.DataFrame()
    with pytest.raises(ValueError):
        calculate_vif(df)

def test_calculate_vif_non_numeric_data():
    data = {'feature1': ['a', 'b', 'c', 'd', 'e'],
            'feature2': [1, 2, 3, 4, 5]}
    df = create_sample_dataframe(data)
    with pytest.raises(TypeError):
         calculate_vif(df)
