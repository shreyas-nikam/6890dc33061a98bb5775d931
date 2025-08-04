import pytest
import pandas as pd
import numpy as np
from definition_662a5e994c0d464daffeee643ea123f7 import calculate_vif
from statsmodels.stats.outliers_influence import variance_inflation_factor


def calculate_vif_mine(X):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    return vif_data.set_index("feature")["VIF"]

@pytest.fixture
def sample_dataframe():
    data = {'col1': [1, 2, 3, 4, 5],
            'col2': [2, 4, 5, 4, 5],
            'col3': [3, 5, 7, 6, 2]}
    return pd.DataFrame(data)

def test_calculate_vif_valid_input(sample_dataframe):
    try:
        result = calculate_vif(sample_dataframe)
        expected = calculate_vif_mine(sample_dataframe)
        assert isinstance(result, pd.Series)
        assert result.index.tolist() == sample_dataframe.columns.tolist()
        np.testing.assert_allclose(result.values, expected.values)
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")


def test_calculate_vif_perfect_correlation():
    data = {'col1': [1, 2, 3, 4, 5],
            'col2': [2, 4, 6, 8, 10]}
    df = pd.DataFrame(data)
    with pytest.raises(ValueError):
        calculate_vif(df)

def test_calculate_vif_constant_column():
    data = {'col1': [1, 1, 1, 1, 1],
            'col2': [2, 4, 6, 8, 10]}
    df = pd.DataFrame(data)
    with pytest.raises(ValueError):
        calculate_vif(df)

def test_calculate_vif_single_column():
    data = {'col1': [1, 2, 3, 4, 5]}
    df = pd.DataFrame(data)
    result = calculate_vif(df)
    assert isinstance(result, pd.Series)
    assert result['col1'] == 1.0

def test_calculate_vif_empty_dataframe():
    df = pd.DataFrame()
    with pytest.raises(ValueError):
        calculate_vif(df)
