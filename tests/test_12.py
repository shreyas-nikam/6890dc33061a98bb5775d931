import pytest
import pandas as pd
from definition_556e5bb41a114c79bfba067bffdfb091 import calculate_vif_values
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def calculate_vif_values(X):
    """Calculates VIF values for multicollinearity assessment.

    Args:
        X (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.Series: A series of VIF values.
    """
    # Ensure X is a DataFrame
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    # Add a constant to the DataFrame to account for the intercept
    X = add_constant(X)

    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return vif_data.set_index("feature")["VIF"]


@pytest.fixture
def sample_dataframe():
    data = {'col1': [1, 2, 3, 4, 5],
            'col2': [2, 4, 6, 8, 10],
            'col3': [1, 3, 5, 7, 9]}
    return pd.DataFrame(data)

def test_calculate_vif_values_typical(sample_dataframe):
    vif_values = calculate_vif_values(sample_dataframe)
    assert isinstance(vif_values, pd.Series)
    assert len(vif_values) == 4
    assert vif_values.index.name == 'feature'

def test_calculate_vif_values_no_multicollinearity():
    data = {'col1': [1, 2, 3, 4, 5],
            'col2': [5, 4, 3, 2, 1],
            'col3': [2, 4, 6, 8, 10]}
    df = pd.DataFrame(data)
    vif_values = calculate_vif_values(df)
    assert all(vif_values < 10)

def test_calculate_vif_values_single_column():
    data = {'col1': [1, 2, 3, 4, 5]}
    df = pd.DataFrame(data)
    vif_values = calculate_vif_values(df)
    assert isinstance(vif_values, pd.Series)
    assert len(vif_values) == 2
    assert vif_values.loc['const'] == 1.0

def test_calculate_vif_values_empty_dataframe():
    df = pd.DataFrame()
    with pytest.raises(ValueError):
        calculate_vif_values(df)

def test_calculate_vif_values_non_dataframe_input():
    with pytest.raises(TypeError):
        calculate_vif_values([1, 2, 3])
