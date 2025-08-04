import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from definition_8d7a9d8aba0440cf8b447d241b573a30 import split_data


def create_sample_dataframe():
    data = {'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]}
    return pd.DataFrame(data)


def test_split_data_valid_split():
    df = create_sample_dataframe()
    target_column = 'target'
    test_size = 0.2
    random_state = 42
    X_train, X_val, y_train, y_val = split_data(df, target_column, test_size, random_state)

    assert len(X_train) == 8
    assert len(X_val) == 2
    assert len(y_train) == 8
    assert len(y_val) == 2
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_val, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_val, pd.Series)


def test_split_data_stratified_split():
    df = create_sample_dataframe()
    target_column = 'target'
    test_size = 0.2
    random_state = 42
    X_train, X_val, y_train, y_val = split_data(df, target_column, test_size, random_state)

    assert y_train.value_counts()[0] == 4
    assert y_train.value_counts()[1] == 4
    assert y_val.value_counts()[0] == 1
    assert y_val.value_counts()[1] == 1


def test_split_data_invalid_test_size():
    df = create_sample_dataframe()
    target_column = 'target'
    test_size = 1.5  # Invalid test size
    random_state = 42
    with pytest.raises(ValueError):
        split_data(df, target_column, test_size, random_state)


def test_split_data_empty_dataframe():
    df = pd.DataFrame()
    target_column = 'target'
    test_size = 0.2
    random_state = 42
    with pytest.raises(KeyError):
         split_data(df, target_column, test_size, random_state)


def test_split_data_invalid_target_column():
    df = create_sample_dataframe()
    target_column = 'non_existent_target'
    test_size = 0.2
    random_state = 42
    with pytest.raises(KeyError):
        split_data(df, target_column, test_size, random_state)
