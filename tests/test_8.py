import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from definition_62b718e5e0c145dbbc1ff7758ba7931c import split_data

def create_sample_data():
    X = pd.DataFrame({'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      'feature2': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]})
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    return X, y

@pytest.fixture
def sample_data():
    return create_sample_data()

def test_split_data_valid_split(sample_data):
    X, y = sample_data
    test_size = 0.2
    random_state = 42
    X_train, X_val, y_train, y_val = split_data(X, y, test_size, random_state)
    assert len(X_val) == int(len(X) * test_size)
    assert len(y_val) == int(len(y) * test_size)
    assert len(X_train) == len(X) - int(len(X) * test_size)
    assert len(y_train) == len(y) - int(len(y) * test_size)

def test_split_data_stratified_split(sample_data):
    X, y = sample_data
    test_size = 0.2
    random_state = 42
    X_train, X_val, y_train, y_val = split_data(X, y, test_size, random_state)
    assert y_train.value_counts()[0] == 4
    assert y_train.value_counts()[1] == 4
    assert y_val.value_counts()[0] == 1
    assert y_val.value_counts()[1] == 1

def test_split_data_empty_data():
    X = pd.DataFrame()
    y = pd.Series()
    test_size = 0.2
    random_state = 42
    X_train, X_val, y_train, y_val = split_data(X, y, test_size, random_state)
    assert len(X_train) == 0
    assert len(X_val) == 0
    assert len(y_train) == 0
    assert len(y_val) == 0

def test_split_data_test_size_zero(sample_data):
    X, y = sample_data
    test_size = 0.0
    random_state = 42
    X_train, X_val, y_train, y_val = split_data(X, y, test_size, random_state)
    assert len(X_train) == len(X)
    assert len(y_train) == len(y)
    assert len(X_val) == 0
    assert len(y_val) == 0

def test_split_data_test_size_one(sample_data):
    X, y = sample_data
    test_size = 1.0
    random_state = 42
    X_train, X_val, y_train, y_val = split_data(X, y, test_size, random_state)
    assert len(X_train) == 0
    assert len(y_train) == 0
    assert len(X_val) == len(X)
    assert len(y_val) == len(y)
