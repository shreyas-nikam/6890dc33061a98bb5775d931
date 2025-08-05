import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from definition_44474519f2f040a3832e116896cf8b39 import split_data

def create_sample_dataframe(rows=100):
    data = {
        'feature1': np.random.rand(rows),
        'feature2': np.random.rand(rows),
        'target': np.random.randint(0, 2, rows)
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_dataframe():
    return create_sample_dataframe()


def test_split_data_basic(sample_dataframe):
    X_train, X_val, y_train, y_val = split_data(sample_dataframe, 'target', 0.2, 42)
    assert len(X_train) == 80
    assert len(X_val) == 20
    assert len(y_train) == 80
    assert len(y_val) == 20
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_val, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_val, pd.Series)

def test_split_data_stratify(sample_dataframe):
    df = sample_dataframe
    target_column = 'target'
    test_size = 0.2
    random_state = 42
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    X_train_custom, X_val_custom, y_train_custom, y_val_custom = split_data(df, target_column, test_size, random_state)
    
    assert (y_train.value_counts(normalize=True).sort_index() == y_train_custom.value_counts(normalize=True).sort_index()).all()
    assert (y_val.value_counts(normalize=True).sort_index() == y_val_custom.value_counts(normalize=True).sort_index()).all()

def test_split_data_test_size_zero(sample_dataframe):
     X_train, X_val, y_train, y_val = split_data(sample_dataframe, 'target', 0.0, 42)
     assert len(X_train) == 100
     assert len(X_val) == 0
     assert len(y_train) == 100
     assert len(y_val) == 0

def test_split_data_test_size_one(sample_dataframe):
    X_train, X_val, y_train, y_val = split_data(sample_dataframe, 'target', 1.0, 42)
    assert len(X_train) == 0
    assert len(X_val) == 100
    assert len(y_train) == 0
    assert len(y_val) == 100

def test_split_data_invalid_target_column(sample_dataframe):
    with pytest.raises(KeyError):
        split_data(sample_dataframe, 'invalid_target', 0.2, 42)
