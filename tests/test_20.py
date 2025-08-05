import pytest
import pandas as pd
from definition_5840cd79decb462683db6fd3f71f1e3b import save_data
import os

def test_save_data_valid_data(tmp_path):
    file_path = tmp_path / "test_data.csv"
    data = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data)
    save_data(df, file_path)
    assert os.path.exists(file_path)

def test_save_data_empty_dataframe(tmp_path):
    file_path = tmp_path / "test_data.csv"
    df = pd.DataFrame()
    save_data(df, file_path)
    assert os.path.exists(file_path)

def test_save_data_different_data_types(tmp_path):
    file_path = tmp_path / "test_data.csv"
    data = {'col1': [1, 2], 'col2': ['a', 'b'], 'col3': [1.1, 2.2]}
    df = pd.DataFrame(data)
    save_data(df, file_path)
    assert os.path.exists(file_path)

def test_save_data_non_dataframe_input(tmp_path):
    file_path = tmp_path / "test_data.csv"
    data = [1, 2, 3]
    with pytest.raises(TypeError):
        save_data(data, file_path)

def test_save_data_invalid_file_path(tmp_path):
    # Test with a directory path instead of a file path
    file_path = tmp_path
    data = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data)
    with pytest.raises(ValueError):  # Expecting ValueError since we are checking if file_path is a string.
        save_data(df, file_path)
