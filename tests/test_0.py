import pytest
import pandas as pd
from definition_308615dec8ec48c2bbb334ad66800642 import load_dataset

@pytest.fixture
def sample_csv_file(tmp_path):
    # Create a simple CSV file for testing
    csv_file = tmp_path / "test_data.csv"
    csv_file.write_text("col1,col2\n1,2\n3,4\n")
    return str(csv_file)

def test_load_dataset_valid_file(sample_csv_file):
    df = load_dataset(sample_csv_file)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert list(df.columns) == ['col1', 'col2']

def test_load_dataset_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_dataset("nonexistent_file.csv")

def test_load_dataset_empty_file(tmp_path):
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text("")
    df = load_dataset(str(csv_file))
    assert isinstance(df, pd.DataFrame)
    assert df.empty

def test_load_dataset_invalid_csv(tmp_path):
    csv_file = tmp_path / "invalid.csv"
    csv_file.write_text("col1,col2\n1,2\n3")
    with pytest.raises(pd.errors.ParserError):
        load_dataset(str(csv_file))

def test_load_dataset_with_index(tmp_path):
    csv_file = tmp_path / "test_data.csv"
    csv_file.write_text("index,col1,col2\n0,1,2\n1,3,4\n")
    df = load_dataset(str(csv_file))
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 3)
    assert list(df.columns) == ['index', 'col1', 'col2']
