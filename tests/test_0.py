import pytest
import pandas as pd
from definition_98103d3a9ae9431dbf767aa44c80eb96 import load_data

def test_load_data_valid_csv(tmp_path):
    d = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data=d)
    p = tmp_path / "test.csv"
    df.to_csv(p, index=False)
    loaded_df = load_data(str(p))
    pd.testing.assert_frame_equal(loaded_df, df)

def test_load_data_empty_csv(tmp_path):
    p = tmp_path / "empty.csv"
    with open(p, 'w') as f:
        f.write('')
    with pytest.raises(pd.errors.EmptyDataError):
        load_data(str(p))

def test_load_data_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        load_data("nonexistent_file.csv")

def test_load_data_invalid_path_type():
    with pytest.raises(TypeError):
        load_data(123)

def test_load_data_corrupted_csv(tmp_path):
    p = tmp_path / "corrupted.csv"
    with open(p, 'w') as f:
        f.write("col1,col2\n1,2\n3")
    with pytest.raises(pd.errors.ParserError):
        load_data(str(p))
