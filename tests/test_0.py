import pytest
import pandas as pd
from definition_d79e99cefa9045a5a2f3a4a056da23b1 import load_data
import os

def test_load_data_valid_csv(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test.csv"
    p.write_text("col1,col2\n1,2\n3,4")

    df = load_data(str(p))
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert list(df.columns) == ['col1', 'col2']


def test_load_data_empty_csv(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "empty.csv"
    p.write_text("")
    
    df = load_data(str(p))
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_load_data_invalid_path():
    with pytest.raises(FileNotFoundError):
        load_data("nonexistent_file.csv")


def test_load_data_different_delimiter(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test_pipe.csv"
    p.write_text("col1|col2\n1|2\n3|4")

    df = load_data(str(p))
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 1) #incorrectly parsed

def test_load_data_no_header(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "no_header.csv"
    p.write_text("1,2\n3,4")

    df = load_data(str(p))
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2,1) #incorrectly parsed
