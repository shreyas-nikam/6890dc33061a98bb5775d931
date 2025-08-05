import pytest
import pandas as pd
from definition_f8ad2dbe246e4bde9db171e4ca91d7c6 import save_rating_grade_cutoffs
import os

def create_dummy_csv(filepath):
    with open(filepath, 'w') as f:
        f.write("Grade,Cutoff\n")
        f.write("AAA,0.01\n")
        f.write("AA,0.02\n")

def read_dummy_csv(filepath):
    with open(filepath, 'r') as f:
        return f.read()


@pytest.fixture
def cleanup_file():
    filepath = "test_cutoffs.csv"
    if os.path.exists(filepath):
        os.remove(filepath)
    yield filepath  # Provide the filepath to the test
    if os.path.exists(filepath):
        os.remove(filepath)



def test_save_rating_grade_cutoffs_valid_data(cleanup_file):
    filepath = cleanup_file
    cutoffs = pd.Series([0.01, 0.02, 0.03], index=['AAA', 'AA', 'A'])
    save_rating_grade_cutoffs(cutoffs, filepath)
    assert os.path.exists(filepath)

    #verify contents
    with open(filepath, 'r') as f:
        content = f.read()

    expected_content = "Grade,Cutoff\nAAA,0.01\nAA,0.02\nA,0.03\n"
    assert content == expected_content


def test_save_rating_grade_cutoffs_empty_series(cleanup_file):
    filepath = cleanup_file
    cutoffs = pd.Series([])
    save_rating_grade_cutoffs(cutoffs, filepath)
    assert os.path.exists(filepath)

    #verify contents
    with open(filepath, 'r') as f:
        content = f.read()
    expected_content = "Grade,Cutoff\n"
    assert content == expected_content



def test_save_rating_grade_cutoffs_non_string_index(cleanup_file):
    filepath = cleanup_file
    cutoffs = pd.Series([0.01, 0.02], index=[1, 2])  # Non-string index
    save_rating_grade_cutoffs(cutoffs, filepath)
    assert os.path.exists(filepath)
    #verify contents
    with open(filepath, 'r') as f:
        content = f.read()
    expected_content = "Grade,Cutoff\n1,0.01\n2,0.02\n"
    assert content == expected_content

def test_save_rating_grade_cutoffs_existing_file(cleanup_file):
    filepath = cleanup_file
    create_dummy_csv(filepath)  # Create the file before the test
    initial_content = read_dummy_csv(filepath)
    cutoffs = pd.Series([0.03, 0.04], index=['BBB', 'BB'])
    save_rating_grade_cutoffs(cutoffs, filepath)

    with open(filepath, 'r') as f:
        content = f.read()
    expected_content = "Grade,Cutoff\nBBB,0.03\nBB,0.04\n"
    assert content == expected_content  # File should be overwritten


def test_save_rating_grade_cutoffs_invalid_filepath():
    filepath = "/invalid/path/cutoffs.csv"
    cutoffs = pd.Series([0.01], index=['AAA'])
    with pytest.raises(FileNotFoundError):
        save_rating_grade_cutoffs(cutoffs, filepath)
