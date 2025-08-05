import pytest
import pandas as pd
from definition_b863c679c00c421da3742e9fb5e99e1c import map_pd_to_rating_grades

@pytest.fixture
def mock_series():
    return pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

def test_map_pd_to_rating_grades_basic(mock_series):
    num_grades = 3
    grades = map_pd_to_rating_grades(mock_series, num_grades)
    assert isinstance(grades, pd.Series)
    assert len(grades) == len(mock_series)
    assert set(grades.unique()) == {0, 1, 2}

def test_map_pd_to_rating_grades_different_num_grades(mock_series):
    num_grades = 5
    grades = map_pd_to_rating_grades(mock_series, num_grades)
    assert set(grades.unique()) == {0, 1, 2, 3, 4}

def test_map_pd_to_rating_grades_empty_series():
    empty_series = pd.Series([])
    num_grades = 4
    grades = map_pd_to_rating_grades(empty_series, num_grades)
    assert isinstance(grades, pd.Series)
    assert len(grades) == 0

def test_map_pd_to_rating_grades_single_value():
    single_series = pd.Series([0.5])
    num_grades = 2
    grades = map_pd_to_rating_grades(single_series, num_grades)
    assert set(grades.unique()) == {0} or set(grades.unique()) == {1}

def test_map_pd_to_rating_grades_same_values():
    same_series = pd.Series([0.5, 0.5, 0.5])
    num_grades = 3
    grades = map_pd_to_rating_grades(same_series, num_grades)
    assert len(grades.unique()) <=3
