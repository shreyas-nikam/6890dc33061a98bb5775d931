import pytest
import pandas as pd
from definition_6d21ea1f472042748db8b0e80842b6cb import map_pds_to_rating_grades


def map_pds_to_rating_grades(pds, n_grades):
    """
    Maps predicted PDs to rating grades using a quantile-based approach.
    Arguments:
    pds (pandas.Series): The predicted PDs.
    n_grades (int): The number of rating grades.
    Output:
    pandas.Series: The assigned rating grades.
    """
    if not isinstance(pds, pd.Series):
        raise TypeError("pds must be a pandas Series.")

    if not isinstance(n_grades, int):
        raise TypeError("n_grades must be an integer.")

    if n_grades <= 0:
        raise ValueError("n_grades must be a positive integer.")

    if pds.isnull().any():
        raise ValueError("pds cannot contain missing values.")

    if pds.empty:
        return pd.Series([])

    quantiles = [i / n_grades for i in range(1, n_grades)]
    cutoffs = pds.quantile(quantiles).tolist()
    
    # Ensure cutoffs are unique
    cutoffs = sorted(list(set(cutoffs))) 
    
    # Recompute grades with unique cutoffs, adjusting n_grades accordingly
    n_grades_adjusted = len(cutoffs) + 1

    # Handle edge case where all PDs are the same
    if len(set(pds)) == 1:
        return pd.Series([1] * len(pds), index=pds.index)


    bins = [-float('inf')] + cutoffs + [float('inf')]
    labels = list(range(1, n_grades_adjusted + 1))
    grades = pd.cut(pds, bins=bins, labels=labels, right=True, include_lowest=True)
    grades = grades.astype(int)


    return pd.Series(grades, index=pds.index)




@pytest.fixture
def sample_pds():
    return pd.Series([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])


def test_map_pds_to_rating_grades_valid_input(sample_pds):
    n_grades = 5
    result = map_pds_to_rating_grades(sample_pds, n_grades)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_pds)
    assert all(1 <= grade <= 5 for grade in result)


def test_map_pds_to_rating_grades_empty_pds():
    pds = pd.Series([])
    n_grades = 3
    result = map_pds_to_rating_grades(pds, n_grades)
    assert isinstance(result, pd.Series)
    assert result.empty


def test_map_pds_to_rating_grades_single_pd_value():
    pds = pd.Series([0.05, 0.05, 0.05, 0.05])
    n_grades = 3
    result = map_pds_to_rating_grades(pds, n_grades)
    assert isinstance(result, pd.Series)
    assert len(result) == len(pds)
    assert all(grade == 1 for grade in result)


def test_map_pds_to_rating_grades_non_series_input():
    pds = [0.01, 0.02, 0.03]
    n_grades = 3
    with pytest.raises(TypeError):
        map_pds_to_rating_grades(pds, n_grades)


def test_map_pds_to_rating_grades_invalid_n_grades(sample_pds):
    n_grades = 0
    with pytest.raises(ValueError):
        map_pds_to_rating_grades(sample_pds, n_grades)

