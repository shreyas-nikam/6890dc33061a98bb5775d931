import pytest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from definition_12f07533255646fea44804ddb728dfc8 import train_logistic_regression


def test_train_logistic_regression_valid_input():
    X_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y_train = pd.Series([0, 1, 0])
    penalty = 'l2'
    C = 1.0
    random_state = 42
    model = train_logistic_regression(X_train, y_train, penalty, C, random_state)
    assert isinstance(model, LogisticRegression)
    assert model.penalty == penalty
    assert model.C == C
    assert model.random_state == random_state


def test_train_logistic_regression_empty_dataframe():
    X_train = pd.DataFrame()
    y_train = pd.Series([])
    penalty = 'l1'
    C = 0.5
    random_state = 123
    model = train_logistic_regression(X_train, y_train, penalty, C, random_state)
    assert isinstance(model, LogisticRegression)


def test_train_logistic_regression_invalid_penalty():
    X_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y_train = pd.Series([0, 1, 0])
    penalty = 'invalid_penalty'
    C = 1.0
    random_state = 42
    with pytest.raises(ValueError):
        train_logistic_regression(X_train, y_train, penalty, C, random_state)


def test_train_logistic_regression_large_c():
    X_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y_train = pd.Series([0, 1, 0])
    penalty = 'l2'
    C = 1000.0
    random_state = 42
    model = train_logistic_regression(X_train, y_train, penalty, C, random_state)
    assert isinstance(model, LogisticRegression)
    assert model.C == C


def test_train_logistic_regression_different_random_state():
    X_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y_train = pd.Series([0, 1, 0])
    penalty = 'l1'
    C = 1.0
    random_state = 100
    model = train_logistic_regression(X_train, y_train, penalty, C, random_state)
    assert isinstance(model, LogisticRegression)
    assert model.random_state == random_state
