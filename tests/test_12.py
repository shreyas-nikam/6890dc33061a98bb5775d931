import pytest
import numpy as np
from definition_5db7cdc74bc64a1f82083e6c3d0475dc import calculate_auc_gini
from sklearn.metrics import roc_auc_score

def test_calculate_auc_gini_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.8, 0.9])
    auc, gini = calculate_auc_gini(y_true, y_pred)
    assert np.isclose(auc, 1.0)
    assert np.isclose(gini, 1.0)

def test_calculate_auc_gini_random():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.5, 0.5, 0.5, 0.5])
    auc, gini = calculate_auc_gini(y_true, y_pred)
    assert np.isclose(auc, 0.5)
    assert np.isclose(gini, 0.0)

def test_calculate_auc_gini_basic():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0.2, 0.8, 0.3, 0.7])
    auc, gini = calculate_auc_gini(y_true, y_pred)
    assert np.isclose(auc, 1.0)
    assert np.isclose(gini, 1.0)

def test_calculate_auc_gini_identical_predictions():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0.7, 0.7, 0.7, 0.7])
    auc, gini = calculate_auc_gini(y_true, y_pred)
    assert np.isclose(auc, 0.5)
    assert np.isclose(gini, 0.0)

def test_calculate_auc_gini_sklearn_comparison():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0.2, 0.8, 0.3, 0.7])
    auc, gini = calculate_auc_gini(y_true, y_pred)
    auc_sklearn = roc_auc_score(y_true, y_pred)
    assert np.isclose(auc, auc_sklearn)
    assert np.isclose(gini, 2 * auc_sklearn - 1)
