import pytest
import pandas as pd
import numpy as np
from definition_2e4fbdd3daee4ba18751f9f31ed4a2ed import calculate_auc_gini
from sklearn.metrics import roc_auc_score


def calculate_auc_gini(y_true, y_pred):
    """Calculates AUC and Gini coefficient.
    Arguments:
    y_true (pandas.Series or array-like): The true target values.
    y_pred (pandas.Series or array-like): The predicted probabilities.
    Output:
    auc, gini: The AUC and Gini coefficient.
    """
    auc = roc_auc_score(y_true, y_pred)
    gini = 2 * auc - 1
    return auc, gini


@pytest.mark.parametrize("y_true, y_pred, expected_auc, expected_gini", [
    ([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], 0.75, 0.5),
    ([0, 1], [0, 1], 1.0, 1.0),
    ([1, 0], [0, 1], 1.0, 1.0),
    ([0, 1], [1, 0], 0.0, -1.0),
    ([1, 0], [1, 0], 0.0, -1.0),
    ([0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4], 0.5, 0.0),
    ([1, 1, 1, 1], [0.1, 0.2, 0.3, 0.4], 0.5, 0.0),
    ([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], 1.0, 1.0),
    ([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2], 1.0, 1.0),
    ([0, 1, 0, 1], [0.9, 0.1, 0.8, 0.2], 0.0, -1.0),
    ([1, 0, 1, 0], [0.1, 0.9, 0.2, 0.8], 0.0, -1.0),
    (pd.Series([0, 0, 1, 1]), pd.Series([0.1, 0.4, 0.35, 0.8]), 0.75, 0.5),
    (np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]), 0.75, 0.5),
    ([0, 0, 1, 1, 0, 1], [0.1, 0.3, 0.6, 0.8, 0.2, 0.9], 0.8333333333333334, 0.6666666666666667) # test case
])
def test_calculate_auc_gini(y_true, y_pred, expected_auc, expected_gini):
    auc, gini = calculate_auc_gini(y_true, y_pred)
    assert auc == pytest.approx(expected_auc)
    assert gini == pytest.approx(expected_gini)


def test_calculate_auc_gini_single_class():
    with pytest.raises(ValueError):
        calculate_auc_gini([0, 0, 0], [0.1, 0.2, 0.3])


def test_calculate_auc_gini_different_lengths():
    with pytest.raises(ValueError):
        calculate_auc_gini([0, 1], [0.1, 0.2, 0.3])
