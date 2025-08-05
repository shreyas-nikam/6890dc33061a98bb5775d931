import pytest
from definition_2a7e8ab335474533ab71b71739144941 import calculate_auc_gini
import numpy as np
from sklearn.metrics import roc_auc_score

def calculate_auc_gini_impl(y_true, y_pred):
    """Calculates AUC and Gini coefficient using sklearn."""
    auc = roc_auc_score(y_true, y_pred)
    gini = 2 * auc - 1
    return auc, gini

@pytest.mark.parametrize("y_true, y_pred, expected_auc, expected_gini", [
    ([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], 0.75, 0.5),
    ([0, 1], [0, 1], 1.0, 1.0),
    ([0, 1], [1, 0], 0.0, -1.0),
    ([1, 1, 1], [0.1, 0.2, 0.3], 0.5, 0.0),  # All positive class
    ([0, 0, 0], [0.1, 0.2, 0.3], 0.5, 0.0),  # All negative class
])
def test_calculate_auc_gini(y_true, y_pred, expected_auc, expected_gini):
    auc, gini = calculate_auc_gini_impl(y_true, y_pred)
    assert np.isclose(auc, expected_auc)
    assert np.isclose(gini, expected_gini)

