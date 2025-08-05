import pytest
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch
from io import StringIO
from definition_aa3867525f324b08a443bfccae29a5c4 import generate_roc_curve
from sklearn.metrics import roc_curve, auc


def _generate_dummy_data():
    y_true = pd.Series([0, 0, 1, 1])
    y_pred = pd.Series([0.1, 0.4, 0.35, 0.8])
    return y_true, y_pred

def test_generate_roc_curve_typical():
    y_true, y_pred = _generate_dummy_data()
    title = "ROC Curve Test"
    fig = generate_roc_curve(y_true, y_pred, title)
    assert isinstance(fig, plt.Figure)


def test_generate_roc_curve_empty_data():
    y_true = pd.Series([])
    y_pred = pd.Series([])
    title = "ROC Curve Empty Data"
    fig = generate_roc_curve(y_true, y_pred, title)
    assert isinstance(fig, plt.Figure)



def test_generate_roc_curve_invalid_input_type_y_true():
    y_true = [0, 1, 0, 1]
    y_pred = pd.Series([0.1, 0.4, 0.35, 0.8])
    title = "ROC Curve Invalid Input Type"
    with pytest.raises(TypeError):
        generate_roc_curve(y_true, y_pred, title)


def test_generate_roc_curve_invalid_input_type_y_pred():
    y_true = pd.Series([0, 0, 1, 1])
    y_pred = [0.1, 0.4, 0.35, 0.8]
    title = "ROC Curve Invalid Input Type"
    with pytest.raises(TypeError):
        generate_roc_curve(y_true, y_pred, title)



def test_generate_roc_curve_mismatched_lengths():
    y_true = pd.Series([0, 1])
    y_pred = pd.Series([0.1, 0.4, 0.8])
    title = "ROC Curve Mismatched Lengths"
    with pytest.raises(ValueError):
        generate_roc_curve(y_true, y_pred, title)

