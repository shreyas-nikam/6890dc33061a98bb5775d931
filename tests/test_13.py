import pytest
import matplotlib.pyplot as plt
from unittest.mock import patch
from definition_b921150ccb3a4047aa3790a558a84123 import generate_roc_curve
import numpy as np


@patch("matplotlib.pyplot.show")
def test_generate_roc_curve_valid_input(mock_show):
    y_true = [0, 0, 1, 1]
    y_pred = [0.1, 0.4, 0.35, 0.8]
    title = "Test ROC Curve"
    generate_roc_curve(y_true, y_pred, title)
    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_generate_roc_curve_perfect_prediction(mock_show):
    y_true = [0, 0, 1, 1]
    y_pred = [0.1, 0.2, 0.8, 0.9]
    title = "Perfect Prediction ROC Curve"
    generate_roc_curve(y_true, y_pred, title)
    assert mock_show.called


def test_generate_roc_curve_invalid_input_length():
    y_true = [0, 1]
    y_pred = [0.1, 0.2, 0.3]
    title = "Invalid Input Length"
    with pytest.raises(Exception):
        generate_roc_curve(y_true, y_pred, title)


def test_generate_roc_curve_empty_input():
    y_true = []
    y_pred = []
    title = "Empty Input ROC Curve"
    with pytest.raises(Exception):
        generate_roc_curve(y_true, y_pred, title)


@patch("matplotlib.pyplot.show")
def test_generate_roc_curve_same_predictions(mock_show):
    y_true = [0, 0, 1, 1]
    y_pred = [0.5, 0.5, 0.5, 0.5]
    title = "Same predictions"
    generate_roc_curve(y_true, y_pred, title)
    assert mock_show.called
