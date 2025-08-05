import pytest
import numpy as np
import matplotlib.pyplot as plt
from definition_7874e9b86bdd497eaa7436897d573ab4 import generate_calibration_plot

def create_dummy_data(size):
    y_true = np.random.randint(0, 2, size=size)
    y_prob = np.random.rand(size)
    return y_true, y_prob

@pytest.fixture
def dummy_data():
    return create_dummy_data(100)

def test_generate_calibration_plot_valid_input(dummy_data):
    y_true, y_prob = dummy_data
    try:
        generate_calibration_plot(y_true, y_prob, n_bins=10, title="Calibration Plot")
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")

def test_generate_calibration_plot_empty_input():
    y_true = np.array([])
    y_prob = np.array([])
    try:
        generate_calibration_plot(y_true, y_prob, n_bins=5, title="Empty Calibration Plot")
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")

def test_generate_calibration_plot_invalid_n_bins(dummy_data):
    y_true, y_prob = dummy_data
    with pytest.raises(ValueError):
        generate_calibration_plot(y_true, y_prob, n_bins=0, title="Invalid n_bins")

def test_generate_calibration_plot_prob_not_in_range(dummy_data):
     y_true, _ = dummy_data
     y_prob = np.random.uniform(1.1, 2, size = 100)
     with pytest.raises(ValueError):
         generate_calibration_plot(y_true, y_prob, n_bins = 10, title = "Invalid prob range")

def test_generate_calibration_plot_different_lengths():
    y_true = np.array([0, 1, 0])
    y_prob = np.array([0.2, 0.8])
    with pytest.raises(ValueError):
        generate_calibration_plot(y_true, y_prob, n_bins=5, title="Different Lengths")
