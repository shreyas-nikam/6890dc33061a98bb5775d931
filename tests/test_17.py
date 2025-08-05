import pytest
from definition_d16a47e0e801448bbd279549e959fef9 import calibrate_pds
import numpy as np

def test_calibrate_pds_perfect_calibration():
    predicted_probabilities = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    actual_default_rates = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    calibrated_probabilities = calibrate_pds(predicted_probabilities, actual_default_rates)
    assert np.allclose(calibrated_probabilities, actual_default_rates)

def test_calibrate_pds_underestimation():
    predicted_probabilities = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    actual_default_rates = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
    calibrated_probabilities = calibrate_pds(predicted_probabilities, actual_default_rates)
    assert np.allclose(calibrated_probabilities, actual_default_rates)

def test_calibrate_pds_overestimation():
    predicted_probabilities = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
    actual_default_rates = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    calibrated_probabilities = calibrate_pds(predicted_probabilities, actual_default_rates)
    assert np.allclose(calibrated_probabilities, actual_default_rates)

def test_calibrate_pds_empty_input():
    predicted_probabilities = np.array([])
    actual_default_rates = np.array([])
    calibrated_probabilities = calibrate_pds(predicted_probabilities, actual_default_rates)
    assert len(calibrated_probabilities) == 0

def test_calibrate_pds_different_lengths():
    predicted_probabilities = np.array([0.1, 0.2])
    actual_default_rates = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        calibrate_pds(predicted_probabilities, actual_default_rates)
