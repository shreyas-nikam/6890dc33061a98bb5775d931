import pytest
import pandas as pd
import numpy as np
from definition_0e18457fd0c446ae825a6b61dc0caa00 import calibrate_pds

def calibrate_pds(pds, actual_default_rates):
    """Calibrates PDs to observed default rates.

    Args:
        pds (pandas.Series): The predicted PDs.
        actual_default_rates (pandas.Series): The actual default rates.

    Returns:
        pandas.Series: The calibrated PDs.
    """

    # Simple implementation: Adjust all PDs by a constant factor
    # More sophisticated calibration methods (e.g., isotonic regression) could be used
    # This is a simplified approach for demonstration purposes

    if pds.empty or actual_default_rates.empty:
        return pds  # Return original PDs if either series is empty

    overall_predicted_default_rate = pds.mean()
    overall_actual_default_rate = actual_default_rates.mean()

    if overall_predicted_default_rate == 0:
        return pds.apply(lambda x: 0.0)

    calibration_factor = overall_actual_default_rate / overall_predicted_default_rate

    calibrated_pds = pds * calibration_factor
    calibrated_pds = calibrated_pds.clip(0, 1)  # Ensure PDs remain between 0 and 1

    return calibrated_pds


@pytest.fixture
def sample_pds():
    return pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])

@pytest.fixture
def sample_actual_default_rates():
    return pd.Series([0.02, 0.03, 0.04, 0.05, 0.06])


def test_calibrate_pds_basic(sample_pds, sample_actual_default_rates):
    calibrated_pds = calibrate_pds(sample_pds, sample_actual_default_rates)
    assert isinstance(calibrated_pds, pd.Series)
    assert all(0 <= p <= 1 for p in calibrated_pds)

def test_calibrate_pds_empty_input():
    pds = pd.Series([])
    actual_default_rates = pd.Series([])
    calibrated_pds = calibrate_pds(pds, actual_default_rates)
    assert calibrated_pds.empty

def test_calibrate_pds_zero_predicted_defaults(sample_actual_default_rates):
    pds = pd.Series([0.0, 0.0, 0.0, 0.0])
    calibrated_pds = calibrate_pds(pds, sample_actual_default_rates)
    assert all(p == 0.0 for p in calibrated_pds)

def test_calibrate_pds_large_values():
    pds = pd.Series([0.5, 0.6, 0.7, 0.8, 0.9])
    actual_default_rates = pd.Series([0.1, 0.2, 0.1, 0.2, 0.1])
    calibrated_pds = calibrate_pds(pds, actual_default_rates)
    assert all(0 <= p <= 1 for p in calibrated_pds)

def test_calibrate_pds_negative_values():
    pds = pd.Series([-0.1, 0.2, 0.3])
    actual_default_rates = pd.Series([0.1, 0.2, 0.3])
    calibrated_pds = calibrate_pds(pds, actual_default_rates)
    assert all(0 <= p <= 1 for p in calibrated_pds)
