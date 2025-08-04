import pytest
from definition_44a0be605cca456abeeeb5e5b020e616 import calibrate_pd_to_observed_default_rates
import numpy as np

@pytest.mark.parametrize("predicted_probabilities, actual_defaults, expected", [
    ([0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]),
    ([0.1, 0.2, 0.3], [0.2, 0.4, 0.6], [0.2, 0.4, 0.6]),
    ([0.1, 0.2, 0.3], [0.05, 0.1, 0.15], [0.05, 0.1, 0.15]),
    ([0.1, 0.2, 0.3], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ([], [], []) # Empty lists

])
def test_calibrate_pd_to_observed_default_rates(predicted_probabilities, actual_defaults, expected):
    # Mock implementation (replace with actual implementation if available)
    if not predicted_probabilities and not actual_defaults:
        assert calibrate_pd_to_observed_default_rates(predicted_probabilities, actual_defaults) == expected
    else:
        calibrated_probabilities = calibrate_pd_to_observed_default_rates(predicted_probabilities, actual_defaults)
        assert calibrated_probabilities == expected
