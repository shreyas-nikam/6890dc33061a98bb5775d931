import pytest
from definition_56ade0322efc4921804d6526c45ba969 import map_pd_to_rating_grades

@pytest.mark.parametrize("predicted_probabilities, num_grades, expected", [
    ([0.1, 0.2, 0.3, 0.4, 0.5], 3, [0, 0, 1, 1, 2]),
    ([0.1, 0.2, 0.3, 0.4, 0.5], 5, [0, 1, 2, 3, 4]),
    ([0.1, 0.1, 0.1, 0.1, 0.1], 3, [0, 0, 1, 1, 2]),
    ([0.9, 0.9, 0.9, 0.9, 0.9], 3, [0, 0, 1, 1, 2]),
    ([0.1, 0.5, 0.9], 4, [0, 1, 2]), # num_grades > len(predicted_probabilities)
])
def test_map_pd_to_rating_grades(predicted_probabilities, num_grades, expected):
    assert map_pd_to_rating_grades(predicted_probabilities, num_grades) == expected
