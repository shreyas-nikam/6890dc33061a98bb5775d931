import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from definition_b1464fb0c2154f399f7a8c1529919eb0 import generate_hosmer_lemeshow_plot


def create_dummy_data(size):
    y_true = np.random.randint(0, 2, size=size)
    y_prob = np.random.rand(size)
    return pd.Series(y_true), pd.Series(y_prob)


def test_generate_hosmer_lemeshow_plot_valid_input():
    y_true, y_prob = create_dummy_data(100)
    n_bins = 10
    title = "Hosmer-Lemeshow Plot"
    fig = generate_hosmer_lemeshow_plot(y_true, y_prob, n_bins, title)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_generate_hosmer_lemeshow_plot_empty_input():
    y_true = pd.Series([])
    y_prob = pd.Series([])
    n_bins = 10
    title = "Hosmer-Lemeshow Plot"
    fig = generate_hosmer_lemeshow_plot(y_true, y_prob, n_bins, title)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_generate_hosmer_lemeshow_plot_mismatched_lengths():
    y_true, _ = create_dummy_data(100)
    y_prob, = create_dummy_data(50)  # Different length
    n_bins = 10
    title = "Hosmer-Lemeshow Plot"
    with pytest.raises(ValueError):
        generate_hosmer_lemeshow_plot(y_true, y_prob, n_bins, title)

def test_generate_hosmer_lemeshow_plot_invalid_bins():
    y_true, y_prob = create_dummy_data(100)
    n_bins = 0 #invalid number of bins
    title = "Hosmer-Lemeshow Plot"
    with pytest.raises(ValueError):
        generate_hosmer_lemeshow_plot(y_true, y_prob, n_bins, title)

def test_generate_hosmer_lemeshow_plot_y_prob_out_of_range():
    y_true = pd.Series([0, 1, 0, 1])
    y_prob = pd.Series([-0.1, 1.2, 0.5, 0.8])  # Probabilities outside [0, 1]
    n_bins = 4
    title = "Hosmer-Lemeshow Plot"
    with pytest.raises(ValueError):
        generate_hosmer_lemeshow_plot(y_true, y_prob, n_bins, title)
