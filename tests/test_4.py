import pytest
from definition_a1953714e1a1458d89283896297e2124 import create_data_log
import yaml
import os

def test_create_data_log_empty_transformations(tmpdir):
    """Test with an empty transformations dictionary."""
    transformations = {}
    filepath = os.path.join(tmpdir, "data_log.yaml")
    create_data_log(transformations, filepath)

    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    assert data == {}

def test_create_data_log_valid_transformations(tmpdir):
    """Test with a valid transformations dictionary."""
    transformations = {
        "step1": "Data cleaning",
        "step2": "Feature engineering",
        "step3": "Model training"
    }
    filepath = os.path.join(tmpdir, "data_log.yaml")
    create_data_log(transformations, filepath)

    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    assert data == transformations

def test_create_data_log_invalid_filepath(tmpdir):
    """Test with an invalid filepath."""
    transformations = {
        "step1": "Data cleaning"
    }
    filepath = os.path.join(tmpdir, "nonexistent_dir", "data_log.yaml")  # Invalid path

    with pytest.raises(FileNotFoundError):
        create_data_log(transformations, filepath)

def test_create_data_log_overwrite_existing_file(tmpdir):
    """Test overwriting an existing file."""
    filepath = os.path.join(tmpdir, "data_log.yaml")
    with open(filepath, 'w') as f:
        f.write("Initial content")

    transformations = {
        "step1": "New transformations"
    }
    create_data_log(transformations, filepath)

    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    assert data == transformations

def test_create_data_log_with_nested_transformations(tmpdir):
    """Test with nested transformations (dictionary within dictionary)."""
    transformations = {
        "step1": "Data cleaning",
        "step2": {
            "feature1": "Engineered feature 1",
            "feature2": "Engineered feature 2"
        }
    }
    filepath = os.path.join(tmpdir, "data_log.yaml")
    create_data_log(transformations, filepath)

    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    assert data == transformations
