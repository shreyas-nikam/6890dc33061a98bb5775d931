import pytest
from definition_25196d77a607415da69060314e6950e8 import create_data_log
import yaml

def test_create_data_log_empty_transformations(tmp_path):
    d = tmp_path / "test_log.yaml"
    create_data_log({}, filename=str(d))
    with open(d, 'r') as f:
        data = yaml.safe_load(f)
    assert data == {}

def test_create_data_log_single_transformation(tmp_path):
    d = tmp_path / "test_log.yaml"
    transformations = {"transformation_1": {"type": "Imputation", "column": "AGE", "method": "median"}}
    create_data_log(transformations, filename=str(d))
    with open(d, 'r') as f:
        data = yaml.safe_load(f)
    assert data == transformations

def test_create_data_log_multiple_transformations(tmp_path):
    d = tmp_path / "test_log.yaml"
    transformations = {
        "transformation_1": {"type": "Imputation", "column": "AGE", "method": "median"},
        "transformation_2": {"type": "Winsorization", "column": "BILL_AMT1", "limit": 0.05}
    }
    create_data_log(transformations, filename=str(d))
    with open(d, 'r') as f:
        data = yaml.safe_load(f)
    assert data == transformations

def test_create_data_log_no_filename(tmp_path, monkeypatch):
    # Mock the os.getcwd() function to return a temporary directory
    monkeypatch.setattr("os.getcwd", lambda: str(tmp_path))

    create_data_log({"transformation_1": {"type": "Imputation", "column": "AGE", "method": "median"}})

    # Check if the default filename was used and if the file was created
    default_file = tmp_path / "data_log.yaml"
    assert default_file.exists()

    # Verify that the content of the file is correct
    with open(default_file, "r") as f:
        data = yaml.safe_load(f)
        expected_data = {"transformation_1": {"type": "Imputation", "column": "AGE", "method": "median"}}
        assert data == expected_data
