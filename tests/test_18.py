import pytest
from definition_de01274d570e4e66ac79fb3200e60555 import create_model_inventory_record
import yaml
import datetime

def is_valid_yaml(yaml_string):
    try:
        yaml.safe_load(yaml_string)
        return True
    except yaml.YAMLError:
        return False

def test_create_model_inventory_record_valid_data():
    record = create_model_inventory_record(
        model_id="model_123",
        model_tier="Tier 1",
        owner="John Doe",
        validator="Jane Smith",
        last_validated=datetime.date(2023, 1, 1),
        next_due=datetime.date(2024, 1, 1)
    )
    assert is_valid_yaml(record)

def test_create_model_inventory_record_empty_strings():
    record = create_model_inventory_record(
        model_id="",
        model_tier="",
        owner="",
        validator="",
        last_validated=None,
        next_due=None
    )
    assert is_valid_yaml(record)

def test_create_model_inventory_record_none_values():
    record = create_model_inventory_record(
        model_id=None,
        model_tier=None,
        owner=None,
        validator=None,
        last_validated=None,
        next_due=None
    )
    assert is_valid_yaml(record)

def test_create_model_inventory_record_special_characters():
    record = create_model_inventory_record(
        model_id="model!@#",
        model_tier="Tier%$^",
        owner="John&Doe",
        validator="Jane*Smith",
        last_validated=datetime.date(2023, 1, 1),
        next_due=datetime.date(2024, 1, 1)
    )
    assert is_valid_yaml(record)

def test_create_model_inventory_record_future_dates():
     record = create_model_inventory_record(
        model_id="model_123",
        model_tier="Tier 1",
        owner="John Doe",
        validator="Jane Smith",
        last_validated=datetime.date(2025, 1, 1),
        next_due=datetime.date(2026, 1, 1)
    )
     assert is_valid_yaml(record)
