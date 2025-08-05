import pytest
from definition_a663463700ad4eebb91e98c462f8dddb import create_model_inventory_record
import yaml
import os

def create_dummy_yaml(filepath, data):
    with open(filepath, 'w') as f:
        yaml.dump(data, f)

@pytest.fixture
def cleanup_yaml():
    filepath = "test_inventory.yaml"
    yield filepath
    if os.path.exists(filepath):
        os.remove(filepath)

def test_create_model_inventory_record_valid(cleanup_yaml):
    filepath = cleanup_yaml
    model_id = "model_001"
    tier = "Tier 1"
    owner = "John Doe"
    validator = "Jane Smith"
    last_validated = "2023-01-01"
    next_due = "2024-01-01"
    
    create_model_inventory_record(model_id, tier, owner, validator, last_validated, next_due, filepath)
    
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    
    assert data['model_id'] == model_id
    assert data['tier'] == tier
    assert data['owner'] == owner
    assert data['validator'] == validator
    assert data['last_validated'] == last_validated
    assert data['next_due'] == next_due

def test_create_model_inventory_record_empty_values(cleanup_yaml):
    filepath = cleanup_yaml
    model_id = ""
    tier = ""
    owner = ""
    validator = ""
    last_validated = ""
    next_due = ""
    
    create_model_inventory_record(model_id, tier, owner, validator, last_validated, next_due, filepath)
    
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    
    assert data['model_id'] == model_id
    assert data['tier'] == tier
    assert data['owner'] == owner
    assert data['validator'] == validator
    assert data['last_validated'] == last_validated
    assert data['next_due'] == next_due

def test_create_model_inventory_record_special_chars(cleanup_yaml):
    filepath = cleanup_yaml
    model_id = "model_@#$"
    tier = "Tier !@#"
    owner = "John Doe!@#"
    validator = "Jane Smith!@#"
    last_validated = "2023-01-01!@#"
    next_due = "2024-01-01!@#"
    
    create_model_inventory_record(model_id, tier, owner, validator, last_validated, next_due, filepath)
    
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    
    assert data['model_id'] == model_id
    assert data['tier'] == tier
    assert data['owner'] == owner
    assert data['validator'] == validator
    assert data['last_validated'] == last_validated
    assert data['next_due'] == next_due

def test_create_model_inventory_record_long_strings(cleanup_yaml):
    filepath = cleanup_yaml
    model_id = "model_" + "a" * 200
    tier = "Tier " + "b" * 200
    owner = "John " + "c" * 200
    validator = "Jane " + "d" * 200
    last_validated = "2023-01-01" + "e" * 200
    next_due = "2024-01-01" + "f" * 200
    
    create_model_inventory_record(model_id, tier, owner, validator, last_validated, next_due, filepath)
    
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    
    assert data['model_id'] == model_id
    assert data['tier'] == tier
    assert data['owner'] == owner
    assert data['validator'] == validator
    assert data['last_validated'] == last_validated
    assert data['next_due'] == next_due

def test_create_model_inventory_record_unicode(cleanup_yaml):
    filepath = cleanup_yaml
    model_id = "模型ID"
    tier = "第一层"
    owner = "约翰·多伊"
    validator = "简·史密斯"
    last_validated = "2023年1月1日"
    next_due = "2024年1月1日"

    create_model_inventory_record(model_id, tier, owner, validator, last_validated, next_due, filepath)

    with open(filepath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    assert data['model_id'] == model_id
    assert data['tier'] == tier
    assert data['owner'] == owner
    assert data['validator'] == validator
    assert data['last_validated'] == last_validated
    assert data['next_due'] == next_due

