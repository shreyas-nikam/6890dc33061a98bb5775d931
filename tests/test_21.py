import pytest
from definition_1697a66305ba4737ae9e2ab89ed0854f import create_model_inventory_record
import yaml
from datetime import date

def test_create_model_inventory_record_valid():
    model_id = "model_001"
    tier = "Tier 1"
    owner = "John Doe"
    validator = "Jane Smith"
    last_validated = date(2023, 1, 1)
    next_due = date(2024, 1, 1)

    record = create_model_inventory_record(model_id, tier, owner, validator, last_validated, next_due)

    assert isinstance(record, str)
    data = yaml.safe_load(record)
    assert data["model_id"] == model_id
    assert data["tier"] == tier
    assert data["owner"] == owner
    assert data["validator"] == validator
    assert data["last_validated"] == last_validated.isoformat()
    assert data["next_due"] == next_due.isoformat()

def test_create_model_inventory_record_empty_values():
    model_id = ""
    tier = ""
    owner = ""
    validator = ""
    last_validated = None
    next_due = None

    record = create_model_inventory_record(model_id, tier, owner, validator, last_validated, next_due)
    data = yaml.safe_load(record)

    assert data["model_id"] == model_id
    assert data["tier"] == tier
    assert data["owner"] == owner
    assert data["validator"] == validator
    assert data["last_validated"] is None
    assert data["next_due"] is None

def test_create_model_inventory_record_none_dates():
    model_id = "model_002"
    tier = "Tier 2"
    owner = "Alice Brown"
    validator = "Bob Williams"
    last_validated = None
    next_due = None

    record = create_model_inventory_record(model_id, tier, owner, validator, last_validated, next_due)

    assert isinstance(record, str)
    data = yaml.safe_load(record)

    assert data["model_id"] == model_id
    assert data["tier"] == tier
    assert data["owner"] == owner
    assert data["validator"] == validator
    assert data["last_validated"] is None
    assert data["next_due"] is None

def test_create_model_inventory_record_invalid_date_type():
    model_id = "model_003"
    tier = "Tier 3"
    owner = "Eve White"
    validator = "Charlie Green"
    last_validated = "2023-01-01"
    next_due = "2024-01-01"

    with pytest.raises(TypeError):
        create_model_inventory_record(model_id, tier, owner, validator, last_validated, next_due)

def test_create_model_inventory_record_special_characters():
    model_id = "model!@#$"
    tier = "Tier %^&"
    owner = "John.Doe"
    validator = "Jane@Smith"
    last_validated = date(2023, 2, 15)
    next_due = date(2024, 2, 15)

    record = create_model_inventory_record(model_id, tier, owner, validator, last_validated, next_due)

    assert isinstance(record, str)
    data = yaml.safe_load(record)
    assert data["model_id"] == model_id
    assert data["tier"] == tier
    assert data["owner"] == owner
    assert data["validator"] == validator
    assert data["last_validated"] == last_validated.isoformat()
    assert data["next_due"] == next_due.isoformat()
