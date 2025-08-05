import pytest
import pickle
import os
from definition_0d586863768745a1afe6d9061354dff1 import save_model

def dummy_model():
    return {'model_type': 'dummy'}

@pytest.fixture
def temp_file_path(tmpdir):
    return os.path.join(tmpdir, "test_model.pkl")

def test_save_model_success(temp_file_path):
    model = dummy_model()
    save_model(model, temp_file_path)
    assert os.path.exists(temp_file_path)
    with open(temp_file_path, 'rb') as f:
        loaded_model = pickle.load(f)
    assert loaded_model['model_type'] == 'dummy'

def test_save_model_invalid_model(temp_file_path):
    with pytest.raises(TypeError):
        save_model("not a model", temp_file_path)

def test_save_model_empty_file_path():
    model = dummy_model()
    with pytest.raises(ValueError):
        save_model(model, "")

def test_save_model_none_file_path():
    model = dummy_model()
    with pytest.raises(ValueError):
        save_model(model, None)
