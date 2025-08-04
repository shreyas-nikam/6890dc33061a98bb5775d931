import pytest
from definition_366d8d13c36e44968fd15ded4371597c import save_model_and_pipeline
import pickle
import os

@pytest.fixture
def dummy_model():
    return "This is a dummy model"

@pytest.fixture
def dummy_pipeline():
    return "This is a dummy pipeline"

@pytest.fixture
def model_path(tmpdir):
    return os.path.join(tmpdir, "dummy_model.pkl")

@pytest.fixture
def pipeline_path(tmpdir):
    return os.path.join(tmpdir, "dummy_pipeline.pkl")

def test_save_model_and_pipeline_success(dummy_model, dummy_pipeline, model_path, pipeline_path):
    save_model_and_pipeline(dummy_model, dummy_pipeline, model_path, pipeline_path)
    assert os.path.exists(model_path)
    assert os.path.exists(pipeline_path)

def test_save_model_and_pipeline_content(dummy_model, dummy_pipeline, model_path, pipeline_path):
    save_model_and_pipeline(dummy_model, dummy_pipeline, model_path, pipeline_path)
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    with open(pipeline_path, 'rb') as f:
        loaded_pipeline = pickle.load(f)

    assert loaded_model == dummy_model
    assert loaded_pipeline == dummy_pipeline

def test_save_model_and_pipeline_empty_model(dummy_pipeline, model_path, pipeline_path):
    save_model_and_pipeline(None, dummy_pipeline, model_path, pipeline_path)
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    assert loaded_model is None

def test_save_model_and_pipeline_empty_pipeline(dummy_model, model_path, pipeline_path):
    save_model_and_pipeline(dummy_model, None, model_path, pipeline_path)
    with open(pipeline_path, 'rb') as f:
        loaded_pipeline = pickle.load(f)
    assert loaded_pipeline is None