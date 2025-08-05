import pytest
from unittest.mock import MagicMock
from definition_f47c292108f049038a586ab823ef7775 import save_preprocessing_pipeline
import pickle
import os

def test_save_pipeline_success(tmp_path):
    """Test that the pipeline is saved successfully."""
    pipeline_mock = MagicMock()
    filepath = os.path.join(tmp_path, "pipeline.pkl")
    save_preprocessing_pipeline(pipeline_mock, filepath)
    assert os.path.exists(filepath)

def test_save_pipeline_file_content(tmp_path):
    """Test that the saved pipeline can be loaded and is the same."""
    import sklearn.pipeline
    from sklearn.preprocessing import StandardScaler
    pipeline = sklearn.pipeline.Pipeline([('scaler', StandardScaler())])

    filepath = os.path.join(tmp_path, "pipeline.pkl")
    save_preprocessing_pipeline(pipeline, filepath)

    with open(filepath, 'rb') as f:
        loaded_pipeline = pickle.load(f)

    assert isinstance(loaded_pipeline, sklearn.pipeline.Pipeline)
    assert isinstance(loaded_pipeline.steps[0][1], StandardScaler)

def test_save_pipeline_invalid_filepath(tmp_path):
    """Test that the function handles invalid filepaths gracefully."""

    pipeline_mock = MagicMock()
    filepath = os.path.join(tmp_path, "nonexistent_dir", "pipeline.pkl")
    with pytest.raises(FileNotFoundError):
        save_preprocessing_pipeline(pipeline_mock, filepath)

def test_save_pipeline_empty_pipeline(tmp_path):
    """Test that function handles case with empty pipeline."""
    import sklearn.pipeline
    pipeline = sklearn.pipeline.Pipeline([])
    filepath = os.path.join(tmp_path, "pipeline.pkl")
    save_preprocessing_pipeline(pipeline, filepath)
    assert os.path.exists(filepath)
    with open(filepath, 'rb') as f:
        loaded_pipeline = pickle.load(f)
    assert isinstance(loaded_pipeline, sklearn.pipeline.Pipeline)
    assert len(loaded_pipeline.steps) == 0
