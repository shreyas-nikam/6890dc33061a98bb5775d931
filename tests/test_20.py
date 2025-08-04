import pytest
import pandas as pd
from definition_09a950706f234da9add93f6307332489 import segment_obligors


@pytest.fixture
def sample_dataframe():
    data = {
        'industry': ['Manufacturing', 'Services', 'Manufacturing', 'Services', 'Manufacturing'],
        'size': ['Small', 'Large', 'Large', 'Small', 'Small'],
        'other_col': [1, 2, 3, 4, 5]
    }
    return pd.DataFrame(data)


def test_segment_obligors_basic(sample_dataframe):
    df = segment_obligors(sample_dataframe.copy(), 'industry', 'size')
    assert 'industry_segment' in df.columns
    assert 'size_segment' in df.columns
    assert df['industry_segment'].tolist() == ['Manufacturing', 'Services', 'Manufacturing', 'Services', 'Manufacturing']
    assert df['size_segment'].tolist() == ['Small', 'Large', 'Large', 'Small', 'Small']


def test_segment_obligors_empty_dataframe():
    df = pd.DataFrame()
    df = segment_obligors(df, 'industry', 'size')
    assert 'industry_segment' in df.columns
    assert 'size_segment' in df.columns
    assert len(df) == 0


def test_segment_obligors_different_column_names(sample_dataframe):
    df = segment_obligors(sample_dataframe.copy(), 'other_col', 'industry')
    assert 'industry_segment' in df.columns
    assert 'size_segment' in df.columns
    assert df['industry_segment'].tolist() == [1, 2, 3, 4, 5]
    assert df['size_segment'].tolist() == ['Manufacturing', 'Services', 'Manufacturing', 'Services', 'Manufacturing']


def test_segment_obligors_missing_columns(sample_dataframe):
    with pytest.raises(KeyError):
        segment_obligors(sample_dataframe.copy(), 'nonexistent_column', 'size')


def test_segment_obligors_already_exists(sample_dataframe):
   df = sample_dataframe.copy()
   df['industry_segment'] = 'Existing'
   df['size_segment'] = 'Existing'
   df = segment_obligors(df, 'industry', 'size')
   assert df['industry_segment'].tolist() == ['Manufacturing', 'Services', 'Manufacturing', 'Services', 'Manufacturing']
   assert df['size_segment'].tolist() == ['Small', 'Large', 'Large', 'Small', 'Small']
