import pytest
import pandas as pd
from definition_0b8e2a0b85c3434c8c996b1bddfd11f9 import impute_missing_values


def test_impute_missing_values_median():
    df = pd.DataFrame({'col1': [1, 2, None, 4], 'col2': [5, None, 7, 8]})
    df_imputed = impute_missing_values(df.copy(), strategy='median')
    assert df_imputed['col1'].median() == 2.5
    assert df_imputed['col2'].median() == 6.0
    assert df_imputed['col1'].isnull().sum() == 0
    assert df_imputed['col2'].isnull().sum() == 0
    assert df_imputed['col1'][2] == 2.5
    assert df_imputed['col2'][1] == 6.0


def test_impute_missing_values_empty_dataframe():
    df = pd.DataFrame()
    df_imputed = impute_missing_values(df.copy(), strategy='median')
    assert df_imputed.empty


def test_impute_missing_values_no_missing_values():
    df = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [5, 6, 7, 8]})
    df_imputed = impute_missing_values(df.copy(), strategy='median')
    pd.testing.assert_frame_equal(df, df_imputed)


def test_impute_missing_values_all_missing_values():
    df = pd.DataFrame({'col1': [None, None, None], 'col2': [None, None, None]})
    df_imputed = impute_missing_values(df.copy(), strategy='median')
    assert df_imputed['col1'].isnull().sum() == 3
    assert df_imputed['col2'].isnull().sum() == 3


def test_impute_missing_values_non_numeric_column():
    df = pd.DataFrame({'col1': [1, 2, None, 4], 'col2': ['a', 'b', 'c', 'd']})
    df_imputed = impute_missing_values(df.copy(), strategy='median')
    assert df_imputed['col1'].isnull().sum() == 0
    assert df_imputed['col2'].equals(df['col2'])
