import pytest
import pandas as pd
import numpy as np
from definition_e754a86084d04ebd8d245284561820ca import perform_data_quality_checks

def create_sample_dataframe(missing_values=False, outliers=False):
    data = {'col1': [1, 2, 3, 4, 5],
            'col2': [6, 7, 8, 9, 10],
            'col3': [11, 12, 13, 14, 15]}
    df = pd.DataFrame(data)

    if missing_values:
        df.loc[1, 'col1'] = np.nan
        df.loc[3, 'col2'] = np.nan
    if outliers:
        df.loc[0, 'col3'] = 100  # Introduce an outlier
    return df


def perform_data_quality_checks(df):
    """Performs data quality checks for missing values and outliers.

    Arguments:
        df (pandas.DataFrame): The input DataFrame.

    Output:
        pandas.DataFrame: The DataFrame with data quality issues reported.
    """
    df_copy = df.copy()

    # Missing Value Checks
    missing_values = df_copy.isnull().sum()
    for col in df_copy.columns:
        if missing_values[col] > 0:
            df_copy.loc[-1, f'{col}_missing'] = missing_values[col]  # add new row with missing counts


    # Outlier Checks (using a simple IQR method)
    for col in df_copy.columns:
        if pd.api.types.is_numeric_dtype(df_copy[col]):  # Check if the column is numeric
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df_copy[(df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)][col].count()
            if outliers > 0:
                 df_copy.loc[-1, f'{col}_outliers'] = outliers

    return df_copy

def test_no_issues():
    df = create_sample_dataframe()
    result_df = perform_data_quality_checks(df)
    assert result_df.shape == df.shape, "Shape should be the same as original df"


def test_missing_values():
    df = create_sample_dataframe(missing_values=True)
    result_df = perform_data_quality_checks(df)
    assert 'col1_missing' in result_df.columns, "Missing values in col1 should be reported"
    assert 'col2_missing' in result_df.columns, "Missing values in col2 should be reported"
    assert result_df.iloc[-1]['col1_missing'] == 1.0
    assert result_df.iloc[-1]['col2_missing'] == 1.0


def test_outliers():
    df = create_sample_dataframe(outliers=True)
    result_df = perform_data_quality_checks(df)
    assert 'col3_outliers' in result_df.columns, "Outliers in col3 should be reported"
    assert result_df.iloc[-1]['col3_outliers'] == 1.0

def test_missing_and_outliers():
    df = create_sample_dataframe(missing_values=True, outliers=True)
    result_df = perform_data_quality_checks(df)
    assert 'col1_missing' in result_df.columns, "Missing values in col1 should be reported"
    assert 'col2_missing' in result_df.columns, "Missing values in col2 should be reported"
    assert 'col3_outliers' in result_df.columns, "Outliers in col3 should be reported"
    assert result_df.iloc[-1]['col1_missing'] == 1.0
    assert result_df.iloc[-1]['col2_missing'] == 1.0
    assert result_df.iloc[-1]['col3_outliers'] == 1.0

def test_empty_dataframe():
    df = pd.DataFrame()
    result_df = perform_data_quality_checks(df)
    assert result_df.empty == True, "Should return an empty DataFrame"
