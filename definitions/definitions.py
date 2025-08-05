import pandas as pd

def load_data(csv_path):
    """Loads the dataset from a CSV file.
    Args:
        csv_path (str): The path to the CSV file.
    Returns:
        pandas.DataFrame: The loaded dataset.
    """
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError
    except pd.errors.ParserError:
        raise pd.errors.ParserError
    except TypeError:
        raise TypeError

import pandas as pd
import numpy as np

def perform_data_quality_checks(df):
    """Performs data quality checks for missing values and outliers.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.DataFrame: The DataFrame with data quality issues reported.
    """
    df_copy = df.copy()

    # Missing Value Checks
    missing_values = df_copy.isnull().sum()
    for col in df_copy.columns:
        if missing_values[col] > 0:
            df_copy.loc[-1, f'{col}_missing'] = missing_values[col]  # add new row with missing counts
            df_copy.index = df_copy.index + 1
            df_copy = df_copy.sort_index()


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
                 df_copy.index = df_copy.index + 1
                 df_copy = df_copy.sort_index()

    return df_copy

import pandas as pd
import numpy as np

def impute_missing_values(df, columns, strategy):
    """Imputes missing values using a specified strategy.
    Args:
        df (pandas.DataFrame): The input DataFrame.
        columns (list): A list of columns to impute.
        strategy (str): The imputation strategy ('median', 'mean').
    Returns:
        pandas.DataFrame: The DataFrame with missing values imputed.
    """
    if df.empty:
        return df
    
    if not columns:
        return df

    for col in columns:
        if strategy == 'median':
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
        elif strategy == 'mean':
            mean_val = df[col].mean()
            df[col].fillna(mean_val, inplace=True)
        else:
            raise ValueError("Invalid imputation strategy. Choose 'median' or 'mean'.")
    return df

import pandas as pd
import numpy as np


def winsorize_outlier_values(df, columns, limits):
    """Winsorizes outlier values in specified columns."""

    if limits[0] >= limits[1]:
        raise ValueError("Lower limit must be less than upper limit.")

    for column in columns:
        if column not in df.columns:
            continue

        lower_limit = df[column].quantile(limits[0])
        upper_limit = df[column].quantile(limits[1])

        df[column] = np.where(df[column] < lower_limit, lower_limit, df[column])
        df[column] = np.where(df[column] > upper_limit, upper_limit, df[column])

    return df

def create_data_log(transformations):
    """Creates a data log recording all data transformations.
    Args:
        transformations (dict): A dictionary of data transformations.
    Output:
        dict: A dictionary representing the data log.
    """
    return transformations

import pandas as pd

def calculate_financial_ratios(df):
    """Calculates financial ratios (ROA, Debt-to-Equity, etc.)."""

    if df.empty:
        return df

    df['ROA'] = df['Net Income'] / df['Total Assets']
    df['Debt-to-Equity'] = df['Total Debt'] / df['Shareholders\' Equity']
    df['Current Ratio'] = df['Current Assets'] / df['Current Liabilities']
    df['Cash Flow Coverage'] = df['EBITDA'] / df['Interest Expense']

    return df

import pandas as pd
import numpy as np

def create_management_experience_score(df):
    """Creates a synthetic 'Management Experience Score'.
    Args:
        df (pandas.DataFrame): The input DataFrame.
    Output:
        pandas.DataFrame: The DataFrame with the management experience score added.
    """
    if df.empty:
        df['Management_Experience_Score'] = []
        return df
    
    df['Years_in_Management'] = df['Years_in_Management'].fillna(0)
    
    # Ensure Years_in_Management is non-negative
    df['Years_in_Management'] = df['Years_in_Management'].clip(lower=0)

    df['Management_Experience_Score'] = (df['Years_in_Management'] * 0.6 +
                                           df['Years_at_Company'] * 0.3 +
                                           df['Age'] * 0.1)
    return df

import pandas as pd

def transform_skewed_variables(df, columns, n_bins):
    """Transforms skewed variables using quantile binning.
    Args:
        df (pandas.DataFrame): The input DataFrame.
        columns (list): A list of columns to transform.
        n_bins (int): The number of bins for quantile binning.
    Returns:
        pandas.DataFrame: The DataFrame with skewed variables transformed.
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.qcut(df[col], q=n_bins, labels=False, duplicates='drop').astype('int64')
    return df

import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df, target_column, test_size, random_state):
    """Splits the dataset into training and validation sets (stratified split)."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_val, y_train, y_val

import pandas as pd
            from sklearn.linear_model import LogisticRegression


            def train_logistic_regression(X_train, y_train, penalty, C, random_state):
                """Trains a logistic regression model."""
                model = LogisticRegression(penalty=penalty, C=C, random_state=random_state, solver='liblinear')
                model.fit(X_train, y_train)
                return model

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

def train_gradient_boosted_trees(X_train, y_train, n_estimators, learning_rate, max_depth, random_state):
    """Trains a gradient-boosted trees model."""
    model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    return model

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train, n_estimators, max_depth, random_state):
    """Trains a random forest model."""

    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("Input X_train must be a pandas DataFrame")
    if not isinstance(y_train, pd.Series):
        raise TypeError("Input y_train must be a pandas Series")
    if X_train.empty or y_train.empty:
        raise ValueError("DataFrame is empty")

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    return model

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def calculate_vif_values(X):
    """Calculates VIF values for multicollinearity assessment.

    Args:
        X (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.Series: A series of VIF values.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if X.empty:
        raise ValueError("Input DataFrame cannot be empty.")

    X = add_constant(X)

    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                       for i in range(X.shape[1])]
    vif_data = vif_data.set_index("feature")

    return vif_data["VIF"]

from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

def calculate_auc_gini(y_true, y_pred):
    """Calculates AUC and Gini coefficient."""
    auc = roc_auc_score(y_true, y_pred)
    gini = 2 * auc - 1
    return auc, gini

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def generate_roc_curve(y_true, y_pred, title):
    """Generates ROC curves.

    Args:
        y_true (pandas.Series): The true target values.
        y_pred (pandas.Series): The predicted probabilities.
        title (str): The title of the plot.

    Returns:
        matplotlib.figure.Figure: The ROC curve plot.
    """
    if not isinstance(y_true, pd.Series) or not isinstance(y_pred, pd.Series):
        raise TypeError("y_true and y_pred must be pandas Series.")

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    if y_true.empty and y_pred.empty:
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], linestyle='--', color='r',
                label='No Skill')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend()
        return fig

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    return fig

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def generate_hosmer_lemeshow_plot(y_true, y_prob, n_bins, title):
    """Generates Hosmer-Lemeshow calibration plots.

    Args:
        y_true (pandas.Series): The true target values.
        y_prob (pandas.Series): The predicted probabilities.
        n_bins (int): The number of bins for the plot.
        title (str): The title of the plot.

    Returns:
        matplotlib.figure.Figure: The Hosmer-Lemeshow calibration plot.
    """

    if len(y_true) != len(y_prob):
        raise ValueError("Length of y_true and y_prob must be the same.")

    if n_bins <= 0:
        raise ValueError("Number of bins must be greater than 0.")

    if not all((0 <= p <= 1) for p in y_prob):
        raise ValueError("Probabilities must be between 0 and 1.")

    if len(y_true) == 0:
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Observed Proportion")
        return fig

    bins = np.linspace(0, 1, n_bins + 1)
    bin_assignments = np.digitize(y_prob, bins) - 1

    observed = []
    predicted = []
    for i in range(n_bins):
        bin_indices = bin_assignments == i
        if np.sum(bin_indices) > 0:
            observed.append(np.mean(y_true[bin_indices]))
            predicted.append(np.mean(y_prob[bin_indices]))
        else:
            observed.append(np.nan)
            predicted.append(np.nan)

    fig, ax = plt.subplots()
    ax.plot(predicted, observed, marker='o', linestyle='-')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')  # Add a diagonal line
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Proportion")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()  # Show the label for the diagonal line
    return fig

import pandas as pd


def map_pds_to_rating_grades(pds, n_grades):
    """Maps predicted PDs to rating grades."""
    if not isinstance(pds, pd.Series):
        raise TypeError("pds must be a pandas Series.")
    if not isinstance(n_grades, int):
        raise TypeError("n_grades must be an integer.")
    if n_grades <= 0:
        raise ValueError("n_grades must be a positive integer.")
    if pds.isnull().any():
        raise ValueError("pds cannot contain missing values.")
    if pds.empty:
        return pd.Series([])

    quantiles = [i / n_grades for i in range(1, n_grades)]
    cutoffs = pds.quantile(quantiles).tolist()
    cutoffs = sorted(list(set(cutoffs))) 
    n_grades_adjusted = len(cutoffs) + 1

    if len(set(pds)) == 1:
        return pd.Series([1] * len(pds), index=pds.index)

    bins = [-float('inf')] + cutoffs + [float('inf')]
    labels = list(range(1, n_grades_adjusted + 1))
    grades = pd.cut(pds, bins=bins, labels=labels, right=True, include_lowest=True).astype(int)
    return pd.Series(grades, index=pds.index)

import pandas as pd

def calibrate_pds(pds, actual_default_rates):
    """Calibrates PDs to observed default rates.
    Args:
        pds (pandas.Series): The predicted PDs.
        actual_default_rates (pandas.Series): The actual default rates.
    Returns:
        pandas.Series: The calibrated PDs.
    """
    if pds.empty or actual_default_rates.empty:
        return pds

    overall_predicted_default_rate = pds.mean()
    overall_actual_default_rate = actual_default_rates.mean()

    if overall_predicted_default_rate == 0:
        return pd.Series([0.0] * len(pds))

    calibration_factor = overall_actual_default_rate / overall_predicted_default_rate

    calibrated_pds = pds * calibration_factor
    calibrated_pds = calibrated_pds.clip(0, 1)

    return calibrated_pds

import pickle
import os

def save_model(model, filepath):
    """Saves the trained model to a file."""
    try:
        with open(filepath, 'wb') as file:
            pickle.dump(model, file)
    except FileNotFoundError:
        raise FileNotFoundError(f"The specified path '{filepath}' does not exist.")
    except Exception as e:
        raise Exception(f"Error saving model to {filepath}: {e}")

import pickle
import os

def save_preprocessing_pipeline(pipeline, filepath):
    """Saves the preprocessing pipeline to a file."""
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(pipeline, f)
    except FileNotFoundError:
        raise FileNotFoundError(f"The specified path '{filepath}' does not exist.")

import pandas as pd
import os

def save_rating_grade_cutoffs(cutoffs, filepath):
    """Saves rating grade cutoffs to a CSV."""
    try:
        with open(filepath, 'w') as f:
            f.write("Grade,Cutoff\n")
            for grade, cutoff in cutoffs.items():
                f.write(f"{grade},{cutoff}\n")
    except FileNotFoundError:
        raise FileNotFoundError

import yaml

def create_model_inventory_record(model_id, tier, owner, validator, last_validated, next_due, filepath):
    """Creates a model inventory record and saves it to a YAML file."""

    data = {
        'model_id': model_id,
        'tier': tier,
        'owner': owner,
        'validator': validator,
        'last_validated': last_validated,
        'next_due': next_due
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True)