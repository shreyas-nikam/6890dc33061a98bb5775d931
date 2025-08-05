
# Streamlit Application Requirements Specification

## 1. Application Overview

The Streamlit application aims to develop a wholesale credit rating model using the UCI Taiwan Credit Default dataset. This application will allow users to interact with the data, explore feature engineering options, train and compare different machine learning models, and calibrate the model outputs to generate a credit rating scale. The application will be designed to be reproducible and well-documented, adhering to regulatory standards for credit risk modeling. The objective of the application is to help users understand the model development process and to see how different parameters affect the outcome of the model.

## 2. User Interface Requirements

- **Layout and Navigation Structure:**
  - A sidebar for selecting data pre-processing steps and model training options.
  - A main panel for displaying data, charts, model performance metrics, and documentation.
  - Tab-based navigation within the main panel for different stages: Data Exploration, Feature Engineering, Model Training, Model Evaluation, and Model Calibration.

- **Input Widgets and Controls:**
  - **Data Ingestion**: Button to load the UCI Taiwan Credit Default dataset.
  - **Data Quality Checks:**
    - Checkbox to enable/disable mean imputation for missing values.
    - Slider to define the outlier threshold (e.g., number of standard deviations).
  - **Feature Engineering:**
    - Selection box for choosing imputation strategy ('median').
    - Sliders for lower and upper quantile values for Winsorization.
    - Checkboxes to select financial ratios to calculate (ROA, Debt-to-Equity, etc.)
    - Sliders for the weights assigned to `EDUCATION`, `MARRIAGE`, `AGE`, and average `PAY_X` for the synthetic management experience score.
    - Number input for the number of bins for quantile binning.
    - Multiselect widget to choose the columns for quantile binning.
  - **Model Training:**
    - Radio buttons to select the model type (Logistic Regression, Gradient Boosted Trees, Random Forest).
    - Slider for the regularization parameter `C` in Logistic Regression.
    - Number input for random state for reproducibility.
  - **Model Evaluation**: Display VIF values for features.
  - **Model Calibration**:
    - Display Predicted Probabilities before and after calibration.
    - Number input for the number of grades for PD to rating grade mapping.
  - **Documentation:** Button to generate a model documentation report.

- **Visualization Components:**
  - **Data Exploration:** Display of the first few rows of the dataframe, missing value counts, and summary statistics.
  - **Feature Exploration:**
    - Histograms and box plots for individual features.
    - Bar chart of univariate AUC/Gini for each predictor.
    - Heatmap of VIF values (highlighting VIF > 5).
  - **Model Evaluation:**
    - ROC curves for the training and validation sets.
    - Hosmer-Lemeshow calibration plots.
    - Histogram of obligor counts per rating grade.
    - Density plot of PD distribution before and after calibration.
  - **Model Performance Table**:
    - Display AUC and Gini coefficient for each model.

- **Interactive Elements and Feedback Mechanisms:**
  - Progress bars to indicate the status of data loading, processing, and model training.
  - Success/error messages to inform users about the outcome of each step.
  - Tooltips to explain the purpose of each widget and visualization.
  - Download buttons for saving model artifacts (e.g., trained models, grade cutoffs).

## 3. Additional Requirements

- **Real-time Updates and Responsiveness:**
  - The application should react instantly to user interactions, updating visualizations and metrics in real-time.
  - Data processing and model training should be performed efficiently to minimize latency.

- **Annotation and Tooltip Specifications:**
  - Tooltips should be provided for all input widgets and visualization elements, explaining their purpose and usage.
  - Annotations should be added to the charts to highlight key features and trends.
  - Units should be labeled on all plots.

## 4. Notebook Content and Code Requirements

### 4.1. Data Loading and Initial Exploration:

- Load the UCI "Taiwan Credit Default" dataset using `ucimlrepo`.
- Display the first few rows of the dataframe to show the data structure.
- Display missing values and the shape of the dataframe.
- Relevant code:
```python
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

# Fetch dataset
default_of_credit_card_clients = fetch_ucirepo(id=350)

# Data (as pandas dataframes)
X = default_of_credit_card_clients.data.features
y = default_of_credit_card_clients.data.targets

# Combine X and y for initial data processing
df = pd.concat([X, y], axis=1)

# Display metadata and variable information (optional)
print('Metadata:')
print(default_of_credit_card_clients.metadata)
print('\nVariable Information:')
print(default_of_credit_card_clients.variables)

print('\nInitial DataFrame head:')
print(df.head())
```

### 4.2. Data Quality Checks and Handling:

- Implement a function `perform_data_quality_checks` to perform missing value imputation and outlier handling.
- Missing values should be imputed using the mean.
- Outliers should be capped at 3 standard deviations from the mean.
- Relevant code:
```python
def perform_data_quality_checks(df):
    """Performs data quality checks for missing values and outliers.
    Args:
        df: pandas DataFrame.
    Returns:
        pandas DataFrame after quality checks.
    """
    if df.empty:
        return df

    # Check if numeric columns exist
    numeric_cols = df.select_dtypes(include=np.number).columns
    if not numeric_cols.any():
        raise TypeError("DataFrame must contain at least one numeric column.")

    # Handle missing values (mean imputation)
    df = df.fillna(df.mean(numeric_only=True))

    # Handle outliers (using 3 standard deviations)
    for col in numeric_cols:
        if df[col].std() == 0:  # Skip constant columns
            continue
        upper_limit = df[col].mean() + 3 * df[col].std()
        df[col] = df[col].apply(lambda x: min(x, upper_limit)) # Capping at upper limit

    return df.fillna(df.mean(numeric_only=True))
```

### 4.3. Imputing Missing Values:

- Implement `impute_missing_values` function, using median imputation strategy.
- Relevant code:
```python
def impute_missing_values(df, strategy):
    """Imputes missing values using a specified strategy.
    Args:
        df: pandas DataFrame.
        strategy: Imputation strategy (e.g., 'median').
    Returns:
        pandas DataFrame with imputed missing values.
    """
    df_copy = df.copy() # Work on a copy to avoid modifying original df directly if not desired
    for col in df_copy.columns:
        if df_copy[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                if strategy == 'median':
                    median_val = df_copy[col].median()
                    df_copy[col].fillna(median_val, inplace=True)
                else:
                    # Add other strategies if needed
                    print(f"Warning: Strategy '{strategy}' not implemented for column '{col}'. Skipping.")
            else:
                print(f"Warning: Skipping imputation for non-numeric column '{col}'.")
    return df_copy
```

### 4.4. Winsorizing Outlier Values:

- Implement `winsorize_outlier_values` function to cap extreme values at specified percentiles.
- Relevant code:
```python
def winsorize_outlier_values(df, columns, lower_quantile, upper_quantile):
    """Winsorizes outlier values in specified columns.
    Args:
        df: pandas DataFrame.
        columns: List of columns to winsorize.
        lower_quantile: Lower quantile (e.g., 0.01).
        upper_quantile: Upper quantile (e.g., 0.99).
    Returns:
        pandas DataFrame with winsorized values.
    """
    if not 0 <= lower_quantile <= 1 or not 0 <= upper_quantile <= 1:
        raise ValueError("Quantiles must be between 0 and 1")

    if not isinstance(df, pd.DataFrame) or df.empty:
        return df

    if not columns:
        return df

    df_copy = df.copy()
    for column in columns:
        if column in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[column]):
            lower_threshold = df_copy[column].quantile(lower_quantile)
            upper_threshold = df_copy[column].quantile(upper_quantile)
            df_copy[column] = np.where(df_copy[column] < lower_threshold, lower_threshold, df_copy[column])
            df_copy[column] = np.where(df_copy[column] > upper_threshold, upper_threshold, df_copy[column])
        else:
            print(f"Warning: Column '{column}' not found or not numeric. Skipping winsorization for this column.")
    return df_copy
```

### 4.5. Feature Engineering:

- Implement `calculate_financial_ratios` function to create financial ratios (ROA, Debt-to-Equity, etc.). Dummy data generation for calculation demonstration.
- Implement `create_management_experience_score` function to create a synthetic management experience score based on `EDUCATION`, `MARRIAGE`, `AGE`, and `PAY_X` variables.
- Implement `transform_skewed_variables` function to transform skewed variables using quantile binning.
- Relevant code:
```python
def calculate_financial_ratios(df):
    """Calculates financial ratios (ROA, Debt-to-Equity, etc.).
    Args:
        df: pandas DataFrame.
    Returns:
        pandas DataFrame with financial ratios.
    """
    if df.empty:
        raise ValueError("DataFrame cannot be empty")

    # --- IMPORTANT NOTE ---
    # The UCI Taiwan Credit Default dataset does NOT contain direct financial statement items
    # like 'Net Income', 'Total Assets', 'Total Debt', 'Shareholders\' Equity', etc.
    # For demonstration purposes, we will create DUMMY columns to allow the function to run.
    # In a real-world scenario, these columns would come from actual financial data.
    print("Note: UCI dataset lacks direct financial statement data. Creating dummy columns for demonstration.")
    df['Net Income'] = df['LIMIT_BAL'] * 0.15 + np.random.rand(len(df)) * 1000 # Dummy
    df['Total Assets'] = df['LIMIT_BAL'] * 2.0 + np.random.rand(len(df)) * 5000 # Dummy
    df['Total Debt'] = df['LIMIT_BAL'] * 1.5 + np.random.rand(len(df)) * 3000 # Dummy
    df['Shareholders\' Equity'] = df['LIMIT_BAL'] * 0.5 + np.random.rand(len(df)) * 1000 # Dummy
    df['Current Assets'] = df['LIMIT_BAL'] * 0.8 + np.random.rand(len(df)) * 2000 # Dummy
    df['Current Liabilities'] = df['LIMIT_BAL'] * 0.4 + np.random.rand(len(df)) * 1000 # Dummy
    df['EBITDA'] = df['LIMIT_BAL'] * 0.2 + np.random.rand(len(df)) * 1500 # Dummy
    df['Interest Expense'] = df['LIMIT_BAL'] * 0.05 + np.random.rand(len(df)) * 500 + 1 # Dummy (add 1 to avoid division by zero)
    
    # Ensure no division by zero for actual calculations
    df.loc[df['Total Assets'] == 0, 'Total Assets'] = np.nan # Handle cases where denominator might be 0
    df.loc[df['Shareholders\' Equity'] == 0, 'Shareholders\' Equity'] = np.nan
    df.loc[df['Current Liabilities'] == 0, 'Current Liabilities'] = np.nan
    df.loc[df['Interest Expense'] == 0, 'Interest Expense'] = np.nan

    # Calculate ROA
    df['ROA'] = df['Net Income'] / df['Total Assets']

    # Calculate Debt-to-Equity Ratio
    df['Debt-to-Equity Ratio'] = df['Total Debt'] / df['Shareholders\' Equity']

    # Calculate Current Ratio
    df['Current Ratio'] = df['Current Assets'] / df['Current Liabilities']

    # Calculate Cash Flow Coverage Ratio
    df['Cash Flow Coverage Ratio'] = df['EBITDA'] / df['Interest Expense']

    # Drop the dummy columns after ratio calculation if they are not needed further
    df = df.drop(columns=['Net Income', 'Total Assets', 'Total Debt', 'Shareholders\' Equity',
                          'Current Assets', 'Current Liabilities', 'EBITDA', 'Interest Expense'], errors='ignore')

    return df.fillna(0) # Fill any NaN introduced by division by zero, for example
```
```python
def create_management_experience_score(df):
    """Creates a synthetic 'Management Experience Score'.
    Args:
        df: pandas DataFrame.
    Output:
        pandas DataFrame with the management experience score.
    """
    if df.empty:
        df['management_experience_score'] = []
        return df

    # Ensure PAY_X columns exist. If not, create dummy ones or handle gracefully.
    pay_cols = [f'PAY_{i}' for i in range(7)] # PAY_0 to PAY_6
    for col in pay_cols:
        if col not in df.columns:
            df[col] = 0 # Default to 0 if missing for this synthetic score
            print(f"Warning: Column '{col}' not found. Adding as dummy for management experience score calculation.")

    df['management_experience_score'] = (
        df['EDUCATION'] * 0.2 + # Assuming EDUCATION is numeric and higher is better
        df['MARRIAGE'] * 0.1 + # Assuming MARRIAGE is numeric and certain values are better
        df['AGE'] * 0.05 + # Assuming AGE is numeric and higher is better
        df[pay_cols].mean(axis=1) * -0.1 # Average payment status; lower (less delay) is better, so negative weight
    )
    return df
```
```python
def transform_skewed_variables(df, columns, n_bins):
    """Transforms skewed variables using quantile binning.
    Args:
        df: pandas DataFrame.
        columns: List of columns to transform.
        n_bins: Number of bins for quantile binning.
    Returns:
        pandas DataFrame with transformed variables.
    """
    if n_bins <= 0:
        raise ValueError("Number of bins must be greater than 0.")

    df_transformed = df.copy()
    for col in columns:
        if col in df_transformed.columns and pd.api.types.is_numeric_dtype(df_transformed[col]):
            # Using qcut to perform quantile-based binning
            # labels=False returns integer indicators for bins
            # duplicates='drop' handles cases where quantiles are not unique (e.g., many identical values)
            df_transformed[col] = pd.qcut(df_transformed[col], q=n_bins, labels=False, duplicates='drop').astype('int64')
        else:
            print(f"Warning: Column '{col}' not found or not numeric. Skipping quantile binning for this column.")

    return df_transformed
```

### 4.6. Creating a Data Log:

- Implement `create_data_log` function to record all transformations performed on the data.
- Relevant code:
```python
import yaml
def create_data_log(transformations, filepath):
    """Creates a data log recording all data transformations.
    Args:
        transformations: Dictionary of data transformations.
        filepath: Path to the YAML file.
    """
    try:
        with open(filepath, 'w') as f:
            yaml.dump(transformations, f, default_flow_style=False)
        print(f"Data log successfully created at: {filepath}")
    except FileNotFoundError:
        raise FileNotFoundError(f"The specified path '{filepath}' does not exist.")
    except Exception as e:
        print(f"Error creating data log: {e}")
```

### 4.7. Splitting Data:

- Implement the `split_data` function for splitting the data into training and validation sets, stratified by the target variable.
- Relevant code:
```python
from sklearn.model_selection import train_test_split
def split_data(X, y, test_size, random_state):
    """Splits data into training and validation sets.
    Args:
        X: Features DataFrame.
        y: Target Series.
        test_size: Proportion of the dataset to include in the test split.
        random_state: Random seed for reproducibility.
    Returns:
        X_train, X_val, y_train, y_val
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_val, y_train, y_val
```

### 4.8. Model Training:

- Implement the `train_logistic_regression` function for training a Logistic Regression model.
- Implement the `train_gradient_boosted_trees` function for training a Gradient Boosted Trees model.
- Implement the `train_random_forest` function for training a Random Forest model.
- Relevant code:
```python
from sklearn.linear_model import LogisticRegression
def train_logistic_regression(X_train, y_train, C, random_state):
    """Trains a logistic regression model.
    Args:
        X_train: Training features.
        y_train: Training target variable.
        C: Regularization parameter (inverse of regularization strength).
        random_state: Random seed.
    Returns:
        Trained logistic regression model.
    """
    model = LogisticRegression(C=C, random_state=random_state, solver='liblinear', max_iter=1000) # Added solver and max_iter for robustness
    model.fit(X_train, y_train)
    return model
```
```python
from sklearn.ensemble import GradientBoostingClassifier
def train_gradient_boosted_trees(X_train, y_train, random_state):
    """Trains a gradient-boosted trees model.
    Args:
        X_train: Training features.
        y_train: Training target variable.
        random_state: Random seed.
    Returns:
        Trained gradient-boosted trees model.
    """
    model = GradientBoostingClassifier(random_state=random_state)
    if not X_train.empty and not y_train.empty:
        model.fit(X_train, y_train)
    return model
```
```python
from sklearn.ensemble import RandomForestClassifier
def train_random_forest(X_train, y_train, random_state):
    """Trains a random forest model.
    Args:
        X_train: Training features.
        y_train: Training target variable.
        random_state: Random seed.
    Returns:
        Trained random forest model.
    """
    if X_train.empty or y_train.empty:
        raise ValueError("Training data cannot be empty.")

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same length.")
    
    # Ensure all features are numeric. The previous steps should have handled this, but a check is good.
    if not all(pd.api.types.is_numeric_dtype(X_train[col]) for col in X_train.columns):
        raise TypeError("All features must be numeric.")

    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model
```

### 4.9. Multicollinearity Assessment:

- Implement the `calculate_vif` function to calculate Variance Inflation Factor (VIF) values.
- Relevant code:
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
def calculate_vif(X):
    """Calculates VIF values for multicollinearity assessment.
    Args:
        X: pandas DataFrame of features.
    Returns:
        pandas Series of VIF values.
    """
    if X.empty:
        raise ValueError("Input DataFrame cannot be empty.")
    if not all(np.issubdtype(X[col].dtype, np.number) for col in X.columns):
        # Filter for numeric columns if non-numeric are present
        X_numeric = X.select_dtypes(include=np.number)
        if X_numeric.empty:
            raise TypeError("DataFrame must contain at least one numeric column for VIF calculation.")
        print("Warning: Non-numeric columns were found and ignored for VIF calculation.")
        X = X_numeric

    # Add a constant term for the intercept in statsmodels VIF calculation
    X = X.copy()
    X['intercept'] = 1

    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                            for i in range(X.shape[1])]
    vif_data = vif_data.set_index("feature")["VIF"]
    
    # Drop the intercept's VIF as it's not a true feature VIF
    if 'intercept' in vif_data.index:
        vif_data = vif_data.drop('intercept')

    return vif_data.sort_values(ascending=False)
```

### 4.10. Model Evaluation Metrics:

- Implement the `calculate_auc_gini` function to calculate AUC and Gini coefficient.
- Relevant code:
```python
from sklearn.metrics import roc_auc_score, roc_curve, auc
def calculate_auc_gini(y_true, y_pred_proba):
    """Calculates AUC and Gini coefficient.
    Args:
        y_true (array-like): True binary labels.
        y_pred_proba (array-like): Predicted probabilities of the positive class.
    Returns:
        tuple: (AUC score, Gini coefficient).
    """
    if len(np.unique(y_true)) < 2: # Check if there are at least two unique classes
        print("Warning: y_true must contain at least two classes to calculate AUC/Gini. Returning NaN.")
        return np.nan, np.nan
    
    auc_score = roc_auc_score(y_true, y_pred_proba)
    gini = 2 * auc_score - 1
    return auc_score, gini
```

### 4.11. Generating ROC Curves:

- Implement the `generate_roc_curve` function for generating ROC curves.
- Relevant code:
```python
import matplotlib.pyplot as plt
def generate_roc_curve(y_true, y_pred_proba, title):
    """Generates ROC curves.
    Args:
        y_true (array-like): True target values.
        y_pred_proba (array-like): Predicted probabilities of the positive class.
        title (str): Title of the plot.
    Raises:
        ValueError: If input arrays are empty or have mismatched lengths.
    Returns:
        None: Displays the ROC curve plot.
    """
    if len(y_true) == 0 or len(y_pred_proba) == 0:
        raise ValueError("Input arrays cannot be empty.")

    if len(y_true) != len(y_pred_proba):
        raise ValueError("Input arrays must have the same length.")

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
```

### 4.12. Generating Calibration Plots:

- Implement the `generate_calibration_plot` function for generating Hosmer-Lemeshow calibration plots.
- Relevant code:
```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def generate_calibration_plot(y_true, y_prob, n_bins, title):
    """Generates Hosmer-Lemeshow calibration plots.
    Args:
        y_true (array-like): True binary labels.
        y_prob (array-like): Predicted probabilities of the positive class.
        n_bins (int): Number of bins for calibration.
        title (str): Title of the plot.
    """
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_prob, np.ndarray):
        y_prob = np.array(y_prob)

    if y_true.size == 0 and y_prob.size == 0:
        print("No data to plot for calibration. Skipping.")
        return

    if y_true.size != y_prob.size:
        raise ValueError("y_true and y_prob must have the same length.")

    if n_bins <= 0:
        raise ValueError("n_bins must be a positive integer.")

    if np.any(y_prob < 0) or np.any(y_prob > 1):
        print("Warning: y_prob contains values outside [0, 1]. Clamping to [0, 1].")
        y_prob = np.clip(y_prob, 0, 1)

    bin_cuts = np.linspace(0, 1, n_bins + 1)
    # Use pd.cut to get bin labels and average predicted probability for each bin
    bins = pd.cut(y_prob, bins=bin_cuts, include_lowest=True, labels=False)

    bin_mean_prob = []
    bin_accuracy = []

    for i in range(n_bins):
        indices = np.where(bins == i)[0]
        if len(indices) > 0:
            mean_pred_prob = np.mean(y_prob[indices])
            observed_proportion = np.mean(y_true[indices])
            bin_mean_prob.append(mean_pred_prob)
            bin_accuracy.append(observed_proportion)

    bin_mean_prob = np.array(bin_mean_prob)
    bin_accuracy = np.array(bin_accuracy)

    plt.figure(figsize=(8, 6))
    plt.plot(bin_mean_prob, bin_accuracy, marker='o', linestyle='-', label='Calibration Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')

    plt.xlabel('Mean Predicted Probability (in bin)')
    plt.ylabel('Observed Proportion of Positives (in bin)')
    plt.title(title)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend()
    plt.grid(True)
    plt.show()
```

### 4.13. Calibrating PDs:

- Implement `calibrate_pds` to calibrate predicted PDs to observed default rates, with a simple scaling factor for demonstration.
- Relevant code:
```python
import numpy as np
def calibrate_pds(predicted_probabilities, actual_default_rates):
    """Calibrates PDs to observed default rates.
    Note: This function is a placeholder. In a real-world scenario, more sophisticated
    calibration techniques (e.g., Platt scaling, Isotonic Regression, or a simple
    scaling factor derived from observed default rates) would be applied here.
    For demonstration, we assume 'actual_default_rates' represent the desired calibrated output.
    Args:
        predicted_probabilities (np.ndarray): Predicted probabilities from the model.
        actual_default_rates (np.ndarray): The observed default rates (or target values).
    Returns:
        np.ndarray: Calibrated probabilities.
    """
    if len(predicted_probabilities) != len(actual_default_rates):
        raise ValueError("Predicted probabilities and actual default rates must have the same length.")

    # A very simplistic 'calibration' for demonstration: just return actuals.
    # In practice, you'd fit a new model (e.g., logistic regression) on pred_proba vs y_true
    # and then transform pred_proba using this new model.
    # For this example, let's just show that this function is where calibration logic would go.
    
    # Let's simulate a simple calibration adjustment for demonstration purposes:
    # Assume a linear scaling for simplicity if you want to show a transformation.
    # Or, if we are just demonstrating the concept of aligning to actuals, we can use binning.
    
    # For a more meaningful demonstration within this function, let's perform
    # a basic re-calibration to match the overall observed default rate.
    overall_observed_default_rate = actual_default_rates.mean()
    overall_predicted_mean_proba = predicted_probabilities.mean()

    if overall_predicted_mean_proba == 0:
        print("Warning: Mean predicted probability is zero, cannot calibrate.")
        return predicted_probabilities

    scaling_factor = overall_observed_default_rate / overall_predicted_mean_proba
    calibrated_probabilities = predicted_probabilities * scaling_factor
    
    # Ensure probabilities remain within [0, 1]
    calibrated_probabilities = np.clip(calibrated_probabilities, 0, 1)

    print(f"Original Mean Predicted Probability: {overall_predicted_mean_proba:.4f}")
    print(f"Overall Observed Default Rate: {overall_observed_default_rate:.4f}")
    print(f"Calibration Scaling Factor: {scaling_factor:.4f}")
    print(f"Calibrated Mean Predicted Probability: {calibrated_probabilities.mean():.4f}")

    return calibrated_probabilities
```

### 4.14. Mapping PDs to Rating Grades:

- Implement `map_pd_to_rating_grades` to map predicted PDs to rating grades using a quantile-based approach.
- Relevant code:
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def map_pd_to_rating_grades(predicted_probabilities, num_grades):
    """Maps predicted PDs to rating grades using a quantile-based approach.
    Args:
        predicted_probabilities (pd.Series): Predicted probabilities of default.
        num_grades (int): The desired number of discrete rating grades.
    Returns:
        pd.Series: Rating grades assigned to each prediction.
    """
    if predicted_probabilities.empty:
        return pd.Series([])

    if num_grades <= 0:
        raise ValueError("Number of grades must be positive.")

    # Calculate quantiles for grade cutoffs. We need num_grades - 1 cutoffs.
    # Ensure q values are valid (0 to 1, exclusive of 0 and 1 for strict cutoffs)
    q_values = [i / num_grades for i in range(1, num_grades)]
    
    # Use np.linspace for potentially more robust quantile calculation if q values are simple
    # Or stick with pd.Series.quantile with unique q values.
    quantiles = predicted_probabilities.quantile(q=q_values)

    # Initialize grades series with a default value (e.g., the lowest grade index)
    grades = pd.Series(0, index=predicted_probabilities.index)

    # Assign grades based on quantiles. Higher PD implies higher risk, lower grade number.
    # The example function maps higher PD to higher grade number (0 is lowest PD, num_grades-1 is highest PD)
    # Let's reverse this for typical credit grades where A is better than C.
    # Let's assign grade 1 to lowest PD (best credit), up to num_grades to highest PD (worst credit).
    # So, the grade increases as PD increases.
    
    grade_map = {} # Store grade boundaries and their corresponding grade
    sorted_quantiles = sorted(quantiles.tolist())
    
    # Assign the first grade (highest credit quality) to probabilities below the first quantile
    # Default to 1 (best grade)
    grades[predicted_probabilities <= sorted_quantiles[0]] = 1
    grade_cutoffs_list = [sorted_quantiles[0]] # Store the actual cutoffs

    # Assign intermediate grades
    for i in range(len(sorted_quantiles) - 1):
        lower_bound = sorted_quantiles[i]
        upper_bound = sorted_quantiles[i+1]
        grades[(predicted_probabilities > lower_bound) & (predicted_probabilities <= upper_bound)] = i + 2
        grade_cutoffs_list.append(upper_bound)

    # Assign the last grade (worst credit quality) to probabilities above the last quantile
    grades[predicted_probabilities > sorted_quantiles[-1]] = num_grades

    # Return grades (1 to num_grades) and cutoffs
    return grades, pd.Series(grade_cutoffs_list, index=[f'Grade_1_Cutoff', ] + [f'Grade_{i+2}_Cutoff' for i in range(num_grades-2)])
```

### 4.15. Saving and Loading Models and Data:

- Implement the `save_model`, `load_model`, and `save_data` functions for persisting model artifacts which will download this data.
- Compulsorily have markdown explanations for these concepts.