import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score, roc_curve, auc
import yaml # For data log

# Re-defining helper functions as per prompt requirements for completeness within each file
# and to avoid cross-page imports if not explicitly specified.

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
                    st.warning(f"Warning: Strategy '{strategy}' not implemented for column '{col}'. Skipping.")
            else:
                st.warning(f"Warning: Skipping imputation for non-numeric column '{col}'.")
    return df_copy

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
        st.error("Quantiles must be between 0 and 1.")
        return df

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
            st.warning(f"Warning: Column '{column}' not found or not numeric. Skipping winsorization for this column.")
    return df_copy

def calculate_financial_ratios(df):
    """Calculates financial ratios (ROA, Debt-to-Equity, etc.).
    Args:
        df: pandas DataFrame.
    Returns:
        pandas DataFrame with financial ratios.
    """
    if df.empty:
        st.error("DataFrame cannot be empty for financial ratio calculation.")
        return df

    df_copy = df.copy()

    # --- IMPORTANT NOTE ---
    # The UCI Taiwan Credit Default dataset does NOT contain direct financial statement items
    # like 'Net Income', 'Total Assets', 'Total Debt', 'Shareholders\' Equity', etc.
    # For demonstration purposes, we will create DUMMY columns to allow the function to run.
    # In a real-world scenario, these columns would come from actual financial data.
    st.info("Note: UCI dataset lacks direct financial statement data. Creating dummy columns for demonstration.")
    df_copy['Net Income'] = df_copy['LIMIT_BAL'] * 0.15 + np.random.rand(len(df_copy)) * 1000 # Dummy
    df_copy['Total Assets'] = df_copy['LIMIT_BAL'] * 2.0 + np.random.rand(len(df_copy)) * 5000 # Dummy
    df_copy['Total Debt'] = df_copy['LIMIT_BAL'] * 1.5 + np.random.rand(len(df_copy)) * 3000 # Dummy
    df_copy['Shareholders_Equity'] = df_copy['LIMIT_BAL'] * 0.5 + np.random.rand(len(df_copy)) * 1000 # Dummy
    df_copy['Current Assets'] = df_copy['LIMIT_BAL'] * 0.8 + np.random.rand(len(df_copy)) * 2000 # Dummy
    df_copy['Current Liabilities'] = df_copy['LIMIT_BAL'] * 0.4 + np.random.rand(len(df_copy)) * 1000 # Dummy
    df_copy['EBITDA'] = df_copy['LIMIT_BAL'] * 0.2 + np.random.rand(len(df_copy)) * 1500 # Dummy
    df_copy['Interest Expense'] = df_copy['LIMIT_BAL'] * 0.05 + np.random.rand(len(df_copy)) * 500 + 1 # Dummy (add 1 to avoid division by zero)
    
    # Ensure no division by zero for actual calculations
    df_copy.loc[df_copy['Total Assets'] == 0, 'Total Assets'] = np.nan # Handle cases where denominator might be 0
    df_copy.loc[df_copy['Shareholders_Equity'] == 0, 'Shareholders_Equity'] = np.nan
    df_copy.loc[df_copy['Current Liabilities'] == 0, 'Current Liabilities'] = np.nan
    df_copy.loc[df_copy['Interest Expense'] == 0, 'Interest Expense'] = np.nan

    # Calculate ROA
    df_copy['ROA'] = df_copy['Net Income'] / df_copy['Total Assets']

    # Calculate Debt-to-Equity Ratio
    df_copy['Debt-to-Equity Ratio'] = df_copy['Total Debt'] / df_copy['Shareholders_Equity']

    # Calculate Current Ratio
    df_copy['Current Ratio'] = df_copy['Current Assets'] / df_copy['Current Liabilities']

    # Calculate Cash Flow Coverage Ratio
    df_copy['Cash Flow Coverage Ratio'] = df_copy['EBITDA'] / df_copy['Interest Expense']

    # Drop the dummy columns after ratio calculation if they are not needed further
    df_copy = df_copy.drop(columns=['Net Income', 'Total Assets', 'Total Debt', 'Shareholders_Equity',
                          'Current Assets', 'Current Liabilities', 'EBITDA', 'Interest Expense'], errors='ignore')

    return df_copy.fillna(0) # Fill any NaN introduced by division by zero, for example

def create_management_experience_score(df, edu_weight, marriage_weight, age_weight, pay_x_weight):
    """Creates a synthetic 'Management Experience Score'.
    Args:
        df: pandas DataFrame.
        edu_weight: Weight for EDUCATION.
        marriage_weight: Weight for MARRIAGE.
        age_weight: Weight for AGE.
        pay_x_weight: Weight for average PAY_X.
    Output:
        pandas DataFrame with the management experience score.
    """
    if df.empty:
        df['management_experience_score'] = []
        return df

    df_copy = df.copy()
    # Ensure PAY_X columns exist. If not, create dummy ones or handle gracefully.
    pay_cols = [f'PAY_{i}' for i in range(7)] # PAY_0 to PAY_6
    for col in pay_cols:
        if col not in df_copy.columns:
            df_copy[col] = 0 # Default to 0 if missing for this synthetic score
            st.warning(f"Warning: Column '{col}' not found. Adding as dummy for management experience score calculation.")

    # Ensure EDUCATION, MARRIAGE, AGE exist and are numeric
    for col in ['EDUCATION', 'MARRIAGE', 'AGE']:
        if col not in df_copy.columns:
            df_copy[col] = 0 # Default to 0 if missing
            st.warning(f"Warning: Column '{col}' not found. Adding as dummy for management experience score calculation.")
        elif not pd.api.types.is_numeric_dtype(df_copy[col]):
            st.warning(f"Warning: Column '{col}' is not numeric. Attempting to convert to numeric for management experience score.")
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0)


    df_copy['management_experience_score'] = (
        df_copy['EDUCATION'] * edu_weight + # Assuming EDUCATION is numeric and higher is better
        df_copy['MARRIAGE'] * marriage_weight + # Assuming MARRIAGE is numeric and certain values are better
        df_copy['AGE'] * age_weight + # Assuming AGE is numeric and higher is better
        df_copy[pay_cols].mean(axis=1) * pay_x_weight # Average payment status; lower (less delay) is better, so negative weight
    )
    return df_copy

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
        st.error("Number of bins must be greater than 0.")
        return df

    df_transformed = df.copy()
    for col in columns:
        if col in df_transformed.columns and pd.api.types.is_numeric_dtype(df_transformed[col]):
            # Check for unique values, if too few, qcut might fail
            if df_transformed[col].nunique() < n_bins:
                st.warning(f"Column '{col}' has fewer unique values ({df_transformed[col].nunique()}) than specified bins ({n_bins}). Skipping quantile binning for this column.")
                continue
            try:
                # Using qcut to perform quantile-based binning
                # labels=False returns integer indicators for bins
                # duplicates='drop' handles cases where quantiles are not unique (e.g., many identical values)
                df_transformed[col] = pd.qcut(df_transformed[col], q=n_bins, labels=False, duplicates='drop').astype('int64')
            except Exception as e:
                st.error(f"Error applying quantile binning to column '{col}': {e}. This might happen if there are too many duplicate values preventing unique quantiles.")
                continue
        else:
            st.warning(f"Warning: Column '{col}' not found or not numeric. Skipping quantile binning for this column.")

    return df_transformed

def create_data_log(transformations, filepath):
    """Creates a data log recording all data transformations.
    Args:
        transformations: Dictionary of data transformations.
        filepath: Path to the YAML file.
    """
    try:
        with open(filepath, 'w') as f:
            yaml.dump(transformations, f, default_flow_style=False)
        st.success(f"Data log successfully created at: {filepath}")
        st.download_button(
            label="Download Data Transformation Log",
            data=yaml.dump(transformations, default_flow_style=False),
            file_name="data_transformation_log.yaml",
            mime="text/yaml"
        )
    except Exception as e:
        st.error(f"Error creating data log: {e}")

def calculate_auc_gini(y_true, y_pred_proba):
    """Calculates AUC and Gini coefficient.
    Args:
        y_true (array-like): True binary labels.
        y_pred_proba (array-like): Predicted probabilities of the positive class.
    Returns:
        tuple: (AUC score, Gini coefficient).
    """
    if len(np.unique(y_true)) < 2: # Check if there are at least two unique classes
        st.warning("Warning: y_true must contain at least two classes to calculate AUC/Gini. Returning NaN.")
        return np.nan, np.nan
    
    # Ensure y_true and y_pred_proba are numeric and same length
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    if len(y_true) != len(y_pred_proba):
        st.warning("y_true and y_pred_proba must have the same length.")
        return np.nan, np.nan
    
    if not (np.issubdtype(y_true.dtype, np.number) and np.issubdtype(y_pred_proba.dtype, np.number)):
        st.warning("y_true and y_pred_proba must be numeric.")
        return np.nan, np.nan


    auc_score = roc_auc_score(y_true, y_pred_proba)
    gini = 2 * auc_score - 1
    return auc_score, gini

def run_feature_engineering():
    st.header("2. Feature Engineering")

    st.markdown("""
    This section focuses on transforming raw data into features that can be effectively used by machine learning models.
    You can apply various techniques such as imputation, outlier handling, creating new ratios, and binning variables.
    Each step's impact on the data can be visualized and tracked.
    """)

    if "df_processed" not in st.session_state or st.session_state.df_processed.empty:
        st.warning("Please go back to 'Data Exploration' and load/process the dataset first.")
        return

    # Initialize a working DataFrame for this page
    if "df_fe" not in st.session_state:
        st.session_state.df_fe = st.session_state.df_processed.copy()
    
    st.subheader("Current Data Head (after previous steps)")
    st.write(st.session_state.df_fe.head())
    
    st.markdown("---")

    # --- Imputing Missing Values ---
    st.subheader("Imputing Missing Values")
    st.markdown("""
    Fill any remaining missing values in your dataset. The current implementation supports median imputation for numerical columns.
    """)
    imputation_strategy = st.selectbox("Select Imputation Strategy", ["median"], key="impute_strategy")
    if st.button("Apply Missing Value Imputation", key="apply_imputation"):
        with st.spinner("Applying median imputation..."):
            st.session_state.df_fe = impute_missing_values(st.session_state.df_fe, imputation_strategy)
            st.success("Missing values imputed.")
            st.write(st.session_state.df_fe.head())
            st.subheader("Missing Values After Imputation")
            missing_values_after = st.session_state.df_fe.isnull().sum()
            missing_values_df_after = pd.DataFrame({'Column': missing_values_after.index, 'Missing Count': missing_values_after.values})
            missing_values_df_after = missing_values_df_after[missing_values_df_after['Missing Count'] > 0]
            if not missing_values_df_after.empty:
                st.dataframe(missing_values_df_after.set_index('Column'))
            else:
                st.info("No missing values after imputation.")


    st.markdown("---")

    # --- Winsorizing Outlier Values ---
    st.subheader("Winsorizing Outlier Values")
    st.markdown("""
    Winsorization caps extreme values at a specified percentile, replacing values outside the range with values at the extremes of the distribution.
    This helps to mitigate the impact of outliers without removing them entirely.
    """)
    numeric_cols = st.session_state.df_fe.select_dtypes(include=np.number).columns.tolist()
    winsorize_cols = st.multiselect("Select columns to Winsorize", numeric_cols, key="winsorize_cols")
    lower_quantile = st.slider("Lower Quantile for Winsorization", min_value=0.0, max_value=0.5, value=0.01, step=0.01, key="lower_q")
    upper_quantile = st.slider("Upper Quantile for Winsorization", min_value=0.0, max_value=1.0, value=0.99, step=0.01, key="upper_q")

    if st.button("Apply Winsorization", key="apply_winsorization"):
        with st.spinner("Applying Winsorization..."):
            st.session_state.df_fe = winsorize_outlier_values(st.session_state.df_fe, winsorize_cols, lower_quantile, upper_quantile)
            st.success("Outliers winsorized.")
            st.write(st.session_state.df_fe.head())

    st.markdown("---")

    # --- Financial Ratio Calculation ---
    st.subheader("Calculate Financial Ratios (Dummy Data)")
    st.markdown("""
    This section demonstrates how financial ratios can be calculated. **Please note:** the UCI Taiwan Credit Default dataset
    does not contain raw financial statement data. Therefore, this function uses **dummy data** based on `LIMIT_BAL`
    to illustrate the calculation. In a real application, you would use actual financial figures.
    """)
    
    st.info("Select the financial ratios you wish to calculate. Dummy financial data will be generated to facilitate these calculations.")
    
    # Checkboxes are for user's preference, not for enabling/disabling function's internal logic as the function
    # calculates all if called. These checkboxes are more for documentation.
    st.checkbox("Include Return on Assets (ROA)", value=True, disabled=True)
    st.checkbox("Include Debt-to-Equity Ratio", value=True, disabled=True)
    st.checkbox("Include Current Ratio", value=True, disabled=True)
    st.checkbox("Include Cash Flow Coverage Ratio", value=True, disabled=True)

    if st.button("Calculate Ratios", key="calculate_ratios"):
        with st.spinner("Calculating dummy financial ratios..."):
            # Ensure 'LIMIT_BAL' exists for dummy data generation
            if 'LIMIT_BAL' in st.session_state.df_fe.columns:
                st.session_state.df_fe = calculate_financial_ratios(st.session_state.df_fe)
                st.success("Financial ratios calculated (using dummy data).")
                st.write(st.session_state.df_fe.head())
                st.markdown("New columns added: `ROA`, `Debt-to-Equity Ratio`, `Current Ratio`, `Cash Flow Coverage Ratio`.")
            else:
                st.error("Cannot calculate financial ratios: 'LIMIT_BAL' column not found in the dataset.")


    st.markdown("---")

    # --- Create Management Experience Score ---
    st.subheader("Create Management Experience Score")
    st.markdown("""
    A synthetic 'Management Experience Score' is created based on `EDUCATION`, `MARRIAGE`, `AGE`, and average `PAY_X` variables.
    You can adjust the weights to see how they influence the composite score.
    Higher scores typically indicate better management experience.
    """)
    
    # Ensure relevant columns for management score are present or warn user
    required_mgmt_cols = ['EDUCATION', 'MARRIAGE', 'AGE'] + [f'PAY_{i}' for i in range(7)]
    if not all(col in st.session_state.df_fe.columns for col in required_mgmt_cols):
        st.warning(f"Some required columns for 'Management Experience Score' (e.g., {\', \'.join([c for c in required_mgmt_cols if c not in st.session_state.df_fe.columns])}) are missing. The score will be calculated with dummy zeros for missing columns.")

    edu_w = st.slider("Weight for EDUCATION", min_value=0.0, max_value=1.0, value=0.2, step=0.05, key="edu_w")
    marriage_w = st.slider("Weight for MARRIAGE", min_value=0.0, max_value=1.0, value=0.1, step=0.05, key="marriage_w")
    age_w = st.slider("Weight for AGE", min_value=0.0, max_value=1.0, value=0.05, step=0.05, key="age_w")
    pay_x_w = st.slider("Weight for Average PAY_X (Negative implies better score for lower pay status)", min_value=-1.0, max_value=0.0, value=-0.1, step=0.05, key="pay_x_w")

    if st.button("Create Score", key="create_score"):
        with st.spinner("Creating management experience score..."):
            st.session_state.df_fe = create_management_experience_score(st.session_state.df_fe, edu_w, marriage_w, age_w, pay_x_w)
            st.success("Management experience score created.")
            st.write(st.session_state.df_fe.head())
            st.markdown("New column added: `management_experience_score`.")

    st.markdown("---")

    # --- Quantile Binning ---
    st.subheader("Quantile Binning Skewed Variables")
    st.markdown("""
    Quantile binning transforms continuous numerical variables into discrete bins.
    This can help normalize skewed distributions and create categorical features from continuous ones.
    """)
    current_numeric_cols = st.session_state.df_fe.select_dtypes(include=np.number).columns.tolist()
    bin_cols = st.multiselect("Select columns for Quantile Binning", current_numeric_cols, key="bin_cols")
    n_bins = st.number_input("Number of Bins", min_value=2, value=5, step=1, key="n_bins")

    if st.button("Apply Quantile Binning", key="apply_binning"):
        if bin_cols:
            with st.spinner("Applying quantile binning..."):
                st.session_state.df_fe = transform_skewed_variables(st.session_state.df_fe, bin_cols, n_bins)
                st.success("Quantile binning applied.")
                st.write(st.session_state.df_fe.head())
        else:
            st.warning("Please select at least one column for Quantile Binning.")

    st.markdown("---")

    # --- Feature Exploration Visualizations ---
    st.subheader("Feature Exploration Visualizations")

    # Histograms and Box Plots
    st.markdown("#### Histograms and Box Plots")
    st.markdown("""
    Visualize the distribution of individual features using histograms and box plots.
    This helps in understanding the data's spread, central tendency, and presence of outliers.
    """)
    visualization_cols = st.session_state.df_fe.select_dtypes(include=np.number).columns.tolist()
    selected_viz_col = st.selectbox("Select a feature to visualize", visualization_cols, key="selected_viz_col")

    if not st.session_state.df_fe.empty and selected_viz_col:
        fig_hist = px.histogram(st.session_state.df_fe, x=selected_viz_col,
                                title=f'Histogram of {selected_viz_col}',
                                template='plotly_white')
        st.plotly_chart(fig_hist, use_container_width=True)

        fig_box = px.box(st.session_state.df_fe, y=selected_viz_col,
                            title=f'Box Plot of {selected_viz_col}',
                            template='plotly_white')
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")

    # Univariate AUC/Gini
    st.markdown("#### Univariate AUC/Gini for Predictors")
    st.markdown("""
    Evaluate the predictive power of individual numerical features against the target variable (default).
    A higher AUC or Gini coefficient indicates a stronger relationship with the target.
    """)
    # Assuming the target column is 'default.payment.next.month' as per UCI dataset info
    target_col_name = 'default payment next month'
    
    if target_col_name in st.session_state.df_fe.columns:
        # Exclude target and any non-numeric columns from feature list
        numeric_features = st.session_state.df_fe.select_dtypes(include=np.number).columns.drop(target_col_name, errors='ignore').tolist()
        
        if st.button("Calculate Univariate AUC/Gini", key="calculate_univariate_auc_gini"):
            if not numeric_features:
                st.warning("No numeric features found to calculate AUC/Gini.")
            else:
                auc_gini_results = []
                y_true = st.session_state.df_fe[target_col_name]
                
                with st.spinner("Calculating univariate AUC/Gini for features..."):
                    for col in numeric_features:
                        # For univariate AUC, we use the feature itself as a "predicted probability" proxy
                        # It needs to have variance and more than one unique value.
                        if st.session_state.df_fe[col].nunique() > 1:
                            try:
                                auc_score, gini_score = calculate_auc_gini(y_true, st.session_state.df_fe[col])
                                auc_gini_results.append({'Feature': col, 'AUC': auc_score, 'Gini': gini_score})
                            except Exception as e:
                                st.error(f"Error calculating AUC/Gini for '{col}': {e}")
                                auc_gini_results.append({'Feature': col, 'AUC': np.nan, 'Gini': np.nan})
                        else:
                            st.info(f"Feature '{col}' has only one unique value. Skipping AUC/Gini calculation.")
                            auc_gini_results.append({'Feature': col, 'AUC': np.nan, 'Gini': np.nan})

                if auc_gini_results:
                    auc_gini_df = pd.DataFrame(auc_gini_results).sort_values(by="Gini", ascending=False).dropna(subset=['Gini'])
                    if not auc_gini_df.empty:
                        st.dataframe(auc_gini_df)

                        fig_gini = px.bar(auc_gini_df, x='Feature', y='Gini',
                                          title='Univariate Gini Coefficient for Each Predictor',
                                          labels={'Gini': 'Gini Coefficient'},
                                          template='plotly_white')
                        st.plotly_chart(fig_gini, use_container_width=True)
                    else:
                        st.info("No valid univariate AUC/Gini results to display. All features might have issues.")
                else:
                    st.info("No univariate AUC/Gini results to display. Check if features are suitable.")
    else:
        st.warning(f"Target column '{target_col_name}' not found in the DataFrame. Cannot calculate univariate AUC/Gini.")


    st.markdown("---")
    
    # --- Data Log ---
    st.subheader("Data Transformation Log")
    st.markdown("""
    A data log records all the transformations applied to the dataset during the feature engineering process.
    This is crucial for reproducibility and auditing the model development lifecycle.
    """)
    if "transformations_log" not in st.session_state:
        st.session_state.transformations_log = []

    if st.button("Generate and Download Data Transformation Log", key="generate_log"):
        # This is a simplified log. In a real scenario, you'd capture parameters of each step.
        log_entry = {
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "steps_applied": []
        }
        
        # Example of how to populate log based on user interactions
        # Note: Session state variables might not exist if user did not interact with corresponding widgets.
        # Check for existence before adding to log.
        if "impute_strategy" in st.session_state:
            log_entry["steps_applied"].append({"Imputation": {"strategy": st.session_state.impute_strategy}})
        
        if "winsorize_cols" in st.session_state and st.session_state.winsorize_cols:
             log_entry["steps_applied"].append({
                "Winsorization": {
                    "columns": st.session_state.winsorize_cols,
                    "lower_quantile": st.session_state.lower_q,
                    "upper_quantile": st.session_state.upper_q
                }
            })
        
        # Financial Ratios - based on button press. Assuming all are calculated if button pressed.
        if st.session_state.get("calculate_ratios"): # Checks if button was pressed
            log_entry["steps_applied"].append({"Financial Ratios": "Calculated (using dummy data from LIMIT_BAL)"})
        
        if st.session_state.get("create_score"): # Checks if button was pressed
            log_entry["steps_applied"].append({
                "Management Experience Score": {
                    "EDUCATION_weight": st.session_state.get("edu_w"),
                    "MARRIAGE_weight": st.session_state.get("marriage_w"),
                    "AGE_weight": st.session_state.get("age_w"),
                    "PAY_X_weight": st.session_state.get("pay_x_w")
                }
            })
        
        if st.session_state.get("bin_cols") and st.session_state.get("apply_binning"): # Check if binning was applied and columns selected
            log_entry["steps_applied"].append({
                "Quantile Binning": {
                    "columns": st.session_state.get("bin_cols"),
                    "n_bins": st.session_state.get("n_bins")
                }
            })

        st.session_state.transformations_log.append(log_entry)
        create_data_log(st.session_state.transformations_log, "data_transformation_log.yaml")
