import streamlit as st
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
import plotly.express as px

# Helper function for data quality checks
def perform_data_quality_checks(df, enable_imputation=False, outlier_threshold=3):
    """Performs data quality checks for missing values and outliers.
    Args:
        df: pandas DataFrame.
        enable_imputation: Boolean, if True, performs mean imputation.
        outlier_threshold: Number of standard deviations for outlier capping.
    Returns:
        pandas DataFrame after quality checks.
    """
    if df.empty:
        return df

    df_cleaned = df.copy()
    numeric_cols = df_cleaned.select_dtypes(include=np.number).columns

    if not numeric_cols.any():
        st.warning("DataFrame must contain at least one numeric column for quality checks.")
        return df_cleaned

    # Handle missing values (mean imputation) if enabled
    if enable_imputation:
        for col in numeric_cols:
            if df_cleaned[col].isnull().any():
                mean_val = df_cleaned[col].mean()
                df_cleaned[col].fillna(mean_val, inplace=True)
        st.success("Missing values imputed using mean.")

    # Handle outliers (using standard deviations)
    for col in numeric_cols:
        if df_cleaned[col].std() == 0:  # Skip constant columns
            continue
        
        upper_limit = df_cleaned[col].mean() + outlier_threshold * df_cleaned[col].std()
        lower_limit = df_cleaned[col].mean() - outlier_threshold * df_cleaned[col].std()
        
        df_cleaned[col] = df_cleaned[col].apply(lambda x: min(x, upper_limit)) # Capping at upper limit
        df_cleaned[col] = df_cleaned[col].apply(lambda x: max(x, lower_limit)) # Capping at lower limit
    st.success(f"Outliers capped at {outlier_threshold} standard deviations.")

    return df_cleaned

def run_data_exploration():
    st.header("1. Data Exploration")

    st.markdown("""
    This section allows you to load the dataset and perform initial data quality checks. You can view the raw data, 
    check for missing values, and review summary statistics. Adjust the settings to see how data quality steps 
    affect the dataset.
    """)

    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame()

    if st.button("Load UCI Taiwan Credit Default Dataset"):
        with st.spinner("Loading dataset..."):
            try:
                # Fetch dataset
                default_of_credit_card_clients = fetch_ucirepo(id=350)

                # Data (as pandas dataframes)
                X = default_of_credit_card_clients.data.features
                y = default_of_credit_card_clients.data.targets

                # Combine X and y for initial data processing
                st.session_state.df = pd.concat([X, y], axis=1)
                st.success("Dataset loaded successfully!")
            except Exception as e:
                st.error(f"Error loading dataset: {e}")

    if not st.session_state.df.empty:
        st.subheader("Raw Data Preview")
        st.write(st.session_state.df.head())

        st.subheader("Missing Values")
        missing_values = st.session_state.df.isnull().sum()
        missing_values_df = pd.DataFrame({'Column': missing_values.index, 'Missing Count': missing_values.values})
        missing_values_df = missing_values_df[missing_values_df['Missing Count'] > 0]
        if not missing_values_df.empty:
            st.dataframe(missing_values_df.set_index('Column'))
        else:
            st.info("No missing values found in the loaded dataset.")

        st.subheader("DataFrame Shape")
        st.write(f"Rows: {st.session_state.df.shape[0]}, Columns: {st.session_state.df.shape[1]}")

        st.subheader("Summary Statistics")
        st.write(st.session_state.df.describe())

        st.subheader("Data Quality Checks and Handling")
        st.markdown("""
        Here you can apply basic data quality checks. 
        \n*   **Mean Imputation**: Fills missing numerical values with the mean of their respective columns.
        \n*   **Outlier Capping**: Extreme values (outliers) are capped at a specified number of standard deviations from the mean.
        This helps in reducing the impact of extreme values on model training.
        """)

        enable_mean_imputation = st.checkbox("Enable Mean Imputation for Missing Values", value=False)
        outlier_threshold = st.slider("Outlier Threshold (Standard Deviations)", min_value=1.0, max_value=5.0, value=3.0, step=0.1)

        if st.button("Apply Data Quality Checks"):
            with st.spinner("Applying data quality checks..."):
                try:
                    st.session_state.df_processed = perform_data_quality_checks(
                        st.session_state.df,
                        enable_imputation=enable_mean_imputation,
                        outlier_threshold=outlier_threshold
                    )
                    st.success("Data quality checks applied successfully!")
                    st.subheader("Processed Data Preview (after quality checks)")
                    st.write(st.session_state.df_processed.head())

                    st.subheader("Missing Values After Processing")
                    missing_values_after = st.session_state.df_processed.isnull().sum()
                    missing_values_df_after = pd.DataFrame({'Column': missing_values_after.index, 'Missing Count': missing_values_after.values})
                    missing_values_df_after = missing_values_df_after[missing_values_df_after['Missing Count'] > 0]
                    if not missing_values_df_after.empty:
                        st.dataframe(missing_values_df_after.set_index('Column'))
                    else:
                        st.info("No missing values after processing.")

                except TypeError as e:
                    st.error(f"Error during data quality checks: {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

        if "df_processed" in st.session_state and not st.session_state.df_processed.empty:
            st.subheader("Processed Data Statistics")
            st.write(st.session_state.df_processed.describe())
    else:
        st.info("Please load the dataset to proceed with data exploration.")
