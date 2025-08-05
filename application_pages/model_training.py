import streamlit as st
import joblib
import io
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.express as px


def split_data(X, y, test_size, random_state):
    """Splits the data into training and validation sets.
    Args:
        X: Features.
        y: Target variable.
        test_size: Proportion of data to use for validation.
        random_state: Random seed for reproducibility.
    Returns:
        Tuple of (X_train, X_val, y_train, y_val).
    """
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_logistic_regression(X_train, y_train, C_param, random_state):
    """Trains a logistic regression model.
    Args:
        X_train: Training features.
        y_train: Training target variable.
        C_param: Regularization parameter.
        random_state: Random seed.
    Returns:
        Trained logistic regression model.
    """
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=C_param, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_gradient_boosted_trees(X_train, y_train, random_state):
    """Trains a gradient boosted trees model.
    Args:
        X_train: Training features.
        y_train: Training target variable.
        random_state: Random seed.
    Returns:
        Trained gradient boosted trees model.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, random_state):
    """Trains a random forest model.
    Args:
        X_train: Training features.
        y_train: Training target variable.
        random_state: Random seed.
    Returns:
        Trained random forest model.
    """
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model


def calculate_vif(X):
    """Calculates the Variance Inflation Factor (VIF) for each feature.
    Args:
        X: Features.
    Returns:
        pandas Series with VIF values for each feature.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif = pd.Series([variance_inflation_factor(X.values, i)
                    for i in range(X.shape[1])], index=X.columns)
    return vif


def evaluate_model(model, X_val, y_val):
    """Evaluates the model on the validation set.
    Args:
        model: The trained model object.
        X_val: Validation features.
        y_val: Validation target variable.
    Returns:
        Tuple of (accuracy, AUC-ROC, precision, recall, F1-score).
    """
    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    accuracy = accuracy_score(y_val, y_pred)
    auc_roc = roc_auc_score(y_val, y_pred_proba)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    return accuracy, auc_roc, precision, recall, f1


def save_model(model, model_name):
    """Saves the trained model as a joblib file.
    Args:
        model: The trained model object.
        model_name: Name of the model (e.g., "Logistic_Regression").
    """
    filepath = f"{model_name}_model.joblib"
    try:
        with io.BytesIO() as buffer:
            joblib.dump(model, buffer)
            st.download_button(
                label=f"Download {model_name} Model",
                data=buffer.getvalue(),
                file_name=filepath,
                mime="application/octet-stream"
            )
        st.success(f"Model '{model_name}' is ready for download.")
    except Exception as e:
        st.error(f"Error saving model: {e}")


def calculate_auc_gini(y_val, y_pred_proba):
    """Calculates the AUC-ROC and Gini coefficient for the validation set.
    Args:
        y_val: Validation target variable.
        y_pred_proba: Validation predicted probabilities.
    Returns:
        Tuple of (AUC-ROC, Gini coefficient).
    """
    from sklearn.metrics import roc_auc_score
    auc_roc = roc_auc_score(y_val, y_pred_proba)
    gini = 2 * auc_roc - 1
    return auc_roc, gini


def calculate_calibration_curve(model, X_val, y_val):
    """Calculates the calibration curve for the validation set.
    Args:
        model: The trained model object.
        X_val: Validation features.
        y_val: Validation target variable.
    Returns:
        Tuple of (accuracy, AUC-ROC, precision, recall, F1-score).
    """
    from sklearn.calibration import calibration_curve
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    return calibration_curve(y_val, y_pred_proba)


def calculate_ks_statistic(y_val, y_pred_proba):
    """Calculates the KS statistic for the validation set.
    Args:
        y_val: Validation target variable.
        y_pred_proba: Validation predicted probabilities.
    Returns:
        Tuple of (accuracy, AUC-ROC, precision, recall, F1-score).
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
    ks_statistic = np.max(tpr - fpr)
    return ks_statistic


def calculate_ks_curve(y_val, y_pred_proba):
    """Calculates the KS curve for the validation set.
    Args:
        y_val: Validation target variable.
        y_pred_proba: Validation predicted probabilities.
    Returns:
        Tuple of (accuracy, AUC-ROC, precision, recall, F1-score).
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
    ks_curve = np.max(tpr - fpr)
    return ks_curve


def generate_roc_curve(y_true, y_pred_proba, title):
    """Generates ROC curves using Plotly.
    Args:
        y_true (array-like): True target values.
        y_pred_proba (array-like): Predicted probabilities of the positive class.
        title (str): Title of the plot.
    Returns:
        plotly.graph_objects.Figure: ROC curve plot.
    """
    if len(y_true) == 0 or len(y_pred_proba) == 0:
        st.warning("Input arrays for ROC curve cannot be empty.")
        return go.Figure()

    if len(y_true) != len(y_pred_proba):
        st.warning("Input arrays for ROC curve must have the same length.")
        return go.Figure()

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                             name=f'ROC curve (area = {roc_auc:.2f})', line=dict(color='darkorange', width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                             name='Random Classifier', line=dict(color='navy', width=2, dash='dash')))
    fig.update_layout(
        title=title,
        xaxis_title='False Positive Rate (FPR)',
        yaxis_title='True Positive Rate (TPR)',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        template='plotly_white',
        legend=dict(x=0.7, y=0.1)
    )
    return fig


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
        raise ValueError(
            "Predicted probabilities and actual default rates must have the same length.")

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
        st.write("Warning: Mean predicted probability is zero, cannot calibrate.")
        return predicted_probabilities

    scaling_factor = overall_observed_default_rate / overall_predicted_mean_proba
    calibrated_probabilities = predicted_probabilities * scaling_factor

    # Ensure probabilities remain within [0, 1]
    calibrated_probabilities = np.clip(calibrated_probabilities, 0, 1)

    st.write(
        f"Original Mean Predicted Probability: {overall_predicted_mean_proba:.4f}")
    st.write(
        f"Overall Observed Default Rate: {overall_observed_default_rate:.4f}")
    st.write(f"Calibration Scaling Factor: {scaling_factor:.4f}")
    st.write(
        f"Calibrated Mean Predicted Probability: {calibrated_probabilities.mean():.4f}")

    return calibrated_probabilities

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


def load_model(uploaded_file):
    """Loads a trained model from an uploaded file.
    Args:
        uploaded_file: The file object from st.file_uploader.
    Returns:
        The loaded model object.
    """
    if uploaded_file is not None:
        try:
            model = joblib.load(uploaded_file)
            st.success("Model loaded successfully!")
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
    return None


def save_data(df, filename="processed_data.csv"):
    """Saves the DataFrame as a CSV file for download.
    Args:
        df: pandas DataFrame to save.
        filename: Name of the file to save.
    """
    try:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"Download {filename}",
            data=csv,
            file_name=filename,
            mime="text/csv",
            key=f"download_{filename}"
        )
        st.success(f"Data '{filename}' is ready for download.")
    except Exception as e:
        st.error(f"Error saving data: {e}")


def run_model_training():
    st.header("3. Model Training, Evaluation & Calibration")

    st.markdown("""
    This section guides you through training different machine learning models, evaluating their performance,
    and calibrating their outputs to fit a credit rating scale.
    """)

    if "df_fe" not in st.session_state or st.session_state.df_fe.empty:
        st.warning(
            "Please complete 'Data Exploration' and 'Feature Engineering' first to prepare the data.")
        return

    df_final = st.session_state.df_fe.copy()
    target_col_name = 'Y'  # Assuming target column name

    if target_col_name not in df_final.columns:
        st.error(
            f"Target column \'{target_col_name}\' not found in the processed data. Cannot proceed with model training.")
        return

    X = df_final.drop(columns=[target_col_name], errors='ignore')
    y = df_final[target_col_name]

    # Ensure X contains only numeric data for training
    X = X.select_dtypes(include=np.number)
    if X.empty:
        st.error(
            "No numeric features available for model training after dropping target. Please check your data.")
        return

    st.markdown("---")
    st.subheader("Data Splitting")
    st.markdown("""
    Before training, the data is split into training and validation sets. 
    The training set is used to train the model, and the validation set is used to evaluate its performance on unseen data.
    Stratified splitting ensures that the proportion of the target class is similar in both sets.
    """)
    test_size = st.slider("Validation Set Size (%)",
                          min_value=10, max_value=50, value=30, step=5) / 100
    random_state = st.number_input(
        "Random State for Reproducibility", min_value=0, value=42, step=1)

    if st.button("Split Data", key="split_data_button"):
        try:
            X_train, X_val, y_train, y_val = split_data(
                X, y, test_size, random_state)
            st.session_state.X_train, st.session_state.X_val = X_train, X_val
            st.session_state.y_train, st.session_state.y_val = y_train, y_val
            st.success(
                f"Data split into training ({len(X_train)} samples) and validation ({len(X_val)} samples) sets.")
            st.write(
                f"Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}")
            st.write(
                f"Training target distribution:\n{y_train.value_counts(normalize=True)}")
            st.write(
                f"Validation target distribution:\n{y_val.value_counts(normalize=True)}")
        except Exception as e:
            st.error(f"Error splitting data: {e}")

    if "X_train" not in st.session_state:
        st.info("Please split the data to proceed with model training.")
        return

    st.markdown("---")
    st.subheader("Model Training")
    st.markdown("""
    Select a machine learning model to train and adjust its hyperparameters. The trained model will be used
    to predict default probabilities.
    """)
    model_type = st.radio("Select Model Type", [
                          "Logistic Regression", "Gradient Boosted Trees", "Random Forest"], key="model_type")

    # Model-specific parameters
    trained_model = None
    if model_type == "Logistic Regression":
        C_param = st.slider("Regularization Parameter (C)",
                            min_value=0.01, max_value=10.0, value=1.0, step=0.01)
        if st.button("Train Logistic Regression Model", key="train_lr"):
            with st.spinner("Training Logistic Regression model..."):
                trained_model = train_logistic_regression(
                    st.session_state.X_train, st.session_state.y_train, C_param, random_state)
                st.session_state.trained_model = trained_model
                st.session_state.model_name = "Logistic Regression"
                st.success("Logistic Regression model trained!")

    elif model_type == "Gradient Boosted Trees":
        if st.button("Train Gradient Boosted Trees Model", key="train_gbt"):
            with st.spinner("Training Gradient Boosted Trees model..."):
                trained_model = train_gradient_boosted_trees(
                    st.session_state.X_train, st.session_state.y_train, random_state)
                st.session_state.trained_model = trained_model
                st.session_state.model_name = "Gradient Boosted Trees"
                st.success("Gradient Boosted Trees model trained!")

    elif model_type == "Random Forest":
        if st.button("Train Random Forest Model", key="train_rf"):
            with st.spinner("Training Random Forest model..."):
                trained_model = train_random_forest(
                    st.session_state.X_train, st.session_state.y_train, random_state)
                st.session_state.trained_model = trained_model
                st.session_state.model_name = "Random Forest"
                st.success("Random Forest model trained!")

    if "trained_model" not in st.session_state or st.session_state.trained_model is None:
        st.info("Train a model to proceed with evaluation and calibration.")
        return

    st.markdown("---")
    st.subheader("Multicollinearity Assessment (VIF)")
    st.markdown("""
    Multicollinearity occurs when independent variables are highly correlated with each other, which can impact the stability and interpretability of model coefficients.
    The Variance Inflation Factor (VIF) quantifies the severity of multicollinearity.
    
    *   A VIF value of 1 means there is no correlation between the feature and any other features.
    *   VIF values between 1 and 5 are generally considered acceptable.
    *   VIF values greater than 5 (or 10, depending on the context) indicate high multicollinearity.
    """)

    if st.button("Calculate and Display VIF", key="calculate_vif_button"):
        with st.spinner("Calculating VIF values..."):
            try:
                vif_series = calculate_vif(st.session_state.X_train)
                if not vif_series.empty:
                    st.write(
                        "VIF Values (Higher values indicate more multicollinearity):")
                    st.dataframe(vif_series.to_frame(name='VIF'))

                    fig_vif = px.bar(vif_series.reset_index().rename(columns={'index': 'Feature', 0: 'VIF'}),
                                     x='Feature', y='VIF', title='Variance Inflation Factor (VIF)',
                                     color='VIF', color_continuous_scale=px.colors.sequential.Plasma,
                                     template='plotly_white')
                    fig_vif.add_hline(y=5, line_dash="dot", line_color="red", annotation_text="VIF > 5 (High Multicollinearity)",
                                      annotation_position="top right", annotation_font_color="red")
                    st.plotly_chart(fig_vif, use_container_width=True)
                else:
                    st.info(
                        "Could not calculate VIF values, possibly due to no numeric features or constant features.")
            except Exception as e:
                st.error(f"Error calculating VIF: {e}")

    st.markdown("---")
    st.subheader("Model Evaluation")
    st.markdown("""
    After training, it's crucial to evaluate the model's performance. This section displays key metrics like AUC and Gini coefficient,
    along with visual aids such as ROC curves and calibration plots.
    
    *   **AUC (Area Under the Receiver Operating Characteristic Curve):** Measures the ability of the model to distinguish between positive and negative classes.
        A higher AUC indicates better model performance.
    *   **Gini Coefficient:** Derived from AUC (Gini = 2 * AUC - 1), it provides a measure of how well the model discriminates between defaulters and non-defaulters.
    *   **ROC Curve:** Plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings.
    *   **Hosmer-Lemeshow Calibration Plot:** Assesses how well the predicted probabilities match the observed probabilities across different bins.
    """)

    if st.button("Evaluate Model", key="evaluate_model_button"):
        with st.spinner("Evaluating model performance..."):
            try:
                # Get predictions
                y_train_pred_proba = st.session_state.trained_model.predict_proba(
                    st.session_state.X_train)[:, 1]
                y_val_pred_proba = st.session_state.trained_model.predict_proba(
                    st.session_state.X_val)[:, 1]

                # Calculate AUC and Gini
                auc_train, gini_train = calculate_auc_gini(
                    st.session_state.y_train, y_train_pred_proba)
                auc_val, gini_val = calculate_auc_gini(
                    st.session_state.y_val, y_val_pred_proba)

                st.session_state.y_train_pred_proba = y_train_pred_proba
                st.session_state.y_val_pred_proba = y_val_pred_proba

                # Display Model Performance Table
                st.markdown("#### Model Performance Metrics")
                performance_df = pd.DataFrame({
                    'Metric': ['AUC', 'Gini Coefficient'],
                    'Training Set': [auc_train, gini_train],
                    'Validation Set': [auc_val, gini_val]
                }).set_index('Metric')
                st.dataframe(performance_df.round(4))

                # Generate ROC Curves
                st.markdown("#### ROC Curves")
                fig_roc_train = generate_roc_curve(
                    st.session_state.y_train, y_train_pred_proba, "ROC Curve - Training Set")
                st.plotly_chart(fig_roc_train, use_container_width=True)
                fig_roc_val = generate_roc_curve(
                    st.session_state.y_val, y_val_pred_proba, "ROC Curve - Validation Set")
                st.plotly_chart(fig_roc_val, use_container_width=True)

                # Generate Calibration Plots
                st.markdown("#### Hosmer-Lemeshow Calibration Plots")
                n_bins_calibration = st.slider(
                    "Number of Bins for Calibration Plot", min_value=5, max_value=20, value=10, step=1, key="n_bins_calib")
                fig_calib_train = generate_calibration_plot(
                    st.session_state.y_train, y_train_pred_proba, n_bins_calibration, "Calibration Plot - Training Set")
                st.plotly_chart(fig_calib_train, use_container_width=True)
                fig_calib_val = generate_calibration_plot(
                    st.session_state.y_val, y_val_pred_proba, n_bins_calibration, "Calibration Plot - Validation Set")
                st.plotly_chart(fig_calib_val, use_container_width=True)

            except Exception as e:
                st.error(f"Error during model evaluation: {e}")

    if "y_val_pred_proba" not in st.session_state:
        st.info("Evaluate the model to see evaluation metrics and plots.")
        return

    st.markdown("---")
    st.subheader("Model Calibration & Rating Grade Mapping")
    st.markdown("""
    Model calibration adjusts predicted probabilities to align with observed default rates.
    This ensures that a predicted probability of, say, 10% truly corresponds to a 10% default rate among similar obligors.
    Following calibration, these probabilities are mapped to discrete credit rating grades.
    
    *   The `calibrate_pds` function here applies a simple scaling to match the overall observed default rate.
    *   `map_pd_to_rating_grades` uses a quantile-based approach to assign probabilities to a user-defined number of grades.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Predicted Probabilities (Before Calibration)")
        if "y_val_pred_proba" in st.session_state:
            st.write(pd.Series(st.session_state.y_val_pred_proba).describe())
            fig_dist_raw = px.histogram(x=st.session_state.y_val_pred_proba, nbins=50, title="Distribution of Raw Predicted PDs",
                                        labels={'x': 'Predicted Probability of Default'}, template='plotly_white')
            st.plotly_chart(fig_dist_raw, use_container_width=True)

    with col2:
        st.markdown("#### Calibrated Probabilities")
        if st.button("Calibrate Probabilities", key="calibrate_pd_button"):
            with st.spinner("Calibrating PDs..."):
                try:
                    calibrated_probs = calibrate_pds(
                        st.session_state.y_val_pred_proba, st.session_state.y_val.values)
                    st.session_state.calibrated_probs = calibrated_probs
                    st.write(pd.Series(calibrated_probs).describe())
                    fig_dist_calib = px.histogram(x=calibrated_probs, nbins=50, title="Distribution of Calibrated PDs",
                                                  labels={'x': 'Calibrated Probability of Default'}, template='plotly_white')
                    st.plotly_chart(fig_dist_calib, use_container_width=True)
                except Exception as e:
                    st.error(f"Error calibrating probabilities: {e}")
        else:
            st.info("Press 'Calibrate Probabilities' to see calibrated PDs.")

    st.markdown("---")
    st.markdown("#### Mapping PDs to Rating Grades")
    num_grades = st.number_input(
        "Number of Rating Grades", min_value=2, max_value=15, value=7, step=1, key="num_grades")

    if "calibrated_probs" in st.session_state and st.button("Map to Rating Grades", key="map_grades_button"):
        with st.spinner("Mapping PDs to rating grades..."):
            try:
                rating_grades, grade_cutoffs = map_pd_to_rating_grades(
                    pd.Series(st.session_state.calibrated_probs), num_grades)
                st.session_state.rating_grades = rating_grades
                st.session_state.grade_cutoffs = grade_cutoffs
                st.success(
                    f"Probabilities mapped to {num_grades} rating grades.")

                st.markdown("##### Rating Grade Distribution")
                grade_counts = rating_grades.value_counts().sort_index().reset_index()
                grade_counts.columns = ['Rating Grade', 'Obligor Count']

                fig_grade_dist = px.bar(grade_counts, x='Rating Grade', y='Obligor Count',
                                        title='Obligor Counts Per Rating Grade',
                                        template='plotly_white')
                st.plotly_chart(fig_grade_dist, use_container_width=True)

                st.markdown("##### Grade Cutoffs")
                st.dataframe(grade_cutoffs.to_frame(name='Probability Cutoff'))

            except Exception as e:
                st.error(f"Error mapping PDs to rating grades: {e}")

    st.markdown("---")
    st.subheader("Saving and Loading Model Artifacts")
    st.markdown("""
    It is critical to save trained models and processed data for future use and reproducibility.
    
    *   **Saving Model:** Exports the trained machine learning model (e.g., Logistic Regression) to a file (e.g., .joblib).
        This allows you to load and reuse the model without retraining.
    *   **Loading Model:** Imports a previously saved model from a file.
    *   **Saving Data:** Exports the final processed dataset, including engineered features, to a CSV file.
        This ensures that the exact data used for training and testing can be accessed later.
    """)

    if "trained_model" in st.session_state and st.session_state.trained_model is not None:
        save_model(st.session_state.trained_model,
                   st.session_state.model_name.replace(" ", "_"))

    st.markdown("#### Save Processed Data")
    if "df_fe" in st.session_state and not st.session_state.df_fe.empty:
        save_data(st.session_state.df_fe, "final_processed_data.csv")
    else:
        st.info("No processed data to save yet. Complete Feature Engineering.")
