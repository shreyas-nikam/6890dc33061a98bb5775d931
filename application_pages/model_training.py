cutoff' to define a new grade, but the upper bound of the last grade.
    # So, we are interested in bins[1:] as the grade cutoffs.
    grade_cutoffs_list = bins[1:].tolist()

    # Create a Series for grade cutoffs for display
    grade_cutoffs_series = pd.Series(grade_cutoffs_list, 
                                      index=[f'Grade {i+1} Upper Cutoff' for i in range(len(grade_cutoffs_list))])

    return grades, grade_cutoffs_series

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
        st.warning("Please complete 'Data Exploration' and 'Feature Engineering' first to prepare the data.")
        return
    
    df_final = st.session_state.df_fe.copy()
    target_col_name = 'default payment next month' # Assuming target column name

    if target_col_name not in df_final.columns:
        st.error(f"Target column \'{target_col_name}\' not found in the processed data. Cannot proceed with model training.")
        return

    X = df_final.drop(columns=[target_col_name], errors='ignore')
    y = df_final[target_col_name]

    # Ensure X contains only numeric data for training
    X = X.select_dtypes(include=np.number)
    if X.empty:
        st.error("No numeric features available for model training after dropping target. Please check your data.")
        return

    st.markdown("---")
    st.subheader("Data Splitting")
    st.markdown("""
    Before training, the data is split into training and validation sets. 
    The training set is used to train the model, and the validation set is used to evaluate its performance on unseen data.
    Stratified splitting ensures that the proportion of the target class is similar in both sets.
    """)
    test_size = st.slider("Validation Set Size (%)", min_value=10, max_value=50, value=30, step=5) / 100
    random_state = st.number_input("Random State for Reproducibility", min_value=0, value=42, step=1)

    if st.button("Split Data", key="split_data_button"):
        try:
            X_train, X_val, y_train, y_val = split_data(X, y, test_size, random_state)
            st.session_state.X_train, st.session_state.X_val = X_train, X_val
            st.session_state.y_train, st.session_state.y_val = y_train, y_val
            st.success(f"Data split into training ({len(X_train)} samples) and validation ({len(X_val)} samples) sets.")
            st.write(f"Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}")
            st.write(f"Training target distribution:\n{y_train.value_counts(normalize=True)}")
            st.write(f"Validation target distribution:\n{y_val.value_counts(normalize=True)}")
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
    model_type = st.radio("Select Model Type", ["Logistic Regression", "Gradient Boosted Trees", "Random Forest"], key="model_type")

    # Model-specific parameters
    trained_model = None
    if model_type == "Logistic Regression":
        C_param = st.slider("Regularization Parameter (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
        if st.button("Train Logistic Regression Model", key="train_lr"):
            with st.spinner("Training Logistic Regression model..."):
                trained_model = train_logistic_regression(st.session_state.X_train, st.session_state.y_train, C_param, random_state)
                st.session_state.trained_model = trained_model
                st.session_state.model_name = "Logistic Regression"
                st.success("Logistic Regression model trained!")

    elif model_type == "Gradient Boosted Trees":
        if st.button("Train Gradient Boosted Trees Model", key="train_gbt"):
            with st.spinner("Training Gradient Boosted Trees model..."):
                trained_model = train_gradient_boosted_trees(st.session_state.X_train, st.session_state.y_train, random_state)
                st.session_state.trained_model = trained_model
                st.session_state.model_name = "Gradient Boosted Trees"
                st.success("Gradient Boosted Trees model trained!")

    elif model_type == "Random Forest":
        if st.button("Train Random Forest Model", key="train_rf"):
            with st.spinner("Training Random Forest model..."):
                trained_model = train_random_forest(st.session_state.X_train, st.session_state.y_train, random_state)
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
                    st.write("VIF Values (Higher values indicate more multicollinearity):")
                    st.dataframe(vif_series.to_frame(name='VIF'))

                    fig_vif = px.bar(vif_series.reset_index().rename(columns={'index':'Feature', 0:'VIF'}),
                                     x='Feature', y='VIF', title='Variance Inflation Factor (VIF)',
                                     color='VIF', color_continuous_scale=px.colors.sequential.Plasma,
                                     template='plotly_white')
                    fig_vif.add_hline(y=5, line_dash="dot", line_color="red", annotation_text="VIF > 5 (High Multicollinearity)", 
                                     annotation_position="top right", annotation_font_color="red")
                    st.plotly_chart(fig_vif, use_container_width=True)
                else:
                    st.info("Could not calculate VIF values, possibly due to no numeric features or constant features.")
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
                y_train_pred_proba = st.session_state.trained_model.predict_proba(st.session_state.X_train)[:, 1]
                y_val_pred_proba = st.session_state.trained_model.predict_proba(st.session_state.X_val)[:, 1]

                # Calculate AUC and Gini
                auc_train, gini_train = calculate_auc_gini(st.session_state.y_train, y_train_pred_proba)
                auc_val, gini_val = calculate_auc_gini(st.session_state.y_val, y_val_pred_proba)

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
                fig_roc_train = generate_roc_curve(st.session_state.y_train, y_train_pred_proba, "ROC Curve - Training Set")
                st.plotly_chart(fig_roc_train, use_container_width=True)
                fig_roc_val = generate_roc_curve(st.session_state.y_val, y_val_pred_proba, "ROC Curve - Validation Set")
                st.plotly_chart(fig_roc_val, use_container_width=True)

                # Generate Calibration Plots
                st.markdown("#### Hosmer-Lemeshow Calibration Plots")
                n_bins_calibration = st.slider("Number of Bins for Calibration Plot", min_value=5, max_value=20, value=10, step=1, key="n_bins_calib")
                fig_calib_train = generate_calibration_plot(st.session_state.y_train, y_train_pred_proba, n_bins_calibration, "Calibration Plot - Training Set")
                st.plotly_chart(fig_calib_train, use_container_width=True)
                fig_calib_val = generate_calibration_plot(st.session_state.y_val, y_val_pred_proba, n_bins_calibration, "Calibration Plot - Validation Set")
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
                                        labels={'x':'Predicted Probability of Default'}, template='plotly_white')
            st.plotly_chart(fig_dist_raw, use_container_width=True)

    with col2:
        st.markdown("#### Calibrated Probabilities")
        if st.button("Calibrate Probabilities", key="calibrate_pd_button"):
            with st.spinner("Calibrating PDs..."):
                try:
                    calibrated_probs = calibrate_pds(st.session_state.y_val_pred_proba, st.session_state.y_val.values)
                    st.session_state.calibrated_probs = calibrated_probs
                    st.write(pd.Series(calibrated_probs).describe())
                    fig_dist_calib = px.histogram(x=calibrated_probs, nbins=50, title="Distribution of Calibrated PDs",
                                                labels={'x':'Calibrated Probability of Default'}, template='plotly_white')
                    st.plotly_chart(fig_dist_calib, use_container_width=True)
                except Exception as e:
                    st.error(f"Error calibrating probabilities: {e}")
        else:
            st.info("Press 'Calibrate Probabilities' to see calibrated PDs.")

    st.markdown("---")
    st.markdown("#### Mapping PDs to Rating Grades")
    num_grades = st.number_input("Number of Rating Grades", min_value=2, max_value=15, value=7, step=1, key="num_grades")

    if "calibrated_probs" in st.session_state and st.button("Map to Rating Grades", key="map_grades_button"):
        with st.spinner("Mapping PDs to rating grades..."):
            try:
                rating_grades, grade_cutoffs = map_pd_to_rating_grades(pd.Series(st.session_state.calibrated_probs), num_grades)
                st.session_state.rating_grades = rating_grades
                st.session_state.grade_cutoffs = grade_cutoffs
                st.success(f"Probabilities mapped to {num_grades} rating grades.")
                
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
        save_model(st.session_state.trained_model, st.session_state.model_name.replace(" ", "_"))

    st.markdown("#### Load a Model")
    uploaded_model_file = st.file_uploader("Upload a .joblib model file", type=["joblib"], key="upload_model")
    if uploaded_model_file is not None:
        loaded_model = load_model(uploaded_model_file)
        if loaded_model:
            st.session_state.loaded_model_for_display = loaded_model # Store for potential further use/display
            st.success("Model loaded successfully into session state for demonstration.")

    st.markdown("#### Save Processed Data")
    if "df_fe" in st.session_state and not st.session_state.df_fe.empty:
        save_data(st.session_state.df_fe, "final_processed_data.csv")
    else:
        st.info("No processed data to save yet. Complete Feature Engineering.")

    st.markdown("---")
    st.markdown("#### Documentation Report (Placeholder)")
    st.markdown("""
    Regulatory standards often require comprehensive model documentation. This section would typically generate a detailed report
    summarizing the data, methodologies, model performance, and validation results. For this lab, it serves as a placeholder.
    """)
    if st.button("Generate Model Documentation Report", key="generate_doc_report"):
        st.info("Generating model documentation report... (This is a placeholder for future functionality)")

