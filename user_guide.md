id: 6890dc33061a98bb5775d931_user_guide
summary: Lab 1.1: Rating models - Development User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# Credit Rating Model Codelab: A Step-by-Step User Guide

This codelab provides a comprehensive guide to using the "Wholesale Credit Rating Model Development" application. The application is designed to help you understand the key concepts and processes involved in building a credit rating model. By interacting with the application, you'll explore data, engineer features, train machine learning models, and calibrate outputs to create a credit rating scale. This hands-on experience will provide valuable insights into how different parameters and choices affect the final model outcome.

## Understanding the Application's Core Concepts
Duration: 00:05

Before diving into the application, it's important to understand the core concepts behind credit rating models. These models are used to assess the creditworthiness of borrowers, typically businesses or individuals. They predict the likelihood that a borrower will default on their debt obligations. Building these models involves several key steps:

1.  **Data Exploration:** Understanding the characteristics of the available data, identifying missing values, and detecting outliers.
2.  **Feature Engineering:** Transforming raw data into meaningful features that can be used by machine learning models. This often involves creating financial ratios, handling outliers, and binning variables.
3.  **Model Training:** Training machine learning models to predict the probability of default (PD) based on the engineered features. Common models include logistic regression, gradient boosted trees, and random forests.
4.  **Model Evaluation:** Assessing the performance of the trained models using metrics such as AUC (Area Under the Curve) and Gini coefficient.
5.  **Model Calibration:** Adjusting the predicted probabilities to match the observed default rates. This ensures that the model's outputs are well-aligned with the actual risk.
6.  **Rating Grade Mapping:** Assigning credit ratings (e.g., AAA, AA, A, etc.) to borrowers based on their calibrated PDs.

The application allows you to explore each of these steps in detail.

## Navigating the Application
Duration: 00:02

The application is divided into three main sections, accessible via the sidebar navigation:

*   **Data Exploration:** Load and explore the UCI Taiwan Credit Default dataset. Perform initial data quality checks.
*   **Feature Engineering:** Transform raw data into features using various techniques.
*   **Model Training:** Train, evaluate, and calibrate machine learning models.

Use the selectbox in the sidebar to move between these sections.

## Data Exploration
Duration: 00:10

The "Data Exploration" section allows you to load the UCI Taiwan Credit Default dataset and perform initial data quality checks.

1.  **Loading the Dataset:** Click the "Load UCI Taiwan Credit Default Dataset" button to load the data. This will fetch the dataset from the UCI repository and display a preview of the raw data.
2.  **Inspecting Missing Values:** The application displays a table showing the number of missing values in each column. This helps you identify columns that may require imputation.
3.  **Understanding the Data Shape:**  The number of rows and columns in the dataset is displayed, providing an overview of the data's size.
4.  **Reviewing Summary Statistics:**  The application displays descriptive statistics (mean, standard deviation, minimum, maximum, etc.) for each numerical column. This helps you understand the distribution of the data and identify potential outliers.
5.  **Applying Data Quality Checks:**
    *   **Enable Mean Imputation:** Check the "Enable Mean Imputation for Missing Values" box to fill missing values with the mean of their respective columns.
    *   **Outlier Threshold:** Use the "Outlier Threshold (Standard Deviations)" slider to set the threshold for outlier capping. Values beyond this threshold will be capped.
    *   Click the "Apply Data Quality Checks" button to apply the selected data quality checks.

<aside class="positive">
<b>Tip:</b> Experiment with different outlier thresholds to see how they affect the processed data. A common starting point is 3 standard deviations.
</aside>

After applying the data quality checks, the application displays a preview of the processed data, the number of missing values after processing, and the descriptive statistics of the processed data.

## Feature Engineering
Duration: 00:20

The "Feature Engineering" section focuses on transforming raw data into features that can be used by machine learning models.

1.  **Imputing Missing Values:** Select an imputation strategy (currently only "median" is available) and click the "Apply Missing Value Imputation" button to fill any remaining missing values.
2.  **Winsorizing Outlier Values:** Select the columns you want to winsorize and set the lower and upper quantiles. Click the "Apply Winsorization" button to cap extreme values at the specified percentiles.
3.  **Calculating Financial Ratios:** The UCI dataset doesn't include direct financial data, so this section uses dummy data based on `LIMIT_BAL` to illustrate the calculation of financial ratios. Click the "Calculate Ratios" button to generate these ratios. New columns will be added for Return on Assets (ROA), Debt-to-Equity Ratio, Current Ratio, and Cash Flow Coverage Ratio.
4.  **Creating a Management Experience Score:** Create a synthetic "Management Experience Score" based on `EDUCATION`, `MARRIAGE`, `AGE`, and average `PAY_X` variables. Adjust the weights for each variable to see how they influence the score. Click the "Create Score" button to generate the score.
5.  **Quantile Binning Skewed Variables:** Select the columns you want to bin and set the number of bins. Click the "Apply Quantile Binning" button to transform continuous numerical variables into discrete bins.
6.  **Feature Exploration Visualizations:**
    *   **Histograms and Box Plots:** Select a feature to visualize its distribution using histograms and box plots.
    *   **Univariate AUC/Gini for Predictors:** Click the "Calculate Univariate AUC/Gini" button to evaluate the predictive power of individual numerical features against the target variable (default).
7.  **Data Transformation Log:** Click the "Generate and Download Data Transformation Log" button to create a YAML file recording all the transformations applied to the dataset.

<aside class="negative">
<b>Warning:</b> Remember that the financial ratios in this application are calculated using dummy data. In a real-world scenario, you would use actual financial figures.
</aside>

## Model Training, Evaluation & Calibration
Duration: 00:30

The "Model Training, Evaluation & Calibration" section guides you through training different machine learning models, evaluating their performance, and calibrating their outputs to fit a credit rating scale.

1.  **Data Splitting:**
    *   Use the "Validation Set Size (%)" slider to set the proportion of data to be used for the validation set.
    *   Set the "Random State for Reproducibility" to ensure consistent results.
    *   Click the "Split Data" button to split the data into training and validation sets.
2.  **Model Training:**
    *   Select a model type from the "Select Model Type" radio buttons (Logistic Regression, Gradient Boosted Trees, or Random Forest).
    *   Adjust the model-specific parameters (e.g., the regularization parameter "C" for Logistic Regression).
    *   Click the "Train" button for the selected model to train the model.
3.  **Multicollinearity Assessment (VIF):**
    *   Click the "Calculate and Display VIF" button to assess multicollinearity among the features. High VIF values indicate strong correlations between features, which can impact model stability.
4.  **Model Evaluation:**
    *   Click the "Evaluate Model" button to assess the model's performance. The application will display the AUC and Gini coefficient for both the training and validation sets, along with ROC curves and Hosmer-Lemeshow calibration plots.
    *   Use the "Number of Bins for Calibration Plot" slider to adjust the number of bins used in the calibration plot.
5.  **Model Calibration & Rating Grade Mapping:**
    *   Click the "Calibrate Probabilities" button to adjust the predicted probabilities to match the observed default rates.
    *   Set the "Number of Rating Grades" to determine the number of credit rating grades.
    *   Click the "Map to Rating Grades" button to assign credit ratings to borrowers based on their calibrated PDs.
6.  **Saving and Loading Model Artifacts:**
    *   Click the "Download" button to save the trained model as a joblib file.
    *   Use the "Upload a .joblib model file" file uploader to load a previously saved model.
    *   Click the "Download" button to save the processed data as a CSV file.
7.  **Documentation Report:**
    *   Click the "Generate Model Documentation Report" button to generate a detailed report summarizing the data, methodologies, model performance, and validation results (this is a placeholder for future functionality).

<aside class="positive">
<b>Tip:</b> Compare the performance of different models and experiment with different hyperparameter settings to see how they affect the model's accuracy and calibration.
</aside>

By following these steps, you can effectively use the "Wholesale Credit Rating Model Development" application to gain a deeper understanding of the credit rating model development process.
