# Wholesale Credit Rating Model Development

## Project Title and Description

This Streamlit application provides a platform for developing and evaluating wholesale credit rating models. It guides users through the process of data exploration, feature engineering, model training, and calibration, using the UCI Taiwan Credit Default dataset as an example. The application allows users to interact with data, explore feature engineering options, train and compare different machine learning models, and calibrate model outputs to generate a credit rating scale. The primary objective is to facilitate understanding of the model development process and the impact of different parameters on model outcomes.

## Features

*   **Data Exploration**:
    *   Load the UCI Taiwan Credit Default dataset.
    *   Display raw data previews, summary statistics, and missing value information.
    *   Perform data quality checks, including mean imputation and outlier capping.
*   **Feature Engineering**:
    *   Impute missing values using strategies like median imputation.
    *   Winsorize outlier values to reduce their impact.
    *   Calculate financial ratios (demonstrated with dummy data due to dataset limitations).
    *   Create a management experience score based on various factors.
    *   Transform skewed variables using quantile binning.
    *   Visualize feature distributions and assess univariate predictive power.
    *   Generate and download a data transformation log in YAML format.
*   **Model Training**:
    *   Split data into training and validation sets.
    *   Train Logistic Regression, Gradient Boosted Trees, and Random Forest models.
    *   Calculate and display Variance Inflation Factor (VIF) for multicollinearity assessment.
*   **Model Evaluation & Calibration**:
    *   Evaluate model performance using AUC and Gini coefficient.
    *   Generate ROC curves and Hosmer-Lemeshow calibration plots.
    *   Calibrate predicted probabilities to match observed default rates.
    *   Map calibrated probabilities to discrete credit rating grades.
    *   Save and load trained models and processed data.
    *   Includes placeholder for generating a model documentation report.

## Getting Started

### Prerequisites

*   Python 3.7 or higher
*   Streamlit
*   Pandas
*   NumPy
*   Scikit-learn
*   ucimlrepo
*   Plotly
*   Statsmodels
*   PyYAML
*   Joblib

### Installation

1.  Clone the repository:

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  Create a virtual environment (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  Install the required packages:

    ```bash
    pip install streamlit pandas numpy scikit-learn ucimlrepo plotly statsmodels pyyaml joblib
    ```

## Usage

1.  Navigate to the project directory in your terminal.

2.  Run the Streamlit application:

    ```bash
    streamlit run app.py
    ```

3.  The application will open in your web browser.

4.  Use the sidebar for navigation between the different sections: "Data Exploration", "Feature Engineering", and "Model Training".

5.  Follow the instructions within each section to load the dataset, perform transformations, train models, and evaluate results.

## Project Structure

```
credit_rating_model/
├── app.py                      # Main Streamlit application file
├── application_pages/
│   ├── data_exploration.py   # Data exploration functionality
│   ├── feature_engineering.py  # Feature engineering functionality
│   └── model_training.py     # Model training and evaluation functionality
├── blank.txt                   # Placeholder file - replace with content of model_training.py in a real deployment
├── README.md                   # This file
└── venv/                       # Virtual environment (if created)
```

## Technology Stack

*   **Streamlit**: For building the interactive web application.
*   **Pandas**: For data manipulation and analysis.
*   **NumPy**: For numerical computations.
*   **Scikit-learn**: For machine learning models and evaluation metrics.
*   **ucimlrepo**: For fetching the UCI Taiwan Credit Default dataset.
*   **Plotly**: For creating interactive visualizations.
*   **Statsmodels**: For statistical analysis (VIF Calculation).
*   **PyYAML**: For data logging in YAML format.
*   **Joblib**: For saving and loading trained models.

## Contributing

Contributions are welcome! Please follow these guidelines:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with descriptive messages.
4.  Submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE) - see the `LICENSE` file for details.

## Contact

*   [Your Name/Organization]
*   [Your Email/Website]
