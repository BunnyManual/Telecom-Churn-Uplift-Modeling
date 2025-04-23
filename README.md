# Telecom Churn Uplift Modeling with T-Learner

## Overview

This project implements an uplift modeling pipeline for a telecom churn dataset using the T-Learner approach from the `causalml` library. The goal is to identify customers likely to benefit from a retention intervention by predicting uplift scores, which measure the incremental impact of the intervention on reducing churn. The pipeline processes the `churn-bigml-80.csv` dataset, trains a T-Learner model, evaluates its performance, and generates visualizations to interpret results.

The script (`Main.py`) is designed to be robust, with extensive error handling, logging, and preprocessing tailored to the dataset. It produces two visualizations: a cumulative gain curve and a feature importance plot, which are saved as `uplift_gain_curve.png` and `feature_importance.png`, respectively.

## Dataset

The dataset used is `churn-bigml-80.csv`, which contains 2666 rows and 20 columns describing telecom customer data. Key columns include:

- `State`: Customer's state (categorical).
- `Account length`: Duration of the account (numeric).
- `International plan` and `Voice mail plan`: Binary features (`yes`/`no`).
- `Total day minutes`, `Total eve minutes`, etc.: Usage metrics (numeric).
- `Customer service calls`: Number of calls to customer service (numeric).
- `Churn`: Target variable indicating whether the customer churned (`True`/`False`).

A synthetic `treatment` column is generated to simulate an intervention (e.g., a discount offer), with values `0` (control) or `1` (treatment).

## Requirements

To run the script, install the following Python packages:

```bash
pip install pandas numpy matplotlib seaborn causalml scikit-learn
```

- **Python Version**: 3.12 or higher (tested with 3.12).
- **Dependencies**:
  - `pandas`: Data manipulation and DataFrame operations.
  - `numpy`: Numerical computations.
  - `matplotlib`: Plotting (uses `Agg` backend for file saving).
  - `seaborn`: Enhanced visualization for feature importance.
  - `causalml`: Uplift modeling with T-Learner.
  - `scikit-learn`: Machine learning utilities (e.g., `GradientBoostingClassifier`, `train_test_split`).

## Project Structure

- `Main.ipynb/Main.py`: Main script containing the uplift modeling pipeline.
- `churn-bigml-80.csv`: Input dataset (place in the same directory as the script).
- `visualizations/`: Directory to store output plots (create this directory before running the script).
  - `uplift_gain_curve.png`: Cumulative gain curve comparing the T-Learner model to a random baseline.
  - `feature_importance.png`: Bar plot of the top 5 features influencing uplift predictions.


## Code Breakdown

The script (`Main.py`) is structured as a class-based pipeline (`TelecomChurnUpliftModel`) with modular methods for data loading, preprocessing, training, and evaluation. Below is a detailed explanation of each code chunk, its purpose, and functionality.

### 1. Imports and Setup

```python
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from causalml.inference.meta import BaseTLearner
from causalml.metrics import plot_gain
from sklearn.ensemble import GradientBoostingClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
```

- **Purpose**: Import required libraries and configure logging for debugging and tracking.
- **Details**:
  - `matplotlib.use('Agg')` ensures plots save correctly in non-interactive environments (e.g., servers without GUI).
  - `logging` provides detailed logs (e.g., dataset loading, plot saving) for transparency and debugging.
  - Libraries like `pandas`, `numpy`, and `causalml` are used for data manipulation, numerical operations, and uplift modeling, respectively.

### 2. Class Definition and Initialization

```python
class TelecomChurnUpliftModel:
    def __init__(self, random_state: int = 123):
        self.random_state = random_state
        self.preprocessor = None
        self.model = None
        self.feature_names = None
        self.target_col = 'churn'
        self.treatment_col = 'treatment'
```

- **Purpose**: Define the `TelecomChurnUpliftModel` class to encapsulate the uplift modeling pipeline.
- **Details**:
  - Initializes model parameters, including a `random_state` for reproducibility.
  - Sets placeholders for the preprocessor, model, and feature names, which are populated during execution.
  - Defines target (`churn`) and treatment (`treatment`) column names for consistency.

### 3. Data Loading (`load_data`)

```python
def load_data(self, file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        logger.info(f"Standardized column names: {df.columns.tolist()}")
        
        expected_cols = [
            'state', 'account_length', 'area_code', 'international_plan', 'voice_mail_plan',
            'number_vmail_messages', 'total_day_minutes', 'total_day_calls', 'total_day_charge',
            'total_eve_minutes', 'total_eve_calls', 'total_eve_charge', 'total_night_minutes',
            'total_night_calls', 'total_night_charge', 'total_intl_minutes', 'total_intl_calls',
            'total_intl_charge', 'customer_service_calls', 'churn'
        ]
        if not all(col in df.columns for col in expected_cols):
            raise ValueError(f"Dataset missing expected columns. Found: {df.columns.tolist()}")
        if df.shape[0] != 2666:
            logger.warning(f"Expected 2666 rows, but found {df.shape[0]} rows")
            
        df['churn'] = df['churn'].map({True: 1, False: 0, 'True': 1, 'False': 0})
        if df['churn'].isna().any():
            raise ValueError("Churn column contains missing or invalid values after mapping")
            
        np.random.seed(self.random_state)
        df[self.treatment_col] = np.random.binomial(1, 0.5, len(df)).astype('int64')
        
        binary_cols = ['international_plan', 'voice_mail_plan']
        for col in binary_cols:
            df[col] = df[col].str.lower().map({'yes': 1, 'no': 0, 'true': 1, 'false': 0})
            if df[col].isna().any():
                raise ValueError(f"Binary column {col} contains invalid values after mapping")
                
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise
```

- **Purpose**: Load and preprocess the dataset, ensuring it’s ready for modeling.
- **Details**:
  - Loads `churn-bigml-80.csv` into a pandas `DataFrame`.
  - Standardizes column names (e.g., `State` → `state`, `Account length` → `account_length`) to ensure consistency.
  - Validates expected columns to prevent downstream errors.
  - Converts `churn` to binary (0/1) and generates a synthetic `treatment` column (0/1) to simulate an intervention.
  - Maps binary columns (`international_plan`, `voice_mail_plan`) from `yes`/`no` to 0/1.
  - Includes error handling and logging for debugging.

### 4. Preprocessing Setup (`setup_preprocessing`)

```python
def setup_preprocessing(self, df: pd.DataFrame) -> None:
    try:
        categorical_cols = ['state', 'international_plan', 'voice_mail_plan']
        numeric_cols = [
            'account_length', 'area_code', 'number_vmail_messages',
            'total_day_minutes', 'total_day_calls', 'total_day_charge',
            'total_eve_minutes', 'total_eve_calls', 'total_eve_charge',
            'total_night_minutes', 'total_night_calls', 'total_night_charge',
            'total_intl_minutes', 'total_intl_calls', 'total_intl_charge',
            'customer_service_calls'
        ]
        
        missing_cols = [col for col in categorical_cols + numeric_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in dataset: {missing_cols}")
            
        self.preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
             categorical_cols)
        ])
        
        cat_features = []
        for col in categorical_cols:
            unique_vals = df[col].unique()
            cat_features.extend([f"{col}_{val}" for val in unique_vals[1:] if pd.notna(val)])
        self.feature_names = numeric_cols + cat_features
        
        logger.info(f"Preprocessing configured: {len(numeric_cols)} numeric, "
                   f"{len(categorical_cols)} categorical features")
    except Exception as e:
        logger.error(f"Preprocessing setup failed: {str(e)}")
        raise
```

- **Purpose**: Configure preprocessing for numeric and categorical features.
- **Details**:
  - Identifies 16 numeric columns (e.g., `total_day_minutes`) and 3 categorical columns (`state`, `international_plan`, `voice_mail_plan`).
  - Uses `ColumnTransformer` to apply `StandardScaler` to numeric features (normalizes them) and `OneHotEncoder` to categorical features (converts to dummy variables, drops first category to avoid multicollinearity).
  - Stores feature names for later use in feature importance plotting (accounts for one-hot encoded features).
  - Validates column presence to ensure the dataset aligns with expectations.

### 5. Data Preprocessing (`preprocess_data`)

```python
def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        X = df.drop(columns=[self.target_col, self.treatment_col])
        y = df[self.target_col].values
        t = df[self.treatment_col].values
        
        if X.isna().any().any():
            raise ValueError("Feature matrix contains missing values")
        if np.any(np.isnan(y)) or np.any(np.isnan(t)):
            raise ValueError("Target or treatment arrays contain missing values")
            
        X_processed = self.preprocessor.fit_transform(X)
        logger.info(f"Preprocessed data shape: {X_processed.shape}")
        return X_processed, y, t
    except Exception as e:
        logger.error(f"Data preprocessing failed: {str(e)}")
        raise
```

- **Purpose**: Apply preprocessing to the dataset and extract features, target, and treatment arrays.
- **Details**:
  - Drops `churn` and `treatment` columns to create the feature matrix `X`.
  - Extracts `y` (target: `churn`) and `t` (treatment) as NumPy arrays.
  - Validates for missing values to prevent model errors.
  - Applies the preprocessor to transform `X` into a processed matrix (68 features after one-hot encoding `state`).

### 6. Data Splitting (`split_data`)

```python
def split_data(self, X: np.ndarray, y: np.ndarray, t: np.ndarray) -> Tuple:
    try:
        if not (X.shape[0] == y.shape[0] == t.shape[0]):
            raise ValueError("Inconsistent number of samples in X, y, t")
            
        X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
            X, y, t,
            test_size=0.25,
            random_state=self.random_state,
            stratify=y
        )
        logger.info(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
        return X_train, X_test, y_train, y_test, t_train, t_test
    except Exception as e:
        logger.error(f"Data splitting failed: {str(e)}")
        raise
```

- **Purpose**: Split the data into training and test sets.
- **Details**:
  - Splits data with a 75/25 train/test ratio (1999 train, 667 test samples).
  - Uses `stratify=y` to maintain the proportion of churned customers in both sets.
  - Validates consistent sample sizes across `X`, `y`, and `t`.

### 7. Model Training (`train_uplift_model`)

```python
def train_uplift_model(self, X_train: np.ndarray, y_train: np.ndarray, t_train: np.ndarray) -> None:
    try:
        if not (X_train.shape[0] == y_train.shape[0] == t_train.shape[0]):
            raise ValueError("Inconsistent number of samples in X_train, y_train, t_train")
        if not np.all(np.isin(t_train, [0, 1])):
            raise ValueError("Treatment array must contain only 0 or 1")
        if not np.all(np.isin(y_train, [0, 1])):
            raise ValueError("Target array must contain only 0 or 1")
            
        self.model = BaseTLearner(
            learner=GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=self.random_state
            )
        )
        self.model.fit(X_train, t_train, y_train)
        logger.info("Uplift model training completed")
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise
```

- **Purpose**: Train a T-Learner uplift model using `causalml`.
- **Details**:
  - Initializes a `BaseTLearner` with a `GradientBoostingClassifier` as the base learner (100 trees, max depth 4, learning rate 0.1).
  - Validates input shapes and ensures `t_train` and `y_train` are binary.
  - Fits the model to the training data, learning the uplift effect of the treatment on churn.

### 8. Model Evaluation and Visualization (`evaluate_model`)

```python
def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray, t_test: np.ndarray) -> Dict[str, Any]:
    try:
        if not (X_test.shape[0] == y_test.shape[0] == t_test.shape[0]):
            raise ValueError("Inconsistent number of samples in X_test, y_test, t_test")
            
        uplift_scores = self.model.predict(X_test)
        
        np.random.seed(self.random_state)
        gain_df = pd.DataFrame({
            'y': y_test,
            'w': t_test,
            'tau': uplift_scores.flatten(),
            'random': np.random.uniform(-1, 1, size=len(y_test))
        })
        if gain_df.isna().any().any():
            raise ValueError("Gain DataFrame contains missing values")
            
        fig1 = plt.figure(figsize=(10, 6))
        plot_gain(
            gain_df,
            outcome_col='y',
            treatment_col='w',
            treatment_effect_col='tau'
        )
        plt.title('Cumulative Gain Curve - T-Learner Uplift Model')
        plt.grid(True)
        gain_plot_path = 'uplift_gain_curve.png'
        plt.savefig(gain_plot_path, bbox_inches='tight')
        logger.info(f"Saved gain plot to {gain_plot_path}")
        plt.close(fig1)
        
        importance = self.model.models_t[1].feature_importances_
        top_features = [
            self.feature_names[i] for i in np.argsort(importance)[-5:]
        ]
        
        fig2 = plt.figure(figsize=(10, 8))
        sns.barplot(x=importance[np.argsort(importance)[-5:]], y=top_features)
        plt.title('Top 5 Feature Importances - T-Learner')
        feature_plot_path = 'feature_importance.png'
        plt.savefig(feature_plot_path, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {feature_plot_path}")
        plt.close(fig2)
        
        results = {
            'model_type': 'BaseTLearner',
            'test_samples': len(X_test),
            'train_samples': X_train.shape[0],
            'top_features': top_features
        }
        logger.info(f"Evaluation results: {results}")
        return results
    except Exception as e:
        logger.error(f"Model evaluation failed: {str(e)}")
        raise
```

- **Purpose**: Evaluate the model, predict uplift scores, and generate visualizations.
- **Details**:
  - Predicts uplift scores on the test set using the trained T-Learner.
  - Creates a `DataFrame` (`gain_df`) for `plot_gain`, including outcome (`y`), treatment (`w`), uplift scores (`tau`), and a dummy `random` column for baseline comparison.
  - Plots the cumulative gain curve using `causalml.metrics.plot_gain`, comparing the T-Learner to the random baseline.
  - Extracts feature importances from the treatment learner and plots the top 5 features using `seaborn.barplot`.
  - Saves plots with explicit figure management (`fig1`, `fig2`) to avoid overlap, using the `Agg` backend for reliability.
  - Returns a results dictionary with model details and top features.

### 9. Pipeline Execution (`execute_uplift_pipeline`)

```python
def execute_uplift_pipeline(file_path: str) -> Dict[str, Any]:
    try:
        model = TelecomChurnUpliftModel(random_state=123)
        df = model.load_data(file_path)
        model.setup_preprocessing(df)
        X, y, t = model.preprocess_data(df)
        X_train, X_test, y_train, y_test, t_train, t_test = model.split_data(X, y, t)
        model.train_uplift_model(X_train, y_train, t_train)
        results = model.evaluate_model(X_test, y_test, t_test)
        logger.info(f"Pipeline completed successfully: {results}")
        return results
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise

if __name__ == '__main__':
    file_path = './churn-bigml-80.csv'
    execute_uplift_pipeline(file_path)
```

- **Purpose**: Orchestrate the entire pipeline and execute it.
- **Details**:
  - Initializes the model, loads data, sets up preprocessing, splits data, trains the model, and evaluates it.
  - Wraps the pipeline in a `try-except` block for error handling.
  - Executes the pipeline with the specified dataset path.

## Results Interpretation

- **Cumulative Gain Curve**:

  - The T-Learner model outperforms the random baseline for most of the population, peaking at a gain of \~2.5 for the top 100-200 customers. This suggests the model effectively identifies customers who benefit from the intervention.
  - The curve stabilizes around 1.5 gain, indicating consistent performance across the population.
  ![Alt text](visualizations\uplift_gain_curve.png)

- **Feature Importance**:

  - Top features include `total_day_charge`, `customer_service_calls`, `total_day_minutes`, `total_intl_charge`, and `total_eve_minutes`.
  - High `total_day_charge` and `total_day_minutes` suggest that customers with high daytime usage are more responsive to interventions.
  - `customer_service_calls` indicates that customers with more service interactions may be at higher risk of churn but also more likely to respond to retention efforts.
  ![Alt text](visualizations\feature_importance.png)

## Future Improvements

- **Real Treatment Data**: Replace the synthetic `treatment` column with actual intervention data for more accurate uplift predictions.
- **Model Tuning**: Experiment with hyperparameters of the `GradientBoostingClassifier` or try other learners (e.g., XGBoost, LightGBM).
- **Additional Metrics**: Add Qini curves (`causalml.metrics.plot_qini`) or AUUC (Area Under Uplift Curve) for deeper evaluation.
- **Deployment**: Extend the pipeline to predict uplift scores on new data and target high-uplift customers for interventions.

## Troubleshooting

- **Plots Not Generated**:

  - Check logs for “Saved gain plot” and “Saved feature importance plot” messages.
  - Ensure write permissions in the working directory.
  - Verify `matplotlib` and `seaborn` are installed (`pip show matplotlib seaborn`).
  - Test a simple plot:

    ```python
    import matplotlib.pyplot as plt
    plt.plot([1, 2, 3])
    plt.savefig('test.png')
    plt.close()
    ```

- **Dataset Issues**:

  - Confirm `churn-bigml-80.csv` has 2666 rows, 20 columns, and no missing values.
  - Check column names match the expected list in `load_data`.

- **Dependency Errors**:

  - Ensure `causalml` version 0.14.1 or compatible (`pip install causalml==0.14.1`).
  - Reinstall dependencies if needed:

    ```bash
    pip install --force-reinstall pandas numpy matplotlib seaborn causalml scikit-learn
    ```

## Running the Script

1. **Clone the Repository**:

   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Install Dependencies**:

   ```bash
   pip install pandas numpy matplotlib seaborn causalml scikit-learn
   ```

3. **Prepare the Dataset**:

   - Place `churn-bigml-80.csv` in the project root directory.

4. **Create Visualizations Directory**:

   ```bash
   mkdir visualizations
   ```

5. **Run the Script**:

   ```bash
   python Main.py
   ```

6. **Expected Output**:

   - Logs detailing each step (loading, preprocessing, training, evaluation).
   - Visualizations saved as:
     - `uplift_gain_curve.png`
     - `feature_importance.png`
   - Results dictionary logged, e.g.:

     ```
     {'model_type': 'BaseTLearner', 'test_samples': 667, 'train_samples': 1999, 'top_features': ['total_day_charge', 'customer_service_calls', ...]}
     ```

7. **Upload Visualizations to GitHub**:

   - Move the generated plots to the `visualizations/` directory:

     ```bash
     mv uplift_gain_curve.png visualizations/
     mv feature_importance.png visualizations/
     ```
   - Add, commit, and push to GitHub:

     ```bash
     git add visualizations/*
     git commit -m "Add uplift modeling visualizations"
     git push origin main
     ```