# %%
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

# %%
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# %%
class TelecomChurnUpliftModel:
    def __init__(self, random_state: int = 123):
        self.random_state = random_state
        self.preprocessor = None
        self.model = None
        self.feature_names = None
        self.target_col = 'churn'
        self.treatment_col = 'treatment'
        
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


