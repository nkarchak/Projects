import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import joblib # For saving the final model

# --- Configuration and Setup ---
# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define global constants for data splits
DEVELOPMENT_RATIO = 0.6
HOLDOUT_RATIO = 0.2
OOT_RATIO = 0.2
RANDOM_STATE = 42
TARGET_COLUMN = 'target'
MODEL_NAME = 'CreditRisk_Classifier_v1'

class ModelDevelopmentPipeline:
    """
    Class to encapsulate the technical steps of the ML Model Development Lifecycle.
    Corresponds mainly to sections 4, 5, 6, 7, 8, and 9 of the MDD outline.
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.datasets = {}
        self.preprocessor = None
        self.models = {}
        self.preferred_model = None

    # --- 4. Model Development Data ---

    def load_and_prepare_data(self):
        """4.a/d/e/f: Data Loading, Preparation, Cleaning, and Variable Definition."""
        logging.info("Starting Data Loading and Preparation.")
        try:
            # 4.a: Data Sources (Simulated data loading)
            self.data = pd.read_csv(self.data_path)
            logging.info(f"Initial data shape: {self.data.shape}")

            # 4.e: Data Cleaning and Treatment (Example: Handling missing values)
            # Assuming 'Income' is a key feature; fill missing with median (A simple Data Control/Treatment)
            median_income = self.data['Income'].median()
            self.data['Income'].fillna(median_income, inplace=True)

            # 4.f: Model Variables (Define feature types)
            self.numerical_features = ['Age', 'Income', 'CreditScore']
            self.categorical_features = ['EmploymentType', 'Region']

            # Define the preprocessing steps
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), self.numerical_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)
                ],
                remainder='passthrough'
            )
            logging.info("Data loaded and preprocessor defined.")

        except FileNotFoundError:
            logging.error(f"Data file not found at {self.data_path}")
            raise

    def create_datasets(self):
        """4.g: Creation of Datasets (Development, Holdout, OOT)"""
        logging.info("Splitting data into Development, Holdout, and OOT datasets.")

        # Separate features (X) and target (y)
        X = self.data.drop(TARGET_COLUMN, axis=1)
        y = self.data[TARGET_COLUMN]

        # 1. Split for OOT (Out-of-Time / Out-of-Sample)
        # Assuming the data is already sorted chronologically for a true OOT split.
        X_temp, X_oot, y_temp, y_oot = train_test_split(
            X, y, test_size=OOT_RATIO, random_state=RANDOM_STATE, stratify=y
        )

        # 2. Split temp data into Development and Holdout
        # Recalculate the test_size ratio for the remaining data
        holdout_from_temp = HOLDOUT_RATIO / (DEVELOPMENT_RATIO + HOLDOUT_RATIO)
        X_dev, X_holdout, y_dev, y_holdout = train_test_split(
            X_temp, y_temp, test_size=holdout_from_temp, random_state=RANDOM_STATE, stratify=y_temp
        )

        self.datasets = {
            'dev': (X_dev, y_dev),
            'holdout': (X_holdout, y_holdout),
            'oot': (X_oot, y_oot)
        }
        logging.info(f"Dataset sizes: Dev={len(X_dev)}, Holdout={len(X_holdout)}, OOT={len(X_oot)}")

    # --- 5 & 6. Model Methodology & Development ---

    def develop_candidate_models(self):
        """5.a/b/c & 6.b/d: Assess/Select Methodology and Train Candidate Models."""
        logging.info("Developing candidate models: Logistic Regression and Random Forest.")

        # 6.b: Define Candidate Models (Logistic Regression - GLM, Random Forest - Tree-based)
        models_dict = {
            'LogisticRegression': LogisticRegression(solver='liblinear', random_state=RANDOM_STATE),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        }

        # 6.d: Model Development Process (Training)
        X_dev, y_dev = self.datasets['dev']

        for name, model in models_dict.items():
            # Create a full pipeline (Preprocessing + Model)
            full_pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('classifier', model)
            ])
            logging.info(f"Training {name}...")
            full_pipeline.fit(X_dev, y_dev)
            self.models[name] = full_pipeline
            logging.info(f"Training of {name} completed.")

        # 5.c: Preferred Methodology (Example: Simple selection based on internal criteria)
        # In a real scenario, this involves cross-validation, hyperparameter tuning, and detailed risk assessment.
        self.preferred_model_name = 'RandomForest'
        self.preferred_model = self.models[self.preferred_model_name]
        logging.info(f"Preferred model selected: {self.preferred_model_name}")

    # --- 7. Model Testing and Conclusions ---

    def evaluate_model(self, model_name, X_data, y_true):
        """Helper function for testing on any dataset."""
        model_pipeline = self.models[model_name]
        y_pred_proba = model_pipeline.predict_proba(X_data)[:, 1]
        y_pred_class = model_pipeline.predict(X_data)

        # 7.a: Quantitative Test (AUC)
        auc = roc_auc_score(y_true, y_pred_proba)

        # 7.c: Feature Importance (Only for Tree-based models like Random Forest)
        feature_imp = None
        if model_name == 'RandomForest':
            # Get feature names after one-hot encoding
            feature_names_out = (
                list(self.numerical_features) +
                list(model_pipeline['preprocessor'].named_transformers_['cat'].get_feature_names_out(self.categorical_features))
            )
            # Get importance from the classifier step in the pipeline
            importances = model_pipeline['classifier'].feature_importances_
            feature_imp = pd.Series(importances, index=feature_names_out).sort_values(ascending=False)

        return {
            'auc': auc,
            'y_pred_proba': y_pred_proba,
            'y_pred_class': y_pred_class,
            'confusion_matrix': confusion_matrix(y_true, y_pred_class),
            'classification_report': classification_report(y_true, y_pred_class, zero_division=0),
            'feature_importance': feature_imp
        }

    def test_and_validate_models(self):
        """7.a/b/c/d/f: Execute testing steps."""
        logging.info("Starting Model Testing and Validation.")

        # 7.f: Model Performance on OOT Dataset (Final check before deployment)
        X_oot, y_oot = self.datasets['oot']
        oot_results = self.evaluate_model(self.preferred_model_name, X_oot, y_oot)

        logging.info(f"\n--- Model Test Results on OOT Set ({self.preferred_model_name}) ---")
        logging.info(f"AUC Score (7.a): {oot_results['auc']:.4f}")

        # Qualitative Check (7.b) & Bias Check (7.d) would be manual/external analysis
        # Example: Check confusion matrix to assess if the model has a higher false negative rate
        # for a protected group (Bias Check).
        logging.info(f"Confusion Matrix (7.a/b):\n{oot_results['confusion_matrix']}")
        
        # 7.c: Feature Importance Interpretation
        if oot_results['feature_importance'] is not None:
             logging.info(f"Top 5 Feature Importance (7.c):\n{oot_results['feature_importance'].head(5)}")

        # Check against Risk Appetite/Acceptance Criteria (2.d, 3.c)
        ACCEPTANCE_THRESHOLD_AUC = 0.75
        if oot_results['auc'] >= ACCEPTANCE_THRESHOLD_AUC:
            logging.info(f"✅ Model meets the Acceptance Criteria (AUC >= {ACCEPTANCE_THRESHOLD_AUC}).")
        else:
            logging.warning(f"❌ Model *DOES NOT* meet Acceptance Criteria (AUC < {ACCEPTANCE_THRESHOLD_AUC}).")


    # --- 8. Final Model ---

    def finalize_and_save_model(self):
        """8.a/b: Final Model Architecture and Implementation (Saving)."""
        logging.info("Finalizing and saving the preferred model.")
        
        # 8.b: Model Implementation (Save artifact)
        save_path = f'./{MODEL_NAME}.pkl'
        joblib.dump(self.preferred_model, save_path)
        logging.info(f"Final Model saved to: {save_path}")

        # 8.a: Model Architecture is defined by the preferred_model pipeline (Preprocessor + Classifier)
        logging.info(f"Final Model Architecture: {self.preferred_model}")


    # --- 9. Model Monitoring (Setup) ---

    def setup_monitoring(self):
        """9.b: Model Monitoring Arrangement (Placeholder for deployment setup)."""
        logging.info("Setting up placeholders for Model Monitoring.")
        
        # In a real system, this would involve setting up APIs, deployment jobs, 
        # and connecting to a monitoring dashboard for:
        # 1. Data Drift checks (Is the OOT data similar to new production data?)
        # 2. Model Performance Decay checks (Is AUC/Accuracy dropping?)
        # 3. Model Explainability/Bias checks (Is feature importance shifting?)
        
        logging.info("Monitoring setup completed. This involves deployment to a production environment (e.g., MLOps Platform) and defining automated triggers.")
        

# --- Execution ---

if __name__ == "__main__":
    # Simulate data creation for demonstration
    np.random.seed(RANDOM_STATE)
    N = 10000
    data_mock = pd.DataFrame({
        'Age': np.random.randint(20, 65, N),
        'Income': np.random.randint(30000, 150000, N),
        'CreditScore': np.random.randint(550, 800, N),
        'EmploymentType': np.random.choice(['Full-time', 'Part-time', 'Unemployed'], N, p=[0.7, 0.2, 0.1]),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], N),
        'target': np.random.choice([0, 1], N, p=[0.85, 0.15]) # Binary target
    })
    # Introduce some missing data (4.e Data Cleaning)
    data_mock.loc[data_mock.sample(frac=0.05).index, 'Income'] = np.nan
    data_mock.to_csv('mock_risk_data.csv', index=False)
    # End of simulation

    # Initialize the pipeline
    pipeline = ModelDevelopmentPipeline('mock_risk_data.csv')

    # 4. Model Development Data
    pipeline.load_and_prepare_data()
    pipeline.create_datasets()

    # 5 & 6. Model Methodology and Development
    pipeline.develop_candidate_models()

    # 7. Model Testing, Test results and Test conclusions
    pipeline.test_and_validate_models()

    # 8. Final Model
    pipeline.finalize_and_save_model()

    # 9. Model risk and Model Monitoring (Setup phase)
    pipeline.setup_monitoring()

    logging.info("\n*** End-to-End Technical Model Development Process Complete ***")
