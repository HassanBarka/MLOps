import os
import mlflow
import dagshub
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

dagshub.init(repo_owner='HassanBarka', repo_name='MLOps', mlflow=True)

def load_data(train_path: str, test_path: str):
    """Load train and test datasets"""
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        X_train = train_data.drop(columns=['score'], axis=1)
        y_train = train_data['score']
        X_test = test_data.drop(columns=['score'], axis=1)
        y_test = test_data['score']
        
        X_train = pd.DataFrame(X_train, columns=X_train.columns)
        X_test = pd.DataFrame(X_test, columns=X_test.columns)
        
        return X_train, X_test, y_train, y_test 
    except Exception as e:
        raise Exception(f"Error in data loading: {e}")

def create_models(n_estimators):
    """Create dictionary of models with their parameters"""
    models = {
        'RandomForest': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': n_estimators,
                'max_depth': 10,
                'min_samples_split': 2,
                'random_state': 42
            }
        }
    }
    return models

def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    return metrics

def train_model(X_train, y_train, model, params, sampling='none'):
    """Train a model with optional SMOTE sampling"""
    if sampling == 'smote':
        smote = SMOTE(random_state=42)
        X_train_sample, y_train_sample = smote.fit_resample(X_train, y_train)
    else:
        X_train_sample, y_train_sample = X_train, y_train
    
    clf = model.__class__(**params)
    clf.fit(X_train_sample, y_train_sample)
    return clf

def main():
    try:
        # Load parameters from params.yaml
        import yaml
        with open("params.yaml", 'r') as f:
            params = yaml.safe_load(f)
        n_estimators = params['model_building']['n_estimators']

        # Load and prepare data
        train_path = "/home/hababi/src/data/processed/train_processed.csv"
        test_path = "/home/hababi/src/data/processed/test_processed.csv"
        X_train, X_test, y_train, y_test = load_data(train_path, test_path)
        
        # Get models
        models = create_models(n_estimators)
        
        best_model = None
        best_score = -1
        best_config = None
        
        # Train and evaluate models
        for model_name, model_info in models.items():
            for sampling in ['none', 'smote']:
                print(f"\nTraining {model_name} with {sampling} sampling...")
                
                with mlflow.start_run(run_name=f"{model_name}_{sampling}"):
                    # Train model
                    model = train_model(
                        X_train, y_train,
                        model_info['model'],
                        model_info['params'],
                        sampling
                    )
                    
                    # Evaluate model
                    metrics = evaluate_model(model, X_test, y_test)
                    
                    # Log parameters and metrics
                    mlflow.log_params(model_info['params'])
                    mlflow.log_param('sampling_method', sampling)
                    mlflow.log_metrics(metrics)
                    
                    # Update best model if current model has better f1 score
                    if metrics['f1'] > best_score:
                        best_score = metrics['f1']
                        best_model = model
                        best_config = {
                            'model_name': model_name,
                            'sampling': sampling,
                            'metrics': metrics
                        }
        
        # Save only the best model
        os.makedirs('./models', exist_ok=True)
        with open("./models/model.pkl", "wb") as file:
            pickle.dump(best_model, file)
        
        # Save best model metrics
        print("\nBest Model Configuration:")
        print(f"Model: {best_config['model_name']}")
        print(f"Sampling: {best_config['sampling']}")
        print("Metrics:", best_config['metrics'])
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()



        # 'XGBoost': {
        #     'model': XGBClassifier(),
        #     'params': {
        #         'n_estimators': 100,
        #         'max_depth': 6,
        #         'learning_rate': 0.1,
        #         'random_state': 42
        #     }
        # },
        # 'KNN': {
        #     'model': KNeighborsClassifier(),
        #     'params': {
        #         'n_neighbors': 5,
        #         'weights': 'uniform',
        #         'metric': 'minkowski'
        #     }
        # }
    