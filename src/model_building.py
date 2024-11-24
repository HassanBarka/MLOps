import os
import mlflow
import dagshub
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Initialize DagsHub and MLflow integration
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
                'max_depth': 5,
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
        # Load and prepare data
        train_path = "/home/hababi/data/processed/train_processed.csv"
        test_path = "/home/hababi/data/processed/test_processed.csv"
        X_train, X_test, y_train, y_test = load_data(train_path, test_path)
        
        # Create models directory if it doesn't exist
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        models_dir = os.path.join(project_root, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        # Create models
        models = create_models(n_estimators=5)
        
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
                    mlflow.log_param("input_rows", X_train.shape[0])
                    mlflow.log_param("input_cols", X_train.shape[1])
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
        if best_model is not None:
            model_path = os.path.join(models_dir, "model.pkl")
            with open(model_path, "wb") as file:
                pickle.dump(best_model, file)
            print(f"Best model saved to {model_path}")
        else:
            raise ValueError("No valid model found to save.")
        
        # Save best model metrics
        metrics_path = os.path.join(models_dir, "best_model_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(best_config['metrics'], f, indent=4)
            print(f"Metrics saved to {metrics_path}")
        
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
    