import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score,roc_auc_score,roc_curve
from sklearn.model_selection import GridSearchCV


def load_data(data_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(data_path)
    except Exception as e:
        raise Exception(f"Error loading data from {data_path}: {e}")

def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X = data.drop(columns=['score'], axis=1)
        y = data['score']
        return X, y
    except Exception as e:
        raise Exception(f"Error preparing data: {e}")

def over_sampling(X_train, y_train):
    sm = SMOTE(random_state = 5)
    X_train_ups, y_train_ups = sm.fit_resample(X_train, y_train.ravel())
    

def train_model_clf(X: pd.DataFrame, y: pd.Series, n_estimators: int) -> RandomForestClassifier:
    try:
        clf = RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(X, y)
        return clf
    except Exception as e:
        raise Exception(f"Error training model 1: {e}")

def save_model_clf(model, model_name: str) -> None:
    try:
        with open(model_name, "wb") as file:
            pickle.dump(model, file)
    except Exception as e:
        raise Exception(f"Error saving model to {model_name}: {e}")


def main():
    try:
        params_path = "params.yaml"
        data_path = "./src/data/processed/train_processed.csv"
        model_name = "./src/models/model_clf.pkl"

        n_estimators = load_params(params_path)
        train_data = load_data(data_path)
        X_train, y_train = prepare_data(train_data)

        model = train_model_1(X_train, y_train, n_estimators)
        save_model_1(model, model_name)
        print("Model trained and saved successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()