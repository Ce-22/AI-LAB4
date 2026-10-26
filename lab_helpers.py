"""
Módulo auxiliar de funções para o Laboratório 4 de IA/ML (Modelos Ensemble: Bagging, Boosting e Stacking).
Autores: Cesar Gritti e Pedro Isamu Yoshissaro (PUC-SP)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vega_datasets import data

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


def load_and_preprocess_seattle_weather(csv_path: str = None):
    """
    Carrega o dataset seattle_weather (localmente ou via vega_datasets),
    realiza engenharia de atributos temporais e meteorológicos, e divide em treino e teste estratificados.
    """
    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        df = data.seattle_weather()

    df['date'] = pd.to_datetime(df['date'])
    df['temp_range'] = df['temp_max'] - df['temp_min']
    df['temp_avg'] = (df['temp_max'] + df['temp_min']) / 2.0
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['precip_3d_rolling'] = df['precipitation'].rolling(window=3, min_periods=1).mean()
    df['wind_temp_ratio'] = df['wind'] / (df['temp_max'] + 10.0)

    feature_cols = [
        'precipitation', 'temp_max', 'temp_min', 'wind',
        'temp_range', 'temp_avg', 'month', 'day', 'dayofweek',
        'is_weekend', 'precip_3d_rolling', 'wind_temp_ratio'
    ]

    X = df[feature_cols].copy()
    le = LabelEncoder()
    y = le.fit_transform(df['weather'])
    target_names = list(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, feature_cols, target_names


def build_stacking_ensemble():
    """
    Cria a arquitetura de StackingClassifier heterogêneo combinando:
    - Random Forest
    - Support Vector Machine (RBF)
    - Multilayer Perceptron (MLP)
    - HistGradientBoosting
    Com meta-aprendiz de Regressão Logística.
    """
    base_learners = [
        ('rf', make_pipeline(SimpleImputer(strategy='median'),
                             RandomForestClassifier(n_estimators=100, random_state=42))),
        ('svm', make_pipeline(SimpleImputer(strategy='median'),
                              StandardScaler(),
                              SVC(probability=True, kernel='rbf', C=1.0, random_state=42))),
        ('mlp', make_pipeline(SimpleImputer(strategy='median'),
                              StandardScaler(),
                              MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800, random_state=42))),
        ('hgb', make_pipeline(SimpleImputer(strategy='median'),
                              HistGradientBoostingClassifier(max_iter=100, random_state=42)))
    ]

    return StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5
    )


def evaluate_models_cv(models_dict, X_train, y_train, X_test, y_test, cv_splits=5):
    """
    Avalia um dicionário de modelos via validação cruzada estratificada e teste holdout.
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    results = []

    for name, model in models_dict.items():
        cv_scores = cross_validate(model, X_train, y_train, cv=cv, scoring=['accuracy', 'f1_macro'])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append({
            'Model': name,
            'CV Accuracy': cv_scores['test_accuracy'].mean(),
            'CV F1-Macro': cv_scores['test_f1_macro'].mean(),
            'Test Accuracy': accuracy_score(y_test, y_pred),
            'Test F1-Macro': f1_score(y_test, y_pred, average='macro'),
            'Test F1-Weighted': f1_score(y_test, y_pred, average='weighted')
        })

    return pd.DataFrame(results)
