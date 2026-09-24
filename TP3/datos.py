from pathlib import Path

import numpy as np
import pandas as pd

DATASET = Path(__file__).parent / 'datasets' / 'fraud_dataset.csv'

# Features con señal según edas/analisis_dataset.md §5. Se descartan timestamp,
# device_screen_resolution y time_since_last_login_s (correlación ~0 con el target)
FEATURES = [
    'amount_usd',
    'quantity_purchased',
    'session_duration_seconds',
    'days_since_last_purchase',
    'account_age_days',
    'items_viewed_before_purchase',
]
# Knowledge Distillation: TinyModel aprende la salida de BigModel
TARGET = 'big_model_fraud_probability'
# Ground truth: NO se usa para entrenar, solo para evaluar y elegir el umbral
LABEL = 'flagged_fraud'


def load(path=DATASET, features=FEATURES):
    df = pd.read_csv(path)
    return df[features].to_numpy(float), df[TARGET].to_numpy(float), df[LABEL].to_numpy(int)


def fit_standardizer(X):
    # En generalización se ajusta solo con el set de entrenamiento
    return X.mean(axis=0), X.std(axis=0)


def standardize(X, mean, std):
    return (X - mean) / std
