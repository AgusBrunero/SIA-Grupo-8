"""Dataset de creadores (Ej. 1): carga, selección de variables y estandarización."""
from pathlib import Path

import numpy as np
import pandas as pd

RUTA_CREADORES = Path(__file__).parent / "datos" / "creadores.csv"

# Columnas que son función exacta (o casi) de otras: no agregan información y duplican peso.
DERIVADAS = {
    "price_ars": "price_usd × 1500",
    "organic_posts_pct": "100 − sponsored_posts_pct",
    "total_engagement_per_1000": "likes + comments + saves + shares (por 1000)",
    "estimated_argentina_reach": "average_reach × argentina_audience_pct / 100",
}

TEMAS = ["topic_football_pct", "topic_makeup_pct", "topic_dogs_pct", "topic_art_pct"]
NOMBRE_TEMA = {
    "topic_football_pct": "fútbol",
    "topic_makeup_pct": "maquillaje",
    "topic_dogs_pct": "perros",
    "topic_art_pct": "arte",
}


def cargar_creadores():
    """DataFrame completo, indexado por `name`."""
    return pd.read_csv(RUTA_CREADORES).set_index("name")


def variables_seleccionadas(df):
    """Columnas numéricas sin las derivadas (ver `DERIVADAS`)."""
    numericas = df.select_dtypes("number").columns
    return [c for c in numericas if c not in DERIVADAS]


def tema_dominante(df):
    """Serie con el tema de mayor porcentaje de cada creador."""
    return df[TEMAS].idxmax(axis=1).map(NOMBRE_TEMA)


def estandarizar(X):
    """Z-score por columna (desvío poblacional). Devuelve (Z, media, desvío)."""
    X = np.asarray(X, dtype=float)
    media, desvio = X.mean(axis=0), X.std(axis=0)
    return (X - media) / desvio, media, desvio
