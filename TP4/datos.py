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

# Supuesto de negocio: los clientes son argentinos (precio en ARS, adopciones locales).
# Su correlación con el resto es compatible con ruido, pero es la única medida de audiencia local.
CONSERVAR_POR_DOMINIO = {
    "argentina_audience_pct": "única medida de audiencia local; los clientes venden/adoptan en Argentina",
}

# 15 variables de entrenamiento elegidas por `seleccion.py` (derivadas → ruido → bloques de
# correlación → PCA hacia atrás). Las descartadas siguen en el DataFrame para interpretar.
VARIABLES = [
    "average_reach", "argentina_audience_pct", "non_follower_reach_pct", "comments_per_1000",
    "saves_per_1000", "shares_per_1000", "profile_visits_per_1000", "sponsored_posts_pct",
    "topic_football_pct", "topic_makeup_pct", "topic_dogs_pct", "topic_art_pct",
    "content_tutorial_pct", "content_showcase_pct", "average_watch_time_seconds",
]

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
    """Las 15 variables de entrenamiento (ver `VARIABLES` y `seleccion.py`)."""
    return list(VARIABLES)


def tema_dominante(df):
    """Serie con el tema de mayor porcentaje de cada creador."""
    return df[TEMAS].idxmax(axis=1).map(NOMBRE_TEMA)


def estandarizar(X):
    """Z-score por columna (desvío poblacional). Devuelve (Z, media, desvío)."""
    X = np.asarray(X, dtype=float)
    media, desvio = X.mean(axis=0), X.std(axis=0)
    return (X - media) / desvio, media, desvio
