"""Generación desde el espacio latente (consigna 1a.4).

Autoencoders diap. 34 (Generative Autoencoder): (1) codificar los patrones en el latente,
(2) dejar de lado el encoder, (3) especificar directamente valores de z1, z2, (4) cada
tupla z da una muestra nueva con el decoder. Acá están los pasos (3) y la comparación de
cada muestra con el dataset; el paso (4) es Autoencoder.decodificar.
Diap. 33 (Concept Vector): moverse en el latente de un patrón a otro.
"""
import numpy as np


def grilla_latente(Z: np.ndarray, n: int) -> np.ndarray:
    """n × n puntos regulares sobre el rectángulo que cubre los datos en el latente 2D.

    Devuelve (n·n, 2), recorrido por filas con z2 de mayor a menor (arriba es z2 máximo),
    para dibujarlo directamente como una grilla de imágenes.
    """
    z1 = np.linspace(Z[:, 0].min(), Z[:, 0].max(), n)
    z2 = np.linspace(Z[:, 1].max(), Z[:, 1].min(), n)
    return np.array([[a, b] for b in z2 for a in z1])


def interpolar(z_desde: np.ndarray, z_hasta: np.ndarray, pasos: int) -> np.ndarray:
    """`pasos` puntos sobre el segmento que une dos códigos, extremos incluidos."""
    t = np.linspace(0.0, 1.0, pasos)[:, None]
    return z_desde + t * (z_hasta - z_desde)


def vecino_mas_cercano(muestras: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Para cada muestra binaria (bool), el patrón del dataset más cercano por distancia de
    Hamming: devuelve (índice del patrón, distancia en píxeles). 0 = es una letra del dataset."""
    distancias = (muestras[:, None, :] != (X > 0.5)[None, :, :]).sum(axis=2)
    return distancias.argmin(axis=1), distancias.min(axis=1)


def vecino_por_pixel(muestras: np.ndarray, X: np.ndarray, excluir_identicas: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Para cada muestra en grises, la imagen del dataset más cercana (consigna 2c).

    Distancia (nuestra): media sobre los píxeles de |muestra − imagen|, en unidades de gris
    (0 = idénticas, 1 = opuestas). Devuelve (índice de la imagen, distancia).
    Con excluir_identicas=True se saltea la imagen de X en la misma posición que la muestra
    (sirve para medir, dentro del dataset, a qué distancia está la otra imagen más cercana).
    """
    indices = np.empty(len(muestras), dtype=int)
    distancias = np.empty(len(muestras))
    for i, muestra in enumerate(muestras):
        d = np.abs(X - muestra).mean(axis=1)
        if excluir_identicas:
            d[i] = np.inf
        indices[i], distancias[i] = d.argmin(), d.min()
    return indices, distancias
