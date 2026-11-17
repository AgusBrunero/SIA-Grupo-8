"""Denoising Autoencoder (consigna 1b): entrenamiento con ruido y evaluación.

Autoencoders diap. 22: la entrada es X̃ (X con ruido) y "la salida que se pone en la red
corresponde al dato X sin alterar". Una vez aprendido, la muestra ruidosa se reemplaza
"por la muestra obtenida en la salida del Decodificador".
El modelo es el mismo Autoencoder de 1a; cambia con qué entrada se entrena.
"""
from collections.abc import Callable

import numpy as np
import pandas as pd

from src.data.ruido import RUIDOS
from src.modelos.autoencoder import Autoencoder
from src.modelos.generacion import vecino_mas_cercano
from src.modelos.metricas import binarizar


def corruptor(p: float, rng: np.random.Generator,
              tipo: str = 'sal_y_pimienta') -> Callable[[np.ndarray], np.ndarray]:
    """Función para `entrenar(corromper=…)`: ruido `tipo` de nivel p (σ si es gaussiano),
    distinto en cada llamada."""
    ruido = RUIDOS[tipo]
    return lambda X: ruido(X, p, rng)


def evaluar_con_ruido(ae: Autoencoder, X: np.ndarray, p: float, realizaciones: int,
                      rng: np.random.Generator, tipo: str = 'sal_y_pimienta') -> pd.DataFrame:
    """Reconstruye `realizaciones` versiones ruidosas nuevas de cada patrón.

    Una fila por (patrón, realización):
      alterados   píxeles de la entrada ruidosa, umbralizada en 0,5, distintos del patrón
                  limpio (con salt-and-pepper, los que el ruido invirtió)
      mal         píxeles de la salida umbralizada que difieren del patrón limpio
      predicho    patrón del dataset más cercano (Hamming) a la salida umbralizada
    """
    filas = []
    for realizacion in range(realizaciones):
        ruidosa = RUIDOS[tipo](X, p, rng)
        salida = binarizar(ae.reconstruir(ruidosa), '01')
        predicho, _ = vecino_mas_cercano(salida, X)
        filas.append(pd.DataFrame({
            'patron': np.arange(len(X)), 'realizacion': realizacion,
            'alterados': (binarizar(ruidosa, '01') != (X > 0.5)).sum(axis=1), 'mal': (salida != (X > 0.5)).sum(axis=1),
            'predicho': predicho}))
    return pd.concat(filas, ignore_index=True)
