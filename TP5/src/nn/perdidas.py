"""Funciones de pérdida. Cada una devuelve (valor, ∂E/∂O).

O y ζ tienen forma (p, n): p patrones (filas) y n neuronas de salida (columnas).
Siempre se suma sobre las n salidas de cada patrón. Sobre los p patrones:
- reduccion='suma'  : se suma, como el E de la Clase 11 (diap. 37 y 54). Decisión D4, AE y DAE.
- reduccion='media' : se divide por p, como el Colab del VAE de la cátedra
                      (Autoencoders diap. 90). Decisión D4, VAE.

error_cuadratico:  E = ½ Σ_μ Σ_i (ζ_i^μ − O_i^μ)²        Clase 11, diap. 37, 51, 55
                   ∂E/∂O = −(ζ − O)

bce:               E = −Σ_μ Σ_i [ζ ln O + (1 − ζ) ln(1 − O)]   Autoencoders diap. 51
                   ∂E/∂O = (O − ζ) / (O (1 − O))
                   Sumada sobre los píxeles como en el Colab de la diap. 90
                   (original_dim * binary_crossentropy), no con el 1/N de la diap. 51.
                   Requiere O en (0, 1): salida logística.
"""
from typing import Callable

import numpy as np

REDUCCIONES = ('suma', 'media')

# Resguardo para el logaritmo si O llega exactamente a 0 o 1. Con la logística de
# activaciones.py no se alcanza (queda a más de 9e−14 de los bordes).
_EPS_LOG = 1e-15


def _escala(O: np.ndarray, reduccion: str) -> float:
    if reduccion not in REDUCCIONES:
        raise ValueError(f'reducción desconocida: {reduccion!r}; opciones: {REDUCCIONES}')
    return 1.0 if reduccion == 'suma' else 1.0 / O.shape[0]


def error_cuadratico(O: np.ndarray, zeta: np.ndarray, reduccion: str = 'suma') -> tuple[float, np.ndarray]:
    escala = _escala(O, reduccion)
    diferencia = zeta - O
    valor = 0.5 * float(np.sum(diferencia ** 2)) * escala
    return valor, -diferencia * escala


def bce(O: np.ndarray, zeta: np.ndarray, reduccion: str = 'suma') -> tuple[float, np.ndarray]:
    escala = _escala(O, reduccion)
    O = np.clip(O, _EPS_LOG, 1.0 - _EPS_LOG)
    valor = -float(np.sum(zeta * np.log(O) + (1.0 - zeta) * np.log(1.0 - O))) * escala
    return valor, (O - zeta) / (O * (1.0 - O)) * escala


Perdida = Callable[[np.ndarray, np.ndarray, str], tuple[float, np.ndarray]]

PERDIDAS: dict[str, Perdida] = {'error_cuadratico': error_cuadratico, 'bce': bce}
