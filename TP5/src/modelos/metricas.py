"""Píxeles incorrectos de una reconstrucción (consigna 1a.2: "error máximo de 1 pixel incorrecto").

Interpretación A (decisión D7): para cada patrón, la cantidad de píxeles cuya salida
umbralizada difiere del original es ≤ 1. Siempre se reportan el máximo por patrón y el total.

El umbral depende de cómo están codificados los datos (doc de cátedra §0.2) y nunca se
mezclan las dos convenciones:
    '01'      datos en {0, 1},  umbral 0,5   (salida logística, Im = (0, 1))
    'bipolar' datos en {−1, 1}, umbral 0     (salida tanh, Im = (−1, 1))
La salida identidad admite cualquiera de las dos.
"""
from dataclasses import dataclass

import numpy as np

UMBRALES = {'01': 0.5, 'bipolar': 0.0}
OBJETIVO_PIXELES = 1

_CODIFICACION_DE_SALIDA = {'logistica': '01', 'tanh': 'bipolar'}


def validar_codificacion(activacion_salida: str, codificacion: str) -> None:
    if codificacion not in UMBRALES:
        raise ValueError(f'codificación desconocida: {codificacion!r}; opciones: {sorted(UMBRALES)}')
    esperada = _CODIFICACION_DE_SALIDA.get(activacion_salida)
    if esperada is not None and esperada != codificacion:
        raise ValueError(f'salida {activacion_salida!r} va con datos {esperada!r}, no {codificacion!r}')


def binarizar(valores: np.ndarray, codificacion: str) -> np.ndarray:
    """True donde el píxel está encendido."""
    return np.asarray(valores) > UMBRALES[codificacion]


def pixeles_incorrectos(salida: np.ndarray, esperado: np.ndarray, codificacion: str) -> np.ndarray:
    """Por patrón (fila): cuántos píxeles de la salida umbralizada difieren del esperado."""
    return (binarizar(salida, codificacion) != binarizar(esperado, codificacion)).sum(axis=1)


@dataclass(frozen=True)
class ResumenPixeles:
    maximo: int                 # peor patrón (interpretación A: objetivo ≤ 1)
    total: int                  # suma sobre todos los patrones (interpretación B)
    patrones_con_error: int     # patrones con más de OBJETIVO_PIXELES píxeles mal
    cumple: bool                # maximo ≤ OBJETIVO_PIXELES


def resumir(incorrectos: np.ndarray) -> ResumenPixeles:
    maximo = int(incorrectos.max())
    return ResumenPixeles(maximo=maximo, total=int(incorrectos.sum()),
                          patrones_con_error=int((incorrectos > OBJETIVO_PIXELES).sum()),
                          cumple=maximo <= OBJETIVO_PIXELES)
