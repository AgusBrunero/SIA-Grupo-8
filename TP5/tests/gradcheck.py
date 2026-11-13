"""Gradient check por diferencias finitas centradas (CLAUDE.md §4).

Paso 1e-5 y error relativo < 1e-6 en float64. Lo usan todos los tests del TP.
"""
from typing import Callable

import numpy as np

PASO = 1e-5
TOLERANCIA = 1e-6


def gradiente_numerico(f: Callable[[], float], x: np.ndarray, paso: float = PASO) -> np.ndarray:
    """∂f/∂x por diferencias centradas. f no recibe argumentos: lee x, que se modifica in-place."""
    gradiente = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=['multi_index'])
    for _ in it:
        i = it.multi_index
        original = x[i]
        x[i] = original + paso
        f_mas = f()
        x[i] = original - paso
        f_menos = f()
        x[i] = original
        gradiente[i] = (f_mas - f_menos) / (2.0 * paso)
    return gradiente


def error_relativo(analitico: np.ndarray, numerico: np.ndarray) -> float:
    """‖a − n‖ / (‖a‖ + ‖n‖); 0 si los dos son nulos."""
    denominador = np.linalg.norm(analitico) + np.linalg.norm(numerico)
    if denominador == 0.0:
        return 0.0
    return float(np.linalg.norm(analitico - numerico) / denominador)
