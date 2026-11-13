"""Funciones de activación θ(h) y su derivada θ'(h).

Fuente: Clase 10.2 (Perceptrón Lineal y No Lineal).
- identidad:  θ(h) = h                          (diap. 8)
- tanh:       θ(h) = tanh(βh),   θ'(h) = β(1 − θ²(h)),      Im = (−1, 1)   (diap. 23, 25–26)
- logistica:  θ(h) = 1/(1 + e^(−2βh)),  θ'(h) = 2βθ(h)(1 − θ(h)),  Im = (0, 1)   (diap. 23, 25–26)

Las dos funciones reciben la excitación h (no θ(h)) y trabajan elemento a elemento.
"""
from dataclasses import dataclass

import numpy as np

NOMBRES = ('identidad', 'tanh', 'logistica')

# Cota de |2βh| en la logística. Evita el overflow de exp y garantiza que θ quede
# estrictamente dentro de (0, 1) en float64 (1 − θ ≥ ~9e−14), así log(θ) y log(1 − θ)
# de la BCE nunca reciben un cero. Es un resguardo numérico, no cambia la función
# en el rango donde se entrena.
_COTA_EXPONENTE = 30.0


@dataclass(frozen=True)
class Activacion:
    nombre: str
    beta: float = 1.0

    def __post_init__(self) -> None:
        if self.nombre not in NOMBRES:
            raise ValueError(f'activación desconocida: {self.nombre!r}; opciones: {NOMBRES}')

    def __call__(self, h: np.ndarray) -> np.ndarray:
        """θ(h)."""
        if self.nombre == 'identidad':
            return h
        if self.nombre == 'tanh':
            return np.tanh(self.beta * h)
        exponente = np.clip(2.0 * self.beta * h, -_COTA_EXPONENTE, _COTA_EXPONENTE)
        return 1.0 / (1.0 + np.exp(-exponente))

    def derivada(self, h: np.ndarray) -> np.ndarray:
        """θ'(h)."""
        if self.nombre == 'identidad':
            return np.ones_like(h)
        theta = self(h)
        if self.nombre == 'tanh':
            return self.beta * (1.0 - theta ** 2)
        return 2.0 * self.beta * theta * (1.0 - theta)
