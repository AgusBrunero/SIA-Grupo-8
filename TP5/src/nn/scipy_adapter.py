"""Método de Powell de scipy sobre el vector aplanado de pesos.

Fuente: Análisis de Datos en Python, diap. 9: "from scipy.optimize import minimize …
En particular, usamos el de Powell para el TP5 sobre Deep Learning."
OptimizacionNOLineal, diap. 35–37: Powell usa direcciones conjugadas, "no necesita
derivadas"; converge en n pasos si la función es convexa, y "incluso un método que
converge en n pasos puede ser demasiado costoso para resolver un problema de redes
neuronales". Por eso se devuelven el tiempo y la cantidad de evaluaciones (decisión D6).

Es el único uso de scipy en el TP. No usa el backprop: solo evalúa la pérdida.
"""
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from src.nn.mlp import MLP


@dataclass(frozen=True)
class ResultadoPowell:
    perdida_inicial: float
    perdida_final: float
    evaluaciones: int          # veces que se evaluó la pérdida
    iteraciones: int
    segundos: float
    convergio: bool            # False si cortó por el límite de evaluaciones
    mensaje: str


def minimizar_powell(redes: list[MLP], perdida: Callable[[], float], max_evaluaciones: int,
                     xtol: float = 1e-4, ftol: float = 1e-4) -> ResultadoPowell:
    """Minimiza perdida() sobre todos los pesos de `redes` (p. ej. [encoder, decoder]).

    perdida no recibe argumentos: hace el forward con los pesos que tengan las redes en
    ese momento y devuelve un número. Al terminar, las redes quedan con los pesos óptimos.
    """
    tamanos = [red.n_params() for red in redes]
    cortes = np.cumsum(tamanos)[:-1]

    def cargar(vector: np.ndarray) -> None:
        for red, parte in zip(redes, np.split(vector, cortes)):
            red.cargar_vector(parte)

    def objetivo(vector: np.ndarray) -> float:
        cargar(vector)
        return perdida()

    inicial = np.concatenate([red.vector() for red in redes])
    perdida_inicial = objetivo(inicial)
    inicio = time.perf_counter()
    resultado = minimize(objetivo, inicial, method='Powell',
                         options={'maxfev': max_evaluaciones, 'xtol': xtol, 'ftol': ftol})
    segundos = time.perf_counter() - inicio
    cargar(resultado.x)
    return ResultadoPowell(
        perdida_inicial=perdida_inicial,
        perdida_final=float(resultado.fun),
        evaluaciones=int(resultado.nfev) + 1,
        iteraciones=int(resultado.nit),
        segundos=segundos,
        convergio=bool(resultado.success),
        mensaje=str(resultado.message),
    )
