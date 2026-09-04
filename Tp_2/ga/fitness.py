"""Función de aptitud.

Error = RMSE por píxel en RGB contra el target. El fitness se normaliza a [0, 1]
y se maximiza, porque ruleta / universal / Boltzmann necesitan valores positivos
proporcionales a la calidad.

    fitness = 1 - RMSE / 255      fitness(target) == 1.0

El resultado se cachea en el individuo: evaluar es el 90% del costo de cómputo.
"""

from __future__ import annotations

import numpy as np

from .individual import Individual
from .render import render_array

MAX_CHANNEL = 255.0


class FitnessEvaluator:
    def __init__(self, target: np.ndarray, background=(255, 255, 255)):
        self.target = target
        self.size = target.shape[0]
        self.background = background
        self.evaluations = 0

    def rmse(self, individual: Individual) -> float:
        rendered = render_array(individual, self.size, self.background)
        return float(np.sqrt(np.mean((rendered - self.target) ** 2)))

    def __call__(self, individual: Individual) -> float:
        if individual.fitness is None:
            individual.fitness = 1.0 - self.rmse(individual) / MAX_CHANNEL
            self.evaluations += 1
        return individual.fitness

    def evaluate_all(self, population: list[Individual]) -> None:
        for ind in population:
            self(ind)
