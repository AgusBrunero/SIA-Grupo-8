"""Representación del individuo.

Un individuo es una imagen candidata: una lista ORDENADA de triángulos (el orden
importa, define el z-order al pintar). Se guarda como un vector plano de floats en
[0, 1] para que cruza y mutación operen sobre un np.ndarray sin conocer la semántica.

Layout por triángulo (GENES_PER_TRIANGLE = 10):
    [x1, y1, x2, y2, x3, y3, r, g, b, a]
Coordenadas normalizadas al canvas; color y alpha normalizados a [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GENES_PER_TRIANGLE = 10


@dataclass
class Individual:
    genes: np.ndarray
    fitness: float | None = None

    @property
    def n_triangles(self) -> int:
        return len(self.genes) // GENES_PER_TRIANGLE

    def copy(self) -> "Individual":
        return Individual(self.genes.copy(), self.fitness)


def random_individual(n_triangles: int, rng: np.random.Generator) -> Individual:
    """Inicialización naive: todos los genes uniformes en [0, 1].

    Es el baseline. Alternativas a comparar más adelante: vértices agrupados
    alrededor de un centro (triángulos chicos) o color muestreado del target.
    """
    return Individual(rng.random(n_triangles * GENES_PER_TRIANGLE, dtype=np.float64))
