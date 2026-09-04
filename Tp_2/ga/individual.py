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


def random_individual(n_triangles: int, rng: np.random.Generator, target=None) -> Individual:
    """Inicialización naive: todos los genes uniformes en [0, 1]. Es el baseline."""
    return Individual(rng.random(n_triangles * GENES_PER_TRIANGLE, dtype=np.float64))


def grid_individual(n_triangles: int, rng: np.random.Generator, target=None) -> Individual:
    """Inicialización informada: un triángulo por celda de una grilla, con el color
    promedio que tiene el target en esa celda.

    Arrancar de ruido desperdicia las primeras generaciones descubriendo la paleta
    y el reparto grueso de la imagen, que se pueden leer directo del target. Los
    vértices siguen siendo aleatorios dentro de la celda: si no, todos los
    individuos arrancarían idénticos y la población no tendría diversidad para
    evolucionar.
    """
    if target is None:
        raise ValueError("la inicialización 'grid' necesita el target para muestrear colores")

    side = int(np.ceil(np.sqrt(n_triangles)))
    height, width = target.shape[:2]
    cell = 1.0 / side
    genes = np.empty((n_triangles, GENES_PER_TRIANGLE))

    for k in range(n_triangles):
        row, col = divmod(k, side)
        x0, y0 = col * cell, row * cell

        # El triángulo cubre media celda: se eligen 3 de las 4 esquinas (una de las
        # dos diagonales, al azar) y se las sacude. Tres puntos uniformes dentro de
        # la celda darían triángulos diminutos que casi no pintan nada.
        corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        picked = corners[[0, 1, 2]] if rng.random() < 0.5 else corners[[0, 2, 3]]
        jitter = rng.uniform(-0.25, 0.25, size=picked.shape)
        vertices = np.clip((picked + jitter) * cell + [x0, y0], 0.0, 1.0)
        genes[k, [0, 2, 4]] = vertices[:, 0]
        genes[k, [1, 3, 5]] = vertices[:, 1]

        patch = target[int(y0 * height):max(int((y0 + cell) * height), int(y0 * height) + 1),
                       int(x0 * width):max(int((x0 + cell) * width), int(x0 * width) + 1)]
        genes[k, 6:9] = patch.reshape(-1, 3).mean(axis=0) / 255.0
        genes[k, 9] = rng.uniform(0.7, 1.0)

    return Individual(genes.ravel())


INITIALIZERS = {
    "random": random_individual,
    "grid": grid_individual,
}


def get_initializer(name: str):
    if name not in INITIALIZERS:
        raise ValueError(f"inicialización '{name}' no implementada. Disponibles: {sorted(INITIALIZERS)}")
    return INITIALIZERS[name]
