"""Fenotipo: del vector de genes a un bitmap.

`draw` es el único lugar donde se pinta: recibe triángulos ya en coordenadas de
píxel y no sabe nada de genotipos. Tanto el render de un individuo como la
reconstrucción desde un archivo (ga/artifact.py) pasan por acá, así que ambos
producen exactamente la misma imagen.

Se pinta sobre un canvas opaco usando ImageDraw en modo "RGBA", que hace el
alpha-blending sobre la imagen base (source-over): un solo buffer para los N
triángulos en vez de N capas compuestas.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

from .individual import GENES_PER_TRIANGLE, Individual

WHITE = (255, 255, 255)

#: regla de composición, declarada en el archivo de salida
COMPOSITING = "rgba-source-over"


def draw(triangles: Iterable, width: int, height: int, background=WHITE) -> Image.Image:
    """triangles: iterable de (vértices en píxeles, color RGBA 0-255).

    Se pintan en orden: el primero queda abajo, el último arriba.
    """
    canvas = Image.new("RGB", (width, height), tuple(background))
    painter = ImageDraw.Draw(canvas, "RGBA")
    for vertices, rgba in triangles:
        painter.polygon([tuple(v) for v in vertices], fill=tuple(rgba))
    return canvas


def to_pixels(individual: Individual, width: int, height: int, precision: int = 2):
    """Decodifica el genotipo a triángulos en píxeles, en orden de pintado.

    Se redondea a `precision` decimales: es la precisión con la que se guarda el
    archivo de salida, y renderizar desde estos mismos valores garantiza que el
    PNG y el archivo describan exactamente la misma imagen.
    """
    for block in individual.genes.reshape(-1, GENES_PER_TRIANGLE):
        vertices = [
            [round(float(block[0]) * width, precision), round(float(block[1]) * height, precision)],
            [round(float(block[2]) * width, precision), round(float(block[3]) * height, precision)],
            [round(float(block[4]) * width, precision), round(float(block[5]) * height, precision)],
        ]
        yield vertices, [int(round(float(c) * 255)) for c in block[6:10]]


def render(individual: Individual, size: int, background=WHITE) -> Image.Image:
    return draw(to_pixels(individual, size, size), size, size, background)


def render_array(individual: Individual, size: int, background=WHITE) -> np.ndarray:
    return np.asarray(render(individual, size, background), dtype=np.float64)


def load_target(path: str, size: int, background=WHITE) -> np.ndarray:
    """Carga el target, lo lleva a cuadrado `size` x `size` y aplana el alpha."""
    img = Image.open(path).convert("RGBA").resize((size, size), Image.LANCZOS)
    flat = Image.new("RGB", (size, size), tuple(background))
    flat.paste(img, (0, 0), img)
    return np.asarray(flat, dtype=np.float64)
