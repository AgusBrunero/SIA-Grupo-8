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


def render(individual: Individual, width: int, height: int | None = None,
           *, background=WHITE) -> Image.Image:
    """`height` omitido significa canvas cuadrado, que es el caso por defecto.

    `background` es keyword-only a propósito: cuando el alto era implícito, el fondo
    iba tercero, y al agregarlo se cuela silenciosamente como alto.
    """
    height = width if height is None else height
    if not isinstance(height, int):
        raise TypeError(f"height debe ser un entero, no {type(height).__name__} "
                        f"({height!r}). ¿Estás pasando el fondo como tercer posicional?")
    return draw(to_pixels(individual, width, height), width, height, background)


def render_array(individual: Individual, width: int, height: int | None = None,
                 *, background=WHITE) -> np.ndarray:
    return np.asarray(render(individual, width, height, background=background),
                      dtype=np.float64)


def canvas_size(path: str, size: int, preserve_aspect: bool) -> tuple[int, int]:
    """Tamaño de trabajo (ancho, alto). Con `preserve_aspect`, `size` es el lado LARGO."""
    if not preserve_aspect:
        return size, size
    with Image.open(path) as img:
        width, height = img.size
    if width >= height:
        return size, max(1, round(size * height / width))
    return max(1, round(size * width / height)), size


def load_target(path: str, size: int, background=WHITE, preserve_aspect: bool = False) -> np.ndarray:
    """Carga el target, lo lleva al tamaño de trabajo y aplana el alpha.

    Por defecto fuerza el cuadrado `size` x `size`. Con `preserve_aspect=True` respeta
    la proporción original y `size` pasa a ser el lado largo: una imagen apaisada
    aplastada a cuadrado se compara contra un target deformado, y el resultado no se
    puede poner al lado del original.
    """
    width, height = canvas_size(path, size, preserve_aspect)
    img = Image.open(path).convert("RGBA").resize((width, height), Image.LANCZOS)
    flat = Image.new("RGB", (width, height), tuple(background))
    flat.paste(img, (0, 0), img)
    return np.asarray(flat, dtype=np.float64)
