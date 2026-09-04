"""Fenotipo: del vector de genes a un bitmap.

Se pinta sobre un canvas opaco usando ImageDraw en modo "RGBA", que hace el
alpha-blending directamente sobre la imagen base: un solo buffer para los N
triángulos en vez de N capas compuestas.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .individual import GENES_PER_TRIANGLE, Individual

WHITE = (255, 255, 255)


def render(individual: Individual, size: int, background=WHITE) -> Image.Image:
    canvas = Image.new("RGB", (size, size), tuple(background))
    draw = ImageDraw.Draw(canvas, "RGBA")
    for block in individual.genes.reshape(-1, GENES_PER_TRIANGLE):
        pts = [
            (float(block[0]) * size, float(block[1]) * size),
            (float(block[2]) * size, float(block[3]) * size),
            (float(block[4]) * size, float(block[5]) * size),
        ]
        rgba = tuple(int(round(float(c) * 255)) for c in block[6:10])
        draw.polygon(pts, fill=rgba)
    return canvas


def render_array(individual: Individual, size: int, background=WHITE) -> np.ndarray:
    return np.asarray(render(individual, size, background), dtype=np.float64)


def load_target(path: str, size: int, background=WHITE) -> np.ndarray:
    """Carga el target, lo lleva a cuadrado `size` x `size` y aplana el alpha."""
    img = Image.open(path).convert("RGBA").resize((size, size), Image.LANCZOS)
    flat = Image.new("RGB", (size, size), tuple(background))
    flat.paste(img, (0, 0), img)
    return np.asarray(flat, dtype=np.float64)
