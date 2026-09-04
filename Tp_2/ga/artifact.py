"""Formato de salida: la enumeración de triángulos como archivo autosuficiente.

El enunciado plantea el TP como un compresor de imágenes, así que la lista de
triángulos tiene que alcanzar por sí sola para reconstruir la imagen. Un volcado
de genes no alcanza: sin el tamaño del canvas, el color de fondo, la regla de
composición y el orden de pintado, el archivo sólo se puede interpretar teniendo
a mano el código que lo escribió.

Este módulo define el documento y su decodificador:

    individuo --build--> documento --render--> imagen

`main.py` renderiza el PNG a partir del documento, no del individuo, así que la
imagen entregada y el archivo describen la misma cosa por construcción. El test
de ida y vuelta en test_ga.py lo verifica.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from .individual import Individual
from .render import COMPOSITING, draw, to_pixels

FORMAT_VERSION = 1

PAINT_ORDER = "El array `triangles` está en orden de pintado: primero el fondo, después el índice 0, luego el 1, etc."


def build(individual: Individual, width: int, height: int, background, **metadata) -> dict:
    """Documento autosuficiente: todo lo necesario para reconstruir la imagen."""
    return {
        "format_version": FORMAT_VERSION,
        "canvas": {"width": width, "height": height, "background": list(background)},
        "compositing": COMPOSITING,
        "paint_order": PAINT_ORDER,
        **metadata,
        "triangles": [
            {"vertices": vertices, "color": color}
            for vertices, color in to_pixels(individual, width, height)
        ],
    }


def render(document: dict) -> Image.Image:
    """Documento -> imagen. Es el camino inverso de `build`."""
    version = document.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(f"format_version {version} desconocida (esta versión lee la {FORMAT_VERSION})")
    if document.get("compositing") != COMPOSITING:
        raise ValueError(f"composición '{document.get('compositing')}' no soportada")

    canvas = document["canvas"]
    triangles = ((t["vertices"], t["color"]) for t in document["triangles"])
    return draw(triangles, canvas["width"], canvas["height"], canvas["background"])


def save(document: dict, path: Path) -> None:
    """Cabecera indentada y un triángulo por línea: con 100 triángulos la
    diferencia entre un archivo de 100 líneas y uno de 1400."""
    head = {key: value for key, value in document.items() if key != "triangles"}
    head_json = json.dumps(head, indent=2, ensure_ascii=False).rstrip().removesuffix("}").rstrip()
    triangles = ",\n".join(
        "    " + json.dumps(triangle, ensure_ascii=False) for triangle in document["triangles"]
    )
    Path(path).write_text(f'{head_json},\n  "triangles": [\n{triangles}\n  ]\n}}\n')


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())
