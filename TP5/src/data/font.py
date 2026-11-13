"""Dataset del ejercicio 1: los 32 caracteres de font.h (consigna 1a).

font.h define `Font3[32][7]`: 32 patrones (0x60 '`' … 0x7f DEL) de 7 filas × 5 columnas.
Cada fila es un entero de 5 bits y el bit 4 (0x10) es la columna izquierda:
    píxel(fila, columna) = (Font3[k][fila] >> (4 − columna)) & 1
Los números se leen del archivo; no hay ninguno copiado acá (CLAUDE.md §4).
"""
import re
from pathlib import Path

import numpy as np

RUTA_FONT = Path(__file__).resolve().parents[2] / 'docs' / 'tp5' / 'font.h'
FILAS, COLUMNAS = 7, 5
N_PATRONES = 32

# Una línea de datos: {0x04, 0x04, …},   // 0x60, `
_LINEA = re.compile(r'^\s*\{([^}]*)\}\s*,?\s*//\s*0x([0-9a-fA-F]{2})')


def cargar_font(ruta: Path = RUTA_FONT) -> tuple[np.ndarray, list[str]]:
    """Devuelve (X, etiquetas): X de forma (32, 35) con píxeles en {0, 1} (float64),
    recorridos por filas, y la etiqueta de cada patrón ('a', '|', 'DEL', …)."""
    patrones, etiquetas = [], []
    for linea in Path(ruta).read_text(encoding='utf-8').splitlines():
        coincidencia = _LINEA.match(linea)
        if coincidencia is None:
            continue
        filas = [int(valor, 16) for valor in coincidencia.group(1).split(',')]
        if len(filas) != FILAS or any(not 0 <= valor < 2 ** COLUMNAS for valor in filas):
            raise ValueError(f'fila mal formada en {ruta}: {linea!r}')
        patrones.append([(valor >> (COLUMNAS - 1 - columna)) & 1
                         for valor in filas for columna in range(COLUMNAS)])
        codigo = int(coincidencia.group(2), 16)
        etiquetas.append('DEL' if codigo == 0x7F else chr(codigo))
    if len(patrones) != N_PATRONES:
        raise ValueError(f'se esperaban {N_PATRONES} patrones en {ruta} y hay {len(patrones)}')
    return np.array(patrones, dtype=np.float64), etiquetas


def a_bipolar(X: np.ndarray) -> np.ndarray:
    """{0, 1} → {−1, 1}, para redes con salida tanh (Clase 10.2: Im = (−1, 1))."""
    return 2.0 * X - 1.0


def a_ascii(patron: np.ndarray) -> str:
    """Un patrón de 35 píxeles en {0, 1} como 7 líneas de '#' y '.'."""
    grilla = np.asarray(patron).reshape(FILAS, COLUMNAS)
    return '\n'.join(''.join('#' if pixel > 0.5 else '.' for pixel in fila) for fila in grilla)


def distancias_hamming(X: np.ndarray) -> np.ndarray:
    """Matriz (n, n) con la cantidad de píxeles distintos entre cada par de patrones en {0, 1}."""
    binario = np.asarray(X) > 0.5
    return (binario[:, None, :] != binario[None, :, :]).sum(axis=2)
