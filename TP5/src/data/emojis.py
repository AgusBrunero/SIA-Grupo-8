"""Dataset del ejercicio 2 (consigna 2a: "elegir o construir un conjunto de datos nuevos,
por ejemplo emojis"): caritas construidas en código.

Cada imagen es una carita de 16 × 16 en grises en [0, 1] (1 = tinta), definida por siete
parámetros continuos sorteados con semilla. No se descarga nada: el dataset es propio.
Los grises salen de dibujar cada carita en una grilla 4 veces más fina y promediar cada
bloque de 4 × 4 (suavizado de bordes).

Los datos en [0, 1] con BCE son los del Colab del VAE de la cátedra (Autoencoders diap. 90).
"""
from dataclasses import dataclass

import numpy as np

LADO = 16
SOBREMUESTREO = 4
N_PIXELES = LADO * LADO

# Rango de cada parámetro (uniforme). Unidades: píxeles de la imagen de 16 × 16, salvo curvatura.
RANGOS: dict[str, tuple[float, float]] = {
    'centro_x': (-0.6, 0.6),          # corrimiento horizontal de toda la cara
    'centro_y': (-0.6, 0.6),          # corrimiento vertical
    'curvatura_boca': (-1.0, 1.0),    # +1 sonrisa, 0 boca recta, −1 boca triste
    'ancho_boca': (2.4, 4.0),         # semiancho de la boca
    'apertura_ojos': (0.35, 1.3),     # semialto de los ojos (chico = entrecerrados)
    'separacion_ojos': (2.2, 3.2),    # distancia horizontal de cada ojo al centro
    'inclinacion_cejas': (-0.8, 0.8), # pendiente de las cejas (+ = enojado, − = preocupado)
}
PARAMETROS = tuple(RANGOS)


@dataclass(frozen=True)
class Caritas:
    X: np.ndarray             # (n, 256), grises en [0, 1], recorridos por filas
    parametros: np.ndarray    # (n, 7), en el orden de PARAMETROS


def dibujar(parametros: np.ndarray) -> np.ndarray:
    """Una carita (16, 16) en [0, 1] a partir de sus siete parámetros."""
    cx, cy, curvatura, ancho, apertura, separacion, inclinacion = parametros
    lado = LADO * SOBREMUESTREO
    # Coordenadas del centro de cada subpíxel, en píxeles de la imagen final, con origen en el centro
    eje = (np.arange(lado) + 0.5) / SOBREMUESTREO - LADO / 2
    x, y = np.meshgrid(eje - cx, eje - cy)          # y crece hacia abajo
    radio = np.hypot(x, y)
    tinta = (radio >= 6.2) & (radio <= 7.2)         # contorno de la cara
    # Ojos: dos elipses
    for lado_ojo in (-1.0, 1.0):
        tinta |= ((x - lado_ojo * separacion) / 0.9) ** 2 + ((y + 1.6) / apertura) ** 2 <= 1.0
        # Ceja: segmento inclinado sobre el ojo, simétrico respecto del eje vertical
        dx = x - lado_ojo * separacion
        tinta |= (np.abs(dx) <= 1.3) & (np.abs(y + 3.9 + lado_ojo * inclinacion * dx * 0.6) <= 0.35)
    # Boca: parábola; con curvatura positiva las puntas quedan más arriba que el centro
    boca_y = 3.2 + curvatura * 1.4 * (1.0 - 2.0 * (x / ancho) ** 2)
    tinta |= (np.abs(x) <= ancho) & (np.abs(y - boca_y) <= 0.45)
    return tinta.reshape(LADO, SOBREMUESTREO, LADO, SOBREMUESTREO).mean(axis=(1, 3))


def generar_caritas(n: int, semilla: int) -> Caritas:
    """n caritas con parámetros uniformes en RANGOS, reproducibles con la semilla."""
    rng = np.random.default_rng(semilla)
    minimos = np.array([RANGOS[nombre][0] for nombre in PARAMETROS])
    maximos = np.array([RANGOS[nombre][1] for nombre in PARAMETROS])
    parametros = rng.uniform(minimos, maximos, size=(n, len(PARAMETROS)))
    X = np.stack([dibujar(fila).ravel() for fila in parametros])
    return Caritas(X=X, parametros=parametros)


def a_ascii(imagen: np.ndarray) -> str:
    """Una imagen de 256 píxeles en [0, 1] como 16 líneas: '#' ≥ 0,66, '+' ≥ 0,33, '.' el resto."""
    grilla = np.asarray(imagen).reshape(LADO, LADO)
    return '\n'.join(''.join('#' if v >= 0.66 else '+' if v >= 0.33 else '.' for v in fila) for fila in grilla)
