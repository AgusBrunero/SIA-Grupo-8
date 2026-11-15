"""Dataset del ejercicio 2 (consigna 2a: "elegir o construir un conjunto de datos nuevos"):
relojes de agujas construidos en código.

Cada imagen es un reloj de 16 × 16 en grises en [0, 1] (1 = tinta): esfera circular con cuatro
marcas (12, 3, 6 y 9), aguja de las horas (corta y gruesa) y aguja de los minutos (larga y fina).
La esfera es siempre la misma; lo único que cambia entre relojes es la hora, un parámetro continuo
sorteado con semilla que fija los ángulos de las dos agujas como en un reloj real (la de las horas
avanza también con los minutos). La hora es cíclica: las 12:00 y las 11:59 son casi la misma imagen.
Probamos además variar el radio y el centro de la esfera: el VAE gastaba el latente en esos
corrimientos (mueven muchos píxeles) y dejaba las agujas borrosas, así que quedaron fijos.
No se descarga nada: el dataset es propio. Los grises salen de dibujar cada reloj en una grilla
4 veces más fina y promediar cada bloque de 4 × 4 (suavizado de bordes).

Los datos en [0, 1] con BCE son los del Colab del VAE de la cátedra (Autoencoders diap. 90).
"""
from dataclasses import dataclass

import numpy as np

LADO = 16
SOBREMUESTREO = 4
N_PIXELES = LADO * LADO

# Rango de cada parámetro (uniforme), en horas: 0 = las 12:00; 3,5 = las 3:30
RANGOS: dict[str, tuple[float, float]] = {'hora': (0.0, 12.0)}
PARAMETROS = tuple(RANGOS)

# Medidas fijas, en píxeles de la imagen de 16 × 16
RADIO = 7.0
GROSOR_ESFERA = 0.8
LARGO_MARCA = 1.2


@dataclass(frozen=True)
class Relojes:
    X: np.ndarray             # (n, 256), grises en [0, 1], recorridos por filas
    parametros: np.ndarray    # (n, 1), en el orden de PARAMETROS


def angulos(hora: float) -> tuple[float, float]:
    """(ángulo de la aguja de las horas, ángulo de la de los minutos), en radianes desde las 12
    y en sentido horario. La de las horas avanza también con los minutos, como en un reloj real."""
    return hora / 12.0 * 2.0 * np.pi, (hora % 1.0) * 2.0 * np.pi


def _segmento(x: np.ndarray, y: np.ndarray, angulo: float, desde: float, hasta: float, semiancho: float) -> np.ndarray:
    """Subpíxeles de un segmento que sale del centro en la dirección del ángulo (y crece hacia abajo)."""
    ux, uy = np.sin(angulo), -np.cos(angulo)
    a_lo_largo = x * ux + y * uy
    al_costado = np.abs(y * ux - x * uy)
    return (a_lo_largo >= desde) & (a_lo_largo <= hasta) & (al_costado <= semiancho)


def dibujar(parametros: np.ndarray) -> np.ndarray:
    """Un reloj (16, 16) en [0, 1] a partir de sus parámetros (solo la hora)."""
    (hora,) = parametros
    lado = LADO * SOBREMUESTREO
    # Coordenadas del centro de cada subpíxel, en píxeles de la imagen final, con origen en el centro
    eje = (np.arange(lado) + 0.5) / SOBREMUESTREO - LADO / 2
    x, y = np.meshgrid(eje, eje)
    radio_interior = RADIO - GROSOR_ESFERA
    tinta = (np.hypot(x, y) >= radio_interior) & (np.hypot(x, y) <= RADIO)          # esfera
    for k in range(4):                                                              # marcas de 12, 3, 6 y 9
        tinta |= _segmento(x, y, k * np.pi / 2, radio_interior - LARGO_MARCA, radio_interior, 0.4)
    angulo_horas, angulo_minutos = angulos(hora)
    tinta |= _segmento(x, y, angulo_horas, -0.3, 0.5 * RADIO, 0.6)                  # horas: corta y gruesa
    tinta |= _segmento(x, y, angulo_minutos, -0.3, 0.85 * RADIO - GROSOR_ESFERA, 0.4)  # minutos: larga y fina
    return tinta.reshape(LADO, SOBREMUESTREO, LADO, SOBREMUESTREO).mean(axis=(1, 3))


def generar_relojes(n: int, semilla: int) -> Relojes:
    """n relojes con parámetros uniformes en RANGOS, reproducibles con la semilla."""
    rng = np.random.default_rng(semilla)
    minimos = np.array([RANGOS[nombre][0] for nombre in PARAMETROS])
    maximos = np.array([RANGOS[nombre][1] for nombre in PARAMETROS])
    parametros = rng.uniform(minimos, maximos, size=(n, len(PARAMETROS)))
    X = np.stack([dibujar(fila).ravel() for fila in parametros])
    return Relojes(X=X, parametros=parametros)


def a_texto(hora: float) -> str:
    """La hora como la leería una persona: 0,5 → '12:30'."""
    minutos_totales = int(round(hora * 60)) % 720
    h, m = divmod(minutos_totales, 60)
    return f'{h if h else 12}:{m:02d}'


def a_ascii(imagen: np.ndarray) -> str:
    """Una imagen de 256 píxeles en [0, 1] como 16 líneas: '#' ≥ 0,66, '+' ≥ 0,33, '.' el resto."""
    grilla = np.asarray(imagen).reshape(LADO, LADO)
    return '\n'.join(''.join('#' if v >= 0.66 else '+' if v >= 0.33 else '.' for v in fila) for fila in grilla)
