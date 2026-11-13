"""Ruido para el Denoising Autoencoder (consigna 1b).

Autoencoders diap. 22: el ruido "se lo caracteriza numericamente (e.g. salt-and-pepper)
o en general con una función de distribución de probabilidad (e.g. Gaussiano, Rayleigh)"
y se agrega a X generando X̃. La cátedra no da niveles: los fija el config.
Solo se implementa salt-and-pepper; el gaussiano es [EXTRA] no aprobado.
"""
import numpy as np


def sal_y_pimienta(X: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """Invierte cada píxel de X (en {0, 1}) con probabilidad p, de forma independiente.

    Un píxel apagado que se enciende es "sal" y uno encendido que se apaga es "pimienta".
    Devuelve una copia; X no se modifica.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f'p debe estar en [0, 1] y vale {p}')
    invertir = rng.random(X.shape) < p
    return np.where(invertir, 1.0 - X, X)
