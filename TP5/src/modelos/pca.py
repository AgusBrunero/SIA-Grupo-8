"""PCA con la SVD de numpy, como línea base del autoencoder (decisión D8).

Autoencoders diap. 7–16: para un autoencoder lineal, si X tiene media cero (diap. 14) y
los pesos del decoder son ortonormales (diap. 11), el código Z coincide con las
proyecciones sobre las componentes principales: T_PCA(X) = X E = Z (ec. 6, diap. 15).
Con X = U Σ Vᵀ (X centrada), las columnas de V son las componentes E.
No se usa sklearn (CLAUDE.md §2).
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PCA:
    media: np.ndarray               # (n,)
    componentes: np.ndarray         # (k, n): filas ortonormales, de mayor a menor varianza
    varianza_explicada: np.ndarray  # (k,): fracción de la varianza total de cada componente

    def proyectar(self, X: np.ndarray) -> np.ndarray:
        """Z = (X − media) E."""
        return (X - self.media) @ self.componentes.T

    def reconstruir(self, Z: np.ndarray) -> np.ndarray:
        """X′ = Z Eᵀ + media."""
        return Z @ self.componentes + self.media


def ajustar_pca(X: np.ndarray, k: int) -> PCA:
    media = X.mean(axis=0)
    _, singulares, Vt = np.linalg.svd(X - media, full_matrices=False)
    varianzas = singulares ** 2
    return PCA(media=media, componentes=Vt[:k], varianza_explicada=varianzas[:k] / varianzas.sum())
