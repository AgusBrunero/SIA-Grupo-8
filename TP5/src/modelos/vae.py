"""Autoencoder Variacional (consigna 2b), a mano en numpy.

Convención (doc de cátedra §2.1, decisiones D3–D5; es la del Colab de la diap. 90):
  - el encoder produce μ(x) y s(x) = log σ²(x), dos salidas lineales (una capa identidad
    de 2k neuronas: las primeras k son μ y las últimas k son s);
  - reparametrización (Autoencoders diap. 45, 80, 83):  z = μ + ϵ ⊙ exp(s/2),  ϵ ~ N(0, 1);
  - J = L + λR  (diap. 78, ec. 13), con
        L = reconstrucción: BCE sumada sobre los píxeles (Colab; fórmula en diap. 51)
        R = KL(q‖p) = ½ Σ_k (exp(s_k) + μ_k² − 1 − s_k)   (diap. 77)
    las dos promediadas sobre el batch.

Gradientes (diap. 84–89):
  - decoder: solo el gradiente de reconstrucción, "igual a un perceptrón multicapa" (diap. 88);
  - encoder: reconstrucción (lo que viene del decoder por ∂z/∂μ = 1 y ∂z/∂s = ½ ϵ exp(s/2))
    más regularización (∂R/∂μ = μ, ∂R/∂s = ½(exp(s) − 1)), sumados (diap. 89):
        δ_μ = ∂L/∂z + λ μ
        δ_s = ∂L/∂z ⊙ ½ ϵ exp(s/2) + λ ½ (exp(s) − 1)
Las diapositivas escriben ∂z/∂σ = ϵ (diap. 87); con s = log σ² la derivada es la de arriba
(DESVIOS.md, V1). Se verifica con gradient check con ϵ fijo.
"""
import time
from dataclasses import dataclass, field

import numpy as np

from src.nn.mlp import MLP
from src.nn.optimizadores import GD, lotes
from src.nn.perdidas import bce


def kl_normal_estandar(mu: np.ndarray, s: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """R = ½ Σ_k (exp(s) + μ² − 1 − s), promediada sobre el batch (Autoencoders diap. 77).

    Devuelve (R, ∂R/∂μ, ∂R/∂s). Vale 0 cuando μ = 0 y s = 0 (q = p = N(0, I)).
    """
    p = mu.shape[0]
    valor = 0.5 * float(np.sum(np.exp(s) + mu ** 2 - 1.0 - s)) / p
    return valor, mu / p, 0.5 * (np.exp(s) - 1.0) / p


class VAE:
    def __init__(self, capas_encoder: list[int], capas_decoder: list[int], activacion_oculta: str,
                 semilla: int, beta: float = 1.0, amplitud_inicial: float = 0.1) -> None:
        """capas_encoder = [entrada, ocultas…, k] y capas_decoder = [k, ocultas…, entrada].

        La última capa del encoder tiene 2k neuronas lineales (μ y s). La salida del
        decoder es logística, para usar BCE con datos en [0, 1].
        """
        if capas_encoder[-1] != capas_decoder[0] or capas_encoder[0] != capas_decoder[-1]:
            raise ValueError('el latente y la entrada tienen que coincidir entre encoder y decoder')
        self.k = capas_encoder[-1]
        semilla_encoder, semilla_decoder = (int(x) for x in np.random.SeedSequence(semilla).generate_state(2))
        self.encoder = MLP(capas_encoder[:-1] + [2 * self.k],
                           [activacion_oculta] * (len(capas_encoder) - 2) + ['identidad'],
                           semilla_encoder, beta, amplitud_inicial)
        self.decoder = MLP(capas_decoder, [activacion_oculta] * (len(capas_decoder) - 2) + ['logistica'],
                           semilla_decoder, beta, amplitud_inicial)
        self.semilla = semilla

    @property
    def params(self) -> list[np.ndarray]:
        return self.encoder.params + self.decoder.params

    def codificar(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(μ(x), s(x) = log σ²(x))."""
        salida = self.encoder.forward(X)
        return salida[:, :self.k], salida[:, self.k:]

    def decodificar(self, Z: np.ndarray) -> np.ndarray:
        """x̂ = decoder(z). Con z ~ N(0, I) genera muestras nuevas."""
        return self.decoder.forward(Z)

    def gradientes(self, X: np.ndarray, epsilon: np.ndarray,
                   lam: float) -> tuple[float, float, list[np.ndarray]]:
        """Un paso completo con ϵ dado: devuelve (L, R, ∂J/∂W de encoder y decoder), J = L + λR."""
        mu, s = self.codificar(X)
        sigma = np.exp(s / 2.0)
        z = mu + epsilon * sigma                                   # reparametrización
        L, dL_dsalida = bce(self.decoder.forward(z), X, 'media')
        gradientes_decoder, dL_dz = self.decoder.backward(dL_dsalida)   # solo reconstrucción
        R, dR_dmu, dR_ds = kl_normal_estandar(mu, s)
        delta_mu = dL_dz + lam * dR_dmu
        delta_s = dL_dz * 0.5 * epsilon * sigma + lam * dR_ds
        gradientes_encoder, _ = self.encoder.backward(np.hstack([delta_mu, delta_s]))
        return L, R, gradientes_encoder + gradientes_decoder


@dataclass
class HistorialVAE:
    """Una fila por época: L y R promediadas sobre los patrones, con los ϵ de esa época."""
    reconstruccion: list[float] = field(default_factory=list)      # L
    regularizacion: list[float] = field(default_factory=list)      # R (sin multiplicar por λ)
    segundos: float = 0.0


def entrenar_vae(vae: VAE, X: np.ndarray, optimizador: GD, epocas: int, tamano_lote: int, lam: float,
                 rng_lotes: np.random.Generator, rng_epsilon: np.random.Generator) -> HistorialVAE:
    """Entrena minimizando J = L + λR. En cada actualización se samplea un ϵ ~ N(0, 1) nuevo
    por patrón y dimensión del latente ("Pick Random Sample ϵ", Autoencoders diap. 80–81)."""
    historial = HistorialVAE()
    inicio = time.perf_counter()
    for _ in range(epocas):
        suma_L = suma_R = 0.0
        for indices in lotes(len(X), tamano_lote, rng_lotes):
            epsilon = rng_epsilon.standard_normal((len(indices), vae.k))
            L, R, gradientes = vae.gradientes(X[indices], epsilon, lam)
            optimizador.paso(vae.params, gradientes)
            suma_L += L * len(indices)
            suma_R += R * len(indices)
        historial.reconstruccion.append(suma_L / len(X))
        historial.regularizacion.append(suma_R / len(X))
    historial.segundos = time.perf_counter() - inicio
    return historial
