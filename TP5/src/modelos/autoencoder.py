"""Autoencoder básico (consigna 1a) y su ciclo de entrenamiento.

Autoencoders diap. 3: "dos MLP", la salida del primero (encoder) es la entrada del segundo
(decoder); el decoder tiene la distribución de neuronas invertida y su salida tiene la
dimensión de la entrada.  Z = f(X),  X′ = g(Z),  L(X, X′) = ‖X − X′‖²  (diap. 3–4).
Diap. 4: se entrena con cualquier método de un MLP, poniendo "como salida esperada el
mismo patrón X". Acá la salida esperada ζ es un argumento aparte de la entrada para que
el DAE (diap. 22: entrada X̃, salida esperada X) use el mismo código.
"""
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from src.modelos.metricas import OBJETIVO_PIXELES, pixeles_incorrectos, validar_codificacion
from src.nn.mlp import MLP
from src.nn.optimizadores import GD, EtaAdaptativo, lotes
from src.nn.perdidas import PERDIDAS


class Autoencoder:
    def __init__(self, capas_encoder: list[int], activaciones_encoder: list[str],
                 activacion_salida: str, semilla: int, beta: float = 1.0,
                 amplitud_inicial: float = 0.1) -> None:
        """capas_encoder = [entrada, …, latente], p. ej. [35, 20, 2].

        El decoder es el espejo (diap. 3): capas invertidas, las mismas activaciones
        ocultas en orden inverso y `activacion_salida` en la última capa. La activación
        de la capa latente es la última de `activaciones_encoder`.
        """
        capas_decoder = capas_encoder[::-1]
        activaciones_decoder = activaciones_encoder[-2::-1] + [activacion_salida]
        # Dos semillas distintas derivadas de la del experimento, una por red
        semilla_encoder, semilla_decoder = (int(s) for s in np.random.SeedSequence(semilla).generate_state(2))
        self.encoder = MLP(capas_encoder, activaciones_encoder, semilla_encoder, beta, amplitud_inicial)
        self.decoder = MLP(capas_decoder, activaciones_decoder, semilla_decoder, beta, amplitud_inicial)
        self.activacion_salida = activacion_salida
        self.semilla = semilla

    @property
    def redes(self) -> list[MLP]:
        return [self.encoder, self.decoder]

    @property
    def params(self) -> list[np.ndarray]:
        return self.encoder.params + self.decoder.params

    def n_params(self) -> int:
        return self.encoder.n_params() + self.decoder.n_params()

    def codificar(self, X: np.ndarray) -> np.ndarray:
        """Z = f(X): el punto de cada patrón en el espacio latente."""
        return self.encoder.forward(X)

    def decodificar(self, Z: np.ndarray) -> np.ndarray:
        """X′ = g(Z). Sirve para generar desde un z cualquiera (diap. 34)."""
        return self.decoder.forward(Z)

    def reconstruir(self, X: np.ndarray) -> np.ndarray:
        return self.decodificar(self.codificar(X))

    def gradientes(self, X: np.ndarray, zeta: np.ndarray, perdida: str = 'error_cuadratico',
                   reduccion: str = 'suma') -> tuple[float, list[np.ndarray]]:
        """Pérdida y ∂E/∂W de todas las capas (encoder y después decoder, como en params).

        El ∂E/∂Z que devuelve el backward del decoder es el ∂E/∂O del encoder.
        """
        valor, dE_dO = PERDIDAS[perdida](self.reconstruir(X), zeta, reduccion)
        gradientes_decoder, dE_dZ = self.decoder.backward(dE_dO)
        gradientes_encoder, _ = self.encoder.backward(dE_dZ)
        return valor, gradientes_encoder + gradientes_decoder


@dataclass
class Historial:
    """Una fila por época. El error es el de la pérdida con reducción 'suma' sobre los
    patrones, medido sobre (X, ζ) completo al terminar la época."""
    error: list[float] = field(default_factory=list)
    max_pixeles: list[int] = field(default_factory=list)
    total_pixeles: list[int] = field(default_factory=list)
    eta: list[float] = field(default_factory=list)
    epoca_objetivo: int | None = None      # primera época (desde 1) con max_pixeles ≤ objetivo
    segundos: float = 0.0


def evaluar(ae: Autoencoder, X: np.ndarray, zeta: np.ndarray, perdida: str,
            codificacion: str) -> tuple[float, np.ndarray]:
    """(error con reducción 'suma', píxeles incorrectos por patrón) de reconstruir X contra ζ."""
    salida = ae.reconstruir(X)
    error, _ = PERDIDAS[perdida](salida, zeta, 'suma')
    return error, pixeles_incorrectos(salida, zeta, codificacion)


def entrenar(ae: Autoencoder, X: np.ndarray, zeta: np.ndarray, optimizador: GD, epocas: int,
             tamano_lote: int, rng: np.random.Generator, perdida: str = 'error_cuadratico',
             codificacion: str = '01', reduccion: str = 'suma',
             eta_adaptativo: EtaAdaptativo | None = None,
             cortar_en_objetivo: bool = False,
             corromper: Callable[[np.ndarray], np.ndarray] | None = None) -> Historial:
    """Entrena por épocas (pseudocódigo de la Clase 11, diap. 52 y 54).

    tamano_lote = len(X) es batch, 1 es online y lo demás minibatch (Clase 11, diap. 53).
    Con cortar_en_objetivo=True se detiene en la primera época con todos los patrones a
    ≤ 1 píxel; si no, corre todas las épocas y solo registra cuándo se alcanzó.

    Denoising (Autoencoders diap. 22): si se pasa `corromper`, en cada época la entrada es
    X̃ = corromper(X), con ruido nuevo, y la salida esperada sigue siendo ζ sin alterar.
    El historial se mide siempre con la entrada X sin ruido.
    """
    validar_codificacion(ae.activacion_salida, codificacion)
    historial = Historial()
    inicio = time.perf_counter()
    for epoca in range(1, epocas + 1):
        entrada = X if corromper is None else corromper(X)
        for indices in lotes(len(X), tamano_lote, rng):
            _, gradientes = ae.gradientes(entrada[indices], zeta[indices], perdida, reduccion)
            optimizador.paso(ae.params, gradientes)
        error, incorrectos = evaluar(ae, X, zeta, perdida, codificacion)
        historial.error.append(error)
        historial.max_pixeles.append(int(incorrectos.max()))
        historial.total_pixeles.append(int(incorrectos.sum()))
        historial.eta.append(optimizador.eta)
        if eta_adaptativo is not None:
            eta_adaptativo.fin_de_epoca(error)
        if incorrectos.max() <= OBJETIVO_PIXELES and historial.epoca_objetivo is None:
            historial.epoca_objetivo = epoca
            if cortar_en_objetivo:
                break
    historial.segundos = time.perf_counter() - inicio
    return historial
