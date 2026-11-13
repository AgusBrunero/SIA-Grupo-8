"""Optimizadores de primer orden (Clase 12.1) y modos de entrenamiento (Clase 11, diap. 53).

Adaptado de TP3/perceptron/optimizadores.py. Cambios respecto del TP3: RMSProp con ϵ
dentro de la raíz (como en la diapositiva), η adaptativo con dos ventanas (k y k′) y sin
weight decay.

Interfaz común: paso(params, gradientes) actualiza in-place la lista de matrices de pesos
con la lista de ∂E/∂W que devuelve MLP.backward. Para entrenar dos redes a la vez
(encoder y decoder) se pasan las dos listas concatenadas.

Solo Adam tiene valores sugeridos por la cátedra (diap. 13) y Momentum un α típico
(diap. 5). Para el resto no hay valores por defecto: van en el config del experimento.
"""
from collections.abc import Iterator

import numpy as np

Params = list[np.ndarray]


class GD:
    """Gradiente descendente: w ← w − η ∂E/∂w  (Clase 12.1, diap. 2–3; NoLineal diap. 22 y 26)."""

    def __init__(self, eta: float) -> None:
        self.eta = eta

    def paso(self, params: Params, gradientes: Params) -> None:
        for w, g in zip(params, gradientes):
            w -= self.eta * g


class Momentum(GD):
    """Δw(t+1) = −η ∂E/∂w + α Δw(t), con α entre 0 y 1; típico 0,8 o 0,9  (Clase 12.1, diap. 5)."""

    def __init__(self, eta: float, alfa: float = 0.9) -> None:
        super().__init__(eta)
        self.alfa = alfa
        self._delta: Params | None = None

    def paso(self, params: Params, gradientes: Params) -> None:
        if self._delta is None:
            self._delta = [np.zeros_like(w) for w in params]
        for w, g, delta in zip(params, gradientes, self._delta):
            delta *= self.alfa
            delta -= self.eta * g
            w += delta


class RMSProp(GD):
    """S_t = γ S_(t−1) + (1 − γ) g_t² ;  Δw = −η g_t / √(S_t + ϵ)  (Clase 12.1, diap. 8).

    ϵ va dentro de la raíz, como en la diapositiva. La cátedra no sugiere valores.
    """

    def __init__(self, eta: float, gamma: float, epsilon: float) -> None:
        super().__init__(eta)
        self.gamma, self.epsilon = gamma, epsilon
        self._S: Params | None = None

    def paso(self, params: Params, gradientes: Params) -> None:
        if self._S is None:
            self._S = [np.zeros_like(w) for w in params]
        for w, g, S in zip(params, gradientes, self._S):
            S *= self.gamma
            S += (1.0 - self.gamma) * g * g
            w -= self.eta * g / np.sqrt(S + self.epsilon)


class Adam(GD):
    """Adam con corrección de sesgo (Clase 12.1, diap. 11–13).

    m_t = β1 m + (1 − β1) g ;  v_t = β2 v + (1 − β2) g² ;  m̂ = m_t/(1 − β1^t) ;
    v̂ = v_t/(1 − β2^t) ;  θ ← θ − α m̂ / (√v̂ + ϵ), con ϵ fuera de la raíz y m0 = v0 = 0.
    Valores sugeridos (diap. 13): α = 0,001, β1 = 0,9, β2 = 0,999, ϵ = 1e−8.
    No se usa la versión de OptimizacionNOLineal diap. 48, que no corrige el sesgo.
    El α de la diapositiva es self.eta.
    """

    def __init__(self, eta: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8) -> None:
        super().__init__(eta)
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon
        self._m: Params | None = None
        self._v: Params | None = None
        self._t = 0

    def paso(self, params: Params, gradientes: Params) -> None:
        if self._m is None or self._v is None:
            self._m = [np.zeros_like(w) for w in params]
            self._v = [np.zeros_like(w) for w in params]
        self._t += 1
        correccion_m = 1.0 - self.beta1 ** self._t
        correccion_v = 1.0 - self.beta2 ** self._t
        for w, g, m, v in zip(params, gradientes, self._m, self._v):
            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * g * g
            w -= self.eta * (m / correccion_m) / (np.sqrt(v / correccion_v) + self.epsilon)


class EtaAdaptativo:
    """η adaptativo (Clase 12.1, diap. 6–7), sobre cualquier optimizador.

        Δη = +a     si ΔE < 0 durante k épocas seguidas
        Δη = −b·η   si ΔE > 0 durante k′ épocas seguidas
        Δη = 0      en otro caso

    Se llama una vez por época con el error. La cátedra no da valores para a, b, k ni k′
    y dice que "en forma consistente se puede parametrizar de muchas formas".
    Interpretación nuestra: después de cambiar η la racha se reinicia, así que η cambia
    como mucho una vez cada k (o k′) épocas.
    """

    def __init__(self, optimizador: GD, a: float, b: float, k: int, k_prima: int) -> None:
        if k < 1 or k_prima < 1:
            raise ValueError('k y k_prima deben ser >= 1')
        self.optimizador = optimizador
        self.a, self.b, self.k, self.k_prima = a, b, k, k_prima
        self._error_anterior: float | None = None
        self._bajadas = 0
        self._subidas = 0

    def fin_de_epoca(self, error: float) -> None:
        if self._error_anterior is not None:
            delta_error = error - self._error_anterior
            self._bajadas = self._bajadas + 1 if delta_error < 0 else 0
            self._subidas = self._subidas + 1 if delta_error > 0 else 0
            if self._bajadas >= self.k:
                self.optimizador.eta += self.a
                self._bajadas = 0
            elif self._subidas >= self.k_prima:
                self.optimizador.eta -= self.b * self.optimizador.eta
                self._subidas = 0
        self._error_anterior = error


OPTIMIZADORES: dict[str, type[GD]] = {'gd': GD, 'momentum': Momentum, 'rmsprop': RMSProp, 'adam': Adam}


def crear(nombre: str, **hiperparametros: float) -> GD:
    """Crea un optimizador desde el config: {"nombre": "adam", "eta": 0.001, …}."""
    if nombre not in OPTIMIZADORES:
        raise ValueError(f'optimizador desconocido: {nombre!r}; opciones: {sorted(OPTIMIZADORES)}')
    return OPTIMIZADORES[nombre](**hiperparametros)


def lotes(p: int, tamano_lote: int, rng: np.random.Generator) -> Iterator[np.ndarray]:
    """Índices de los patrones de cada actualización dentro de una época (Clase 11, diap. 53).

    tamano_lote = p : lote / batch (una actualización con todos los patrones, en orden)
    tamano_lote = 1 : incremental / online (una actualización por patrón)
    otro valor      : mini lote / mini batch
    En online y minibatch los patrones se recorren en orden aleatorio, distinto en cada época.
    """
    if not 1 <= tamano_lote <= p:
        raise ValueError(f'tamano_lote debe estar entre 1 y p = {p}')
    if tamano_lote == p:
        yield np.arange(p)
        return
    orden = rng.permutation(p)
    for inicio in range(0, p, tamano_lote):
        yield orden[inicio:inicio + tamano_lote]
