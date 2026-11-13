"""Perceptrón multicapa con backpropagation matricial en numpy (Clase 11).

Adaptado de TP3/perceptron/mlp.py. Cambios respecto del TP3: activación por capa, bias
como x0 = 1 dentro de la matriz de pesos, solo inicialización uniforme chica, sin ReLU,
softmax ni dropout, y backward desde un gradiente externo sobre la salida.

Notación (Clase 11, diap. 38 y 69): m = capa (1…M), V^m = salida de la capa m, V^0 = x,
O = V^M, h^m = excitación, θ = activación, p = cantidad de datos (filas de X).

Feed-forward (diap. 13–14):   h^m = [1, V^(m−1)] · W^m ,   V^m = θ_m(h^m)
Bias (diap. 66–68): a la entrada de cada capa se le agrega x0 = 1; la fila 0 de W^m es
el peso w0 (el bias), "como si fuera un peso sináptico más".
W^m tiene forma (n_(m−1) + 1, n_m): fila = neurona de la capa anterior, columna = neurona
de la capa m.

Backprop (diap. 40–51). La cátedra define δ con el signo de (ζ − O) y actualiza
w ← w + η δ V. Acá se devuelve el gradiente ∂E/∂W, que es −δ_cátedra · V, y el
optimizador hace w ← w − η ∂E/∂W: es la misma actualización.
    capa de salida:  ∂E/∂h^M = ∂E/∂O ⊙ θ'_M(h^M)
    capas ocultas:   ∂E/∂h^m = (∂E/∂h^(m+1) · W^(m+1)ᵀ sin la fila del bias) ⊙ θ'_m(h^m)
    pesos:           ∂E/∂W^m = [1, V^(m−1)]ᵀ · ∂E/∂h^m       (suma sobre los p datos, diap. 54)
"""
import numpy as np

from src.nn.activaciones import Activacion
from src.nn.perdidas import PERDIDAS


class MLP:
    def __init__(self, capas: list[int], activaciones: list[str], semilla: int,
                 beta: float = 1.0, amplitud_inicial: float = 0.1) -> None:
        """capas = [n_0, n_1, …, n_M] (n_0 = entradas); activaciones = una por capa 1…M.

        Pesos iniciales: uniforme en (−amplitud_inicial, amplitud_inicial). Clase 11,
        diap. 58: valores aleatorios chicos; en cero aparece el problema de simetría.
        """
        if len(capas) < 2:
            raise ValueError('hacen falta al menos la capa de entrada y una capa con pesos')
        if len(activaciones) != len(capas) - 1:
            raise ValueError(f'hay {len(capas) - 1} capas con pesos y {len(activaciones)} activaciones')
        if amplitud_inicial <= 0:
            raise ValueError('amplitud_inicial debe ser > 0 (pesos en cero: problema de simetría)')
        self.capas = list(capas)
        self.activaciones = [Activacion(nombre, beta) for nombre in activaciones]
        self.semilla = semilla
        self.beta = beta
        self.amplitud_inicial = amplitud_inicial
        rng = np.random.default_rng(semilla)
        self.W: list[np.ndarray] = [
            rng.uniform(-amplitud_inicial, amplitud_inicial, size=(n_entrada + 1, n_salida))
            for n_entrada, n_salida in zip(capas[:-1], capas[1:])
        ]
        self._cache: tuple[list[np.ndarray], list[np.ndarray]] | None = None

    # ---------- parámetros ----------

    @property
    def params(self) -> list[np.ndarray]:
        """Las matrices W^1…W^M (los mismos arrays: los optimizadores las modifican in-place)."""
        return self.W

    def n_params(self) -> int:
        return sum(W.size for W in self.W)

    def vector(self) -> np.ndarray:
        """Todos los pesos en un único vector (para scipy.optimize)."""
        return np.concatenate([W.ravel() for W in self.W])

    def cargar_vector(self, vector: np.ndarray) -> None:
        if vector.size != self.n_params():
            raise ValueError(f'el vector tiene {vector.size} pesos y la red {self.n_params()}')
        inicio = 0
        for W in self.W:
            W[...] = vector[inicio:inicio + W.size].reshape(W.shape)
            inicio += W.size

    # ---------- feed-forward ----------

    @staticmethod
    def _con_x0(V: np.ndarray) -> np.ndarray:
        """Agrega la columna x0 = 1 (Clase 11, diap. 66)."""
        return np.hstack([np.ones((V.shape[0], 1)), V])

    def salidas_por_capa(self, X: np.ndarray) -> list[np.ndarray]:
        """[V^1, …, V^M]: la salida de cada capa. La última es O. Guarda lo que usa backward."""
        V = np.asarray(X, dtype=np.float64)
        entradas, excitaciones, salidas = [], [], []
        for W, theta in zip(self.W, self.activaciones):
            entrada = self._con_x0(V)
            h = entrada @ W
            V = theta(h)
            entradas.append(entrada)
            excitaciones.append(h)
            salidas.append(V)
        self._cache = (entradas, excitaciones)
        return salidas

    def forward(self, X: np.ndarray) -> np.ndarray:
        """O = V^M."""
        return self.salidas_por_capa(X)[-1]

    # ---------- backprop ----------

    def backward(self, dE_dO: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        """Retropropaga un gradiente ∂E/∂O aplicado sobre la salida del último forward.

        Devuelve ([∂E/∂W^1, …, ∂E/∂W^M], ∂E/∂X). ∂E/∂X permite encadenar redes: es lo
        que el decoder le pasa al encoder en un autoencoder (Autoencoders diap. 3 y 86).
        """
        if self._cache is None:
            raise RuntimeError('backward necesita un forward previo')
        entradas, excitaciones = self._cache
        gradientes: list[np.ndarray] = [np.empty(0)] * len(self.W)
        dE_dV = np.asarray(dE_dO, dtype=np.float64)
        for m in range(len(self.W) - 1, -1, -1):
            dE_dh = dE_dV * self.activaciones[m].derivada(excitaciones[m])
            gradientes[m] = entradas[m].T @ dE_dh
            dE_dV = dE_dh @ self.W[m][1:].T          # sin la fila 0: x0 = 1 no depende de la capa anterior
        return gradientes, dE_dV

    def gradientes(self, X: np.ndarray, zeta: np.ndarray, perdida: str = 'error_cuadratico',
                   reduccion: str = 'suma') -> tuple[float, list[np.ndarray]]:
        """Pérdida y ∂E/∂W de cada capa para los datos (X, ζ)."""
        O = self.forward(X)
        valor, dE_dO = PERDIDAS[perdida](O, zeta, reduccion)
        gradientes, _ = self.backward(dE_dO)
        return valor, gradientes
