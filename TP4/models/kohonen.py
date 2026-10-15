import numpy as np

SQRT3_2 = np.sqrt(3) / 2


def euclidean(X, W):
    """Distancia euclídea entre X (n,) y cada fila de W (m, n)."""
    return np.linalg.norm(W - X, axis=-1)


def linear_decay(start, end):
    """f(t, T): va de `start` (t = 0) a `end` (t = T − 1) linealmente."""
    return lambda t, T: start + (end - start) * t / max(T - 1, 1)


class Kohonen:
    """Red de Kohonen (SOM) sobre una grilla hexagonal k×k.

    Neurona `i` ↔ (fila, columna) = divmod(i, k). Las filas impares están corridas media
    neurona: así cada neurona interior tiene 6 vecinas a distancia exactamente 1.
    `learning_rate_f(t, T)` y `radius_f(t, T)` reciben la iteración global t y el total T.
    """

    def __init__(self, input_size, k, learning_rate_f, radius_f, distance_f=euclidean,
                 init="samples", seed=None):
        self.input_size = input_size
        self.k = k
        self.learning_rate_f = learning_rate_f
        self.radius_f = radius_f
        self.distance_f = distance_f
        self.init = init
        self.rng = np.random.default_rng(seed)
        filas, columnas = np.divmod(np.arange(k * k), k)
        self.positions = np.column_stack([columnas + 0.5 * (filas % 2), filas * SQRT3_2])
        self.grid_dist = np.linalg.norm(self.positions[:, None] - self.positions[None], axis=-1)
        self.weights = None
        self._processed_ = False

    def init_weights(self, inputs):
        """`samples`: cada neurona arranca en un registro al azar del dataset.
        `random`: uniforme dentro del rango de cada variable."""
        inputs = np.asarray(inputs, dtype=float)
        m = self.k * self.k
        if self.init == "samples":
            self.weights = inputs[self.rng.choice(len(inputs), m, replace=len(inputs) < m)].copy()
        elif self.init == "random":
            self.weights = self.rng.uniform(inputs.min(axis=0), inputs.max(axis=0),
                                            (m, self.input_size))
        else:
            raise ValueError(f"init desconocido: {self.init}")

    def get_bmu(self, X):
        return int(np.argmin(self.distance_f(X, self.weights)))

    def train(self, inputs, epochs):
        """Una época = una pasada por todos los registros en orden aleatorio."""
        inputs = np.asarray(inputs, dtype=float)
        if self.weights is None:
            self.init_weights(inputs)
        T = epochs * len(inputs)
        t = 0
        for _ in range(epochs):
            for X in inputs[self.rng.permutation(len(inputs))]:
                bmu = self.get_bmu(X)
                vecinas = self.grid_dist[bmu] <= self.radius_f(t, T)
                self.weights[vecinas] += self.learning_rate_f(t, T) * (X - self.weights[vecinas])
                t += 1
        self._processed_ = False

    def process(self, inputs):
        inputs = np.asarray(inputs, dtype=float)
        distancias = np.stack([self.distance_f(X, self.weights) for X in inputs])
        orden = np.argsort(distancias, axis=1)
        bmus, segundas = orden[:, 0], orden[:, 1]
        self.bmus = bmus
        self.clusters = np.column_stack(np.divmod(bmus, self.k))
        self.hit_map = np.bincount(bmus, minlength=self.k * self.k).reshape(self.k, self.k)
        self.error = float(distancias[np.arange(len(inputs)), bmus].mean())
        # Error topográfico: fracción de registros cuya 1.ª y 2.ª BMU no son vecinas en la grilla.
        self.topographic_error = float((self.grid_dist[bmus, segundas] > 1 + 1e-9).mean())
        self._processed_ = True

    def check_processed(self):
        if not self._processed_:
            raise RuntimeError("No se procesó ningún conjunto de datos. Use process(inputs).")

    # ----
    # EDAs
    # ----

    def get_clusters(self):
        """(fila, columna) de la BMU de cada registro procesado."""
        self.check_processed()
        return self.clusters

    def predict(self, X):
        """
        Mapea input al índice de su BMU.
        Retorna: tupla (fila, columna)
        """
        return divmod(self.get_bmu(np.asarray(X, dtype=float)), self.k)

    def get_u_matrix(self, radius=1):
        """
        Distancia promedio de cada neurona a sus vecinas en la grilla (radio 1 = las 6 adyacentes).
        Retorna: np.ndarray de forma (k, k)
        """
        u = np.zeros(self.k * self.k)
        for i, w in enumerate(self.weights):
            vecinas = (self.grid_dist[i] <= radius + 1e-9) & (self.grid_dist[i] > 0)
            u[i] = self.distance_f(w, self.weights[vecinas]).mean()
        return u.reshape(self.k, self.k)

    def get_hit_map(self):
        """
        Cantidad de registros por neurona.
        Retorna: np.ndarray de forma (k, k)
        """
        self.check_processed()
        return self.hit_map

    def get_component_matrix(self, feature_index):
        """
        Peso de la variable 'feature_index' en cada neurona.
        Retorna: np.ndarray de forma (k, k)
        """
        return self.weights[:, feature_index].reshape(self.k, self.k)

    def get_quantization_error(self):
        """Distancia promedio entre cada input y los pesos de su BMU."""
        self.check_processed()
        return self.error

    def get_topographic_error(self):
        self.check_processed()
        return self.topographic_error

    def get_dead_neurons(self):
        """Neuronas que no son BMU de ningún registro."""
        self.check_processed()
        return int((self.hit_map == 0).sum())
