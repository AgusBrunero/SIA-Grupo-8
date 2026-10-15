import numpy as np


class Oja:
    """Regla de Oja: una neurona lineal y = w·x cuyo peso converge a la primera componente
    principal de los datos (centrados), con ‖w‖ → 1.

        Δw = η · y · (x − y · w)

    `learning_rate_f(t, T)` recibe la iteración global t y el total T (misma convención que
    `Kohonen`). Guarda en `history` el vector w al final de cada época para estudiar la
    convergencia.
    """

    def __init__(self, input_size, learning_rate_f, init_norm=1.0, seed=None):
        self.input_size = input_size
        self.learning_rate_f = learning_rate_f
        self.init_norm = init_norm
        self.rng = np.random.default_rng(seed)
        self.weights = None
        self.history = None
        self.diverged = False

    def init_weights(self):
        """Dirección uniforme al azar, escalada a `init_norm`."""
        w = self.rng.uniform(-1, 1, self.input_size)
        self.weights = w * self.init_norm / np.linalg.norm(w)

    def train(self, inputs, epochs):
        """Una época = una pasada por todos los registros en orden aleatorio. Si w deja de ser
        finito (η demasiado grande) corta y marca `diverged`; la historia queda en NaN."""
        inputs = np.asarray(inputs, dtype=float)
        if self.weights is None:
            self.init_weights()
        n = len(inputs)
        T = epochs * n
        self.history = np.full((epochs, self.input_size), np.nan)
        w = self.weights
        with np.errstate(over="ignore", invalid="ignore"):
            for epoch in range(epochs):
                for j, i in enumerate(self.rng.permutation(n)):
                    x = inputs[i]
                    y = w @ x
                    w = w + self.learning_rate_f(epoch * n + j, T) * y * (x - y * w)
                if not np.all(np.isfinite(w)) or np.linalg.norm(w) > 1e6:
                    self.diverged = True
                    break
                self.history[epoch] = w
        self.weights = w
        return self

    def project(self, X):
        """Índice de cada registro: proyección sobre w."""
        return np.asarray(X, dtype=float) @ self.weights
