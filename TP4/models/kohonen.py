import numpy as np
import math

class Neuron:
    def __init__(self, input_size, x_idx, y_idx, seed=None):
        self.input_size=input_size
        rng=np.random.default_rng(seed)
        self.weights = rng.random(input_size)
        self.x_idx = x_idx
        self.y_idx = y_idx
        self.x = x_idx - 0.5 if y_idx%2==0 else float(x_idx)
        self.y = y_idx*math.sqrt(3)/2

    def update_weights(self, X, learning_rate):
        X = np.array(X)
        self.weights += learning_rate*(X-self.weights)

    def is_neighbour(self, other, radius):
        return math.hypot(other.x - self.x, other.y - self.y) <= radius

class Kohonen:
    def __init__(self, input_size, k, learning_rate_f, radius_f, similarity_f):
        self.k = k
        self.similarity_f = similarity_f
        self.radius_f = radius_f
        self.learning_rate_f = learning_rate_f
        self.hit_map = None
        self.clusters = None
        self.error = None
        self._processed_ = False
        self.network = [
            Neuron(input_size = input_size, x_idx = x, y_idx = y)
            for y in range(k)
            for x in range(k)
        ]

    def get_bmu(self,X):
        return min(self.network, key=lambda n: self.similarity_f(X,n.weights))

    def train(self, inputs):
        for i, X in enumerate(inputs):
            bmu = self.get_bmu(X)
            learning_rate = self.learning_rate_f(i)
            radius = self.radius_f(i)
            for other in self.network:
                if other.is_neighbour(bmu, radius):
                    other.update_weights(X, learning_rate)

    def process(self, inputs):
        self.hit_map = np.zeros((self.k, self.k), dtype=int)
        errors = np.zeros(len(inputs))
        self.clusters = np.zeros((len(inputs),2), dtype=int)
        for i, X in enumerate(inputs):
            bmu = self.get_bmu(X)
            self.hit_map[bmu.y_idx,bmu.x_idx] += 1
            self.clusters[i] = (bmu.y_idx, bmu.x_idx)
            errors[i] = self.similarity_f(X,bmu.weights)
        self.error = float(np.mean(errors))
        self._processed_ = True

    def check_processed(self):
        if not self._processed_:
            raise RuntimeError("No se procesó ningún conjunto de datos. Use process(inputs).")

    # ----
    # EDAs
    # ----

    def get_clusters(self):
        self.check_processed()
        return self.clusters

    def predict(self, X):
        """
        Mapea input al índice de su BMU.
        Retorna: tupla (i1, j1)
        """
        bmu = self.get_bmu(X)
        return (bmu.y_idx, bmu.x_idx)

    def get_u_matrix(self, radius):
        """
        Calcula la Matriz U para cierto radio de vecindad
        Retorna: np.ndarray de forma (k, k)
        """
        u_matrix = np.zeros((self.k, self.k))
        for neuron in self.network:
            neighbors = [
                other for other in self.network
                if other is not neuron and neuron.is_neighbour(other, radius=radius)
            ]
            if neighbors:
                distances = [self.similarity_f(neuron.weights, n.weights) for n in neighbors]
                u_matrix[neuron.y_idx, neuron.x_idx] = np.mean(distances)
        return u_matrix

    def get_hit_map(self):
        """
        Mapa de calor de datos por neurona.
        Retorna: np.ndarray de forma (k, k)
        """
        self.check_processed()
        return self.hit_map

    def get_component_matrix(self, feature_index):
        """
        Extrae el valor del peso de la variable 'feature_index' para cada neurona del grid.
        Retorna: np.ndarray de forma (k, k)
        """
        comp_matrix = np.zeros((self.k, self.k))
        for neuron in self.network:
            comp_matrix[neuron.y_idx, neuron.x_idx] = neuron.weights[feature_index]
        return comp_matrix

    def get_quantization_error(self):
        """
        Mide la distancia promedio entre cada input y los pesos de su BMU.
        Retorna: float
        """
        self.check_processed()
        return self.error