import numpy as np

# Funciones matemáticas usando numpy para que admitan vectores
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x): return sigmoid(x) * (1 - sigmoid(x))
def linear(x): return x
def linear_derivative(x): return np.ones(np.shape(x))
def relu(x): return np.maximum(0, x)
# En h = 0 no es derivable; se toma 0
def relu_derivative(x): return np.where(x > 0, 1.0, 0.0)
def heaviside(x): return np.where(x > 0, 1, 0)
def tanh(x): return np.tanh(x)
# La derivada recibe h, no tanh(h)
def tanh_derivative(x): return 1 - np.tanh(x) ** 2

class Neuron:
    def __init__(self, input_size, activation_function, derivative_function, learning_rate, seed=None):
        self.activation_function = activation_function
        self.derivative_function = derivative_function
        self.learning_rate = learning_rate

        # Semilla para que los experimentos sean reproducibles
        rng = np.random.default_rng(seed)
        # Supuestamente el truco del umbral y el x0 es peor computacionalmente (NC)
        self.weights = rng.random(input_size)
        self.bias = rng.random()

    def compute(self, X):
        # np.dot calcula el producto punto
        return np.dot(X, self.weights) + self.bias

    def activate(self, X):
        return self.activation_function(self.compute(X))

    def learn_online(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)

        for x, y in zip(X, Y):
            # h en este caso es un escalar, compute calcula vectorialmente x.w
            h = self.compute(x)
            output = self.activation_function(h)

            factor = self.learning_rate * (y - output) * self.derivative_function(h)

            # Operaciones vectorizadas
            self.weights += factor * x
            self.bias += factor

    def learn_batch(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)

        # h en este caso es un vector, compute calcula matricialmente x.w
        h = self.compute(X)
        outputs = self.activation_function(h)

        # factors en este caso es un vector porque se opera vectorialmente.
        # Se divide por la cantidad de muestras (gradiente promedio) para que el
        # learning rate no dependa del tamaño del dataset
        factors = self.learning_rate * (Y - outputs) * self.derivative_function(h) / len(X)

        # x.t es x transverso, o sea que queda un vector con la suma de cada peso
        self.weights += np.dot(X.T, factors)
        self.bias += np.sum(factors)

    def get_error(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)

        # compute calcula matricialmente x.w y activate devuelve el vector de outputs
        outputs = self.activate(X)
        # np.sum hace una suma de todos los elementos de dicha operación vectorial
        return 0.5 * np.sum((Y - outputs) ** 2)

# Pequeño test con OR
if __name__ == '__main__':
    # Datos de entrada (Compuerta Lógica OR)
    X = [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]]
    Y = [0, 1, 1, 1]

    # Instanciamos la neurona
    neuron = Neuron(input_size=2,
                    activation_function=tanh,
                    derivative_function=tanh_derivative,
                    learning_rate=0.1)

    # Entrenamos por lotes 1000 épocas
    for epoch in range(1000):
        neuron.learn_batch(X, Y)

    # Probamos los resultados
    print("Resultados de predicción:", np.round(neuron.activate(X), 3))
    print("Error final:", neuron.get_error(X, Y))