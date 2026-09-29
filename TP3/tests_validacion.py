"""Ejercicio de validación: comprueba que las herramientas implementadas funcionan.

No se presenta, pero es la mejor forma de encontrar bugs antes de pasar a los datos reales.

Uso (desde TP3/):  python tests_validacion.py      (o  python -m pytest tests_validacion.py)
"""
import numpy as np

from perceptron.perceptron import (Neuron, sign, step_derivative, linear, linear_derivative,
                                   tanh, tanh_derivative)
from perceptron.mlp import MLP
from perceptron.entrenamiento import TrainConfig, train
from perceptron import optimizadores

X_LOGIC = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]], dtype=float)
Y_AND = np.array([-1, -1, -1, 1])
Y_XOR = np.array([1, 1, -1, -1])


# ---------- perceptrón simple ----------

def step_train(Y, epochs=100, seed=0):
    neuron = Neuron(2, sign, step_derivative, learning_rate=0.1, seed=seed)
    for epoch in range(1, epochs + 1):
        neuron.learn_online(X_LOGIC, Y)
        if np.array_equal(neuron.activate(X_LOGIC), Y):
            return neuron, epoch
    return neuron, None


def test_step_and_converges():
    for seed in range(10):
        neuron, epoch = step_train(Y_AND, seed=seed)
        assert epoch is not None, f'AND no convergió con seed {seed}'


def test_step_xor_does_not_converge():
    # XOR no es linealmente separable: el perceptrón simple nunca acierta los 4
    for seed in range(10):
        _, epoch = step_train(Y_XOR, epochs=1000, seed=seed)
        assert epoch is None


def test_linear_fits_identity():
    x = np.linspace(-5, 5, 50).reshape(-1, 1)
    y = x.ravel()
    neuron = Neuron(1, linear, linear_derivative, learning_rate=0.01, seed=0)
    for _ in range(500):
        neuron.learn_online(x, y)
    assert abs(neuron.weights[0] - 1) < 1e-3 and abs(neuron.bias) < 1e-3


def test_nonlinear_fits_tanh():
    x = np.linspace(-3, 3, 50).reshape(-1, 1)
    y = np.tanh(x).ravel()
    neuron = Neuron(1, tanh, tanh_derivative, learning_rate=0.05, seed=0)
    for _ in range(2000):
        neuron.learn_online(x, y)
    assert neuron.get_error(x, y) < 1e-4


# ---------- perceptrón multicapa ----------

def xor_mlp(layers, seed, epochs=2000):
    model = MLP(layers, hidden='tanh', output='tanh', seed=seed, init='xavier')
    Y = Y_XOR.reshape(-1, 1).astype(float)
    opt = optimizadores.GD(lr=0.1)
    for _ in range(epochs):
        _, grads = model.gradients(X_LOGIC, Y)
        opt.step(model.params, grads)
    return np.array_equal(np.sign(model.forward(X_LOGIC).ravel()), Y_XOR)


def test_mlp_solves_xor():
    for layers in ([2, 2, 1], [2, 3, 2, 1]):
        solved = sum(xor_mlp(layers, seed) for seed in range(10))
        # [2,2,1] puede caer en un mínimo local con alguna semilla; se pide mayoría
        assert solved >= 8, f'{layers}: resolvió XOR solo en {solved}/10 semillas'


def numerical_gradients(model, X, Y, eps=1e-6):
    grads = []
    for p in model.params:
        g = np.zeros_like(p)
        for idx in np.ndindex(p.shape):
            old = p[idx]
            p[idx] = old + eps
            up = model.loss(model.forward(X), Y)
            p[idx] = old - eps
            down = model.loss(model.forward(X), Y)
            p[idx] = old
            g[idx] = (up - down) / (2 * eps)
        grads.append(g)
    return grads


def test_gradient_check():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(7, 5))
    cases = [('tanh', 'logistic', 'mse'), ('logistic', 'tanh', 'mse'),
             ('relu', 'softmax', 'cross_entropy'), ('tanh', 'softmax', 'cross_entropy')]
    for hidden, output, loss in cases:
        model = MLP([5, 4, 3, 3], hidden=hidden, output=output, loss=loss, beta=0.7, seed=1)
        # Bias no nulos: con b = 0 una ReLU que recibe todo ceros queda justo en h = 0,
        # donde no es derivable y la diferencia finita no coincide (no es un bug de backprop)
        model.b = [rng.uniform(0.1, 0.5, b.shape) for b in model.b]
        Y = np.eye(3)[rng.integers(0, 3, 7)]
        _, analytic = model.gradients(X, Y)
        numeric = numerical_gradients(model, X, Y)
        for a, n in zip(analytic, numeric):
            rel = np.abs(a - n).max() / max(1e-8, np.abs(a).max() + np.abs(n).max())
            assert rel < 1e-6, f'{hidden}/{output}/{loss}: error relativo {rel:.2e}'


def test_input_gradient():
    # La saliency de la pizarra: ∂h_salida/∂x comparado contra diferencias finitas
    rng = np.random.default_rng(2)
    model = MLP([6, 5, 4, 3], hidden='tanh', output='softmax', loss='cross_entropy', seed=4)
    x = rng.normal(size=(1, 6))
    analytic = model.input_gradient(x, 1)[0]

    def logit(v):
        O = v
        for l, (W, b) in enumerate(zip(model.W, model.b)):
            h = O @ W + b
            O = model.activation_of(l)(h) if l < len(model.W) - 1 else h
        return h[0, 1]

    eps = 1e-6
    numeric = np.array([(logit(x + eps * e) - logit(x - eps * e)) / (2 * eps) for e in np.eye(6)[:, None, :]])
    assert np.allclose(analytic, numeric, atol=1e-6)


def test_hand_calculation_221():
    """Una iteración de feed-forward y backprop para [2,2,1] calculada 'a mano' (escalar por escalar)."""
    model = MLP([2, 2, 1], hidden='tanh', output='tanh', seed=0)
    model.W = [np.array([[0.1, -0.2], [0.3, 0.4]]), np.array([[0.5], [-0.6]])]
    model.b = [np.array([0.05, -0.05]), np.array([0.1])]
    x, z = np.array([1.0, -1.0]), 1.0

    # Capa oculta: h_j = Σ_i x_i w_ij + b_j ;  V_j = tanh(h_j)
    h1 = 1 * 0.1 + (-1) * 0.3 + 0.05          # −0.15
    h2 = 1 * (-0.2) + (-1) * 0.4 - 0.05       # −0.65
    V1, V2 = np.tanh(h1), np.tanh(h2)
    # Salida: H = V1 W1 + V2 W2 + b ;  O = tanh(H)
    H = V1 * 0.5 + V2 * (-0.6) + 0.1
    O = np.tanh(H)
    # δ salida (E = ½(ζ−O)²):  ∂E/∂H = −(ζ − O)·(1 − O²)
    d_out = -(z - O) * (1 - O ** 2)
    # δ ocultas: ∂E/∂h_j = d_out · W_j · (1 − V_j²)
    d1 = d_out * 0.5 * (1 - V1 ** 2)
    d2 = d_out * (-0.6) * (1 - V2 ** 2)
    expected = [np.array([[1 * d1, 1 * d2], [-1 * d1, -1 * d2]]), np.array([d1, d2]),
                np.array([[V1 * d_out], [V2 * d_out]]), np.array([d_out])]

    _, grads = model.gradients(x.reshape(1, -1), np.array([[z]]))
    assert np.isclose(model.forward(x.reshape(1, -1))[0, 0], O)
    for g, e in zip(grads, expected):
        assert np.allclose(g, e), (g, e)


def test_hand_calculation_2321():
    """Lo mismo para [2,3,2,1], con las fórmulas de la clase escritas neurona por neurona
    (listas y floats, sin productos matriciales) y comparadas contra el MLP matricial."""
    model = MLP([2, 3, 2, 1], hidden='tanh', output='tanh', seed=7)
    x, z = [1.0, -1.0], 1.0
    W = [w.tolist() for w in model.W]   # W[l][i][j]: peso de la neurona i de la capa l a la j de la l+1
    b = [c.tolist() for c in model.b]

    # Feed-forward: h_j = Σ_i V_i w_ij + b_j ;  V_j = tanh(h_j)
    V, hs = [x], []
    for l in range(3):
        h = [sum(V[l][i] * W[l][i][j] for i in range(len(V[l]))) + b[l][j] for j in range(len(b[l]))]
        hs.append(h)
        V.append([float(np.tanh(v)) for v in h])
    O = V[-1][0]

    # Backprop: δ salida = −(ζ − O)(1 − O²) ;  δ_i(l) = (Σ_j δ_j(l+1) w_ij) (1 − V_i²)
    deltas = [None, None, [-(z - O) * (1 - O ** 2)]]
    for l in (1, 0):
        deltas[l] = [sum(deltas[l + 1][j] * W[l + 1][i][j] for j in range(len(deltas[l + 1])))
                     * (1 - V[l + 1][i] ** 2) for i in range(len(V[l + 1]))]
    # ∂E/∂w_ij = V_i δ_j ;  ∂E/∂b_j = δ_j
    expected = []
    for l in range(3):
        expected.append(np.array([[V[l][i] * deltas[l][j] for j in range(len(deltas[l]))]
                                  for i in range(len(V[l]))]))
        expected.append(np.array(deltas[l]))

    _, grads = model.gradients(np.array([x]), np.array([[z]]))
    assert np.isclose(model.forward(np.array([x]))[0, 0], O)
    for g, e in zip(grads, expected):
        assert np.allclose(g, e), (g, e)


def test_optimizers_reduce_loss():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)   # no lineal
    Y = np.eye(2)[y]
    for name, lr in [('gd', 0.5), ('momentum', 0.1), ('rmsprop', 0.01), ('adam', 0.01)]:
        model = MLP([4, 16, 2], hidden='tanh', output='softmax', loss='cross_entropy', seed=0)
        hist = train(model, X, Y, TrainConfig(epochs=60, batch_size=20, optimizer=name, lr=lr), verbose=False)
        assert hist['loss'][-1] < 0.5 * hist['loss'][0], name
        assert hist['acc'][-1] > 0.9, name


def test_save_load_roundtrip(tmp_path=None):
    import tempfile
    from pathlib import Path
    path = Path(tmp_path or tempfile.mkdtemp()) / 'modelo.npz'
    model = MLP([5, 3, 2], hidden='relu', output='softmax', loss='cross_entropy', seed=3)
    model.save(path, extra={'nota': 'test'})
    loaded, extra = MLP.load(path)
    X = np.random.default_rng(0).normal(size=(4, 5))
    assert np.allclose(model.forward(X), loaded.forward(X)) and extra == {'nota': 'test'}


if __name__ == '__main__':
    tests = [(name, fn) for name, fn in globals().items() if name.startswith('test_')]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f'OK    {name}')
        except AssertionError as e:
            failed += 1
            print(f'FALLA {name}: {e}')
    print(f'\n{len(tests) - failed}/{len(tests)} tests pasaron')
    raise SystemExit(1 if failed else 0)
