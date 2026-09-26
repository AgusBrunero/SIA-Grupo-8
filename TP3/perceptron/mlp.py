"""Perceptrón multicapa con backpropagation matricial (numpy).

Convenciones (clase 11):
- Cada capa l tiene W[l] de forma (n_entrada, n_salida) y b[l] de forma (n_salida,):
  el bias se agrega capa a capa (equivale a x0 = 1 con su peso w0).
- h = X·W + b es la excitación y O = θ(h) la salida de la capa.
- Las derivadas reciben h (no θ(h)), igual que en perceptron.py.
- Pérdida:
    'mse'           E = ½ Σ (ζ − O)² / p        (la de la cátedra, con logística/tanh a la salida)
    'cross_entropy' E = −Σ ζ·log(O) / p        (solo con softmax a la salida; extra)
  Se divide por p (tamaño del batch) para que η no dependa del tamaño del batch.
"""
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np


# ---------- activaciones: θ(h) y θ'(h) con β ----------

@dataclass(frozen=True)
class Activation:
    name: str
    beta: float = 1.0

    def __call__(self, h):
        b = self.beta
        if self.name == 'identity':
            return h
        if self.name == 'tanh':
            return np.tanh(b * h)
        if self.name == 'logistic':
            # 1 / (1 + e^{−2βh}) como en la cátedra; clip para evitar overflow en exp
            return 1.0 / (1.0 + np.exp(-np.clip(2 * b * h, -60, 60)))
        if self.name == 'relu':
            return np.maximum(0, h)
        if self.name == 'softmax':
            z = h - h.max(axis=1, keepdims=True)
            e = np.exp(z)
            return e / e.sum(axis=1, keepdims=True)
        raise ValueError(f'activación desconocida: {self.name}')

    def derivative(self, h):
        b = self.beta
        if self.name == 'identity':
            return np.ones_like(h)
        if self.name == 'tanh':
            return b * (1 - np.tanh(b * h) ** 2)
        if self.name == 'logistic':
            o = self(h)
            return 2 * b * o * (1 - o)
        if self.name == 'relu':
            return (h > 0).astype(h.dtype)
        raise ValueError(f'{self.name} no tiene derivada elemento a elemento')


class MLP:
    def __init__(self, layers: list[int], hidden: str = 'tanh', output: str = 'logistic',
                 beta: float = 1.0, loss: str = 'mse', seed: int | None = None,
                 init: str = 'xavier', dtype=np.float64):
        if output == 'softmax' and loss != 'cross_entropy':
            raise ValueError('softmax se usa con cross_entropy')
        self.layers = list(layers)
        self.hidden = Activation(hidden, beta)
        self.output = Activation(output, beta)
        self.loss_name = loss
        self.init = init
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for n_in, n_out in zip(layers[:-1], layers[1:]):
            # Pesos chicos y aleatorios: si arrancan todos iguales (p. ej. en 0) las neuronas
            # de una capa reciben el mismo gradiente y nunca se diferencian (simetría).
            if init == 'xavier':
                limit = np.sqrt(6 / (n_in + n_out))
            else:  # 'small': uniforme chico, la versión de la clase
                limit = 0.1
            self.W.append(rng.uniform(-limit, limit, (n_in, n_out)).astype(dtype))
            self.b.append(np.zeros(n_out, dtype=dtype))

    # ---------- parámetros como lista plana (para optimizadores y guardado) ----------

    @property
    def params(self) -> list[np.ndarray]:
        return [p for pair in zip(self.W, self.b) for p in pair]

    def n_params(self) -> int:
        return sum(p.size for p in self.params)

    def activation_of(self, layer: int) -> Activation:
        return self.output if layer == len(self.W) - 1 else self.hidden

    # ---------- feed-forward ----------

    def forward(self, X, keep: bool = False, dropout: float = 0.0, rng=None):
        """Devuelve la salida. Con keep=True guarda h y O de cada capa para backprop."""
        O = X
        hs, outs, masks = [], [X], []
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            h = O @ W + b
            O = self.activation_of(l)(h)
            last = l == len(self.W) - 1
            mask = None
            if dropout > 0 and not last:
                # Dropout invertido: se apagan neuronas ocultas y se reescala el resto
                mask = (rng.random(O.shape) >= dropout) / (1 - dropout)
                O = O * mask
            if keep:
                hs.append(h)
                outs.append(O)
                masks.append(mask)
        if keep:
            self._cache = (hs, outs, masks)
        return O

    predict_proba = forward

    def predict(self, X):
        return self.forward(X).argmax(axis=1)

    def hidden_activations(self, X) -> list[np.ndarray]:
        """Salida de cada capa oculta (para visualizar qué se activa)."""
        O, acts = X, []
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            O = self.activation_of(l)(O @ W + b)
            if l < len(self.W) - 1:
                acts.append(O)
        return acts

    # ---------- pérdida ----------

    def loss(self, O, Y) -> float:
        p = len(Y)
        if self.loss_name == 'mse':
            return float(0.5 * np.sum((Y - O) ** 2) / p)
        return float(-np.sum(Y * np.log(np.clip(O, 1e-12, 1))) / p)

    # ---------- backprop ----------

    def backward(self, Y) -> list[np.ndarray]:
        """Gradientes ∂E/∂W y ∂E/∂b (mismo orden que params) usando el último forward(keep=True)."""
        hs, outs, masks = self._cache
        p = len(Y)
        O = outs[-1]
        # δ de la capa de salida
        if self.loss_name == 'cross_entropy':
            # softmax + cross-entropy: ∂E/∂h = O − ζ
            delta = (O - Y) / p
        else:
            delta = -(Y - O) * self.output.derivative(hs[-1]) / p

        grads = [None] * (2 * len(self.W))
        for l in range(len(self.W) - 1, -1, -1):
            grads[2 * l] = outs[l].T @ delta          # ∂E/∂W[l] = O[l−1]ᵀ δ[l]
            grads[2 * l + 1] = delta.sum(axis=0)      # ∂E/∂b[l]
            if l > 0:
                # Retropropagar: δ[l−1] = (δ[l] W[l]ᵀ) ⊙ θ'(h[l−1]) (⊙ máscara de dropout)
                delta = (delta @ self.W[l].T) * self.hidden.derivative(hs[l - 1])
                if masks[l - 1] is not None:
                    delta = delta * masks[l - 1]
        return grads

    def gradients(self, X, Y, dropout: float = 0.0, rng=None):
        O = self.forward(X, keep=True, dropout=dropout, rng=rng)
        return self.loss(O, Y), self.backward(Y)

    def input_gradient(self, X, target: int) -> np.ndarray:
        """∂h_target/∂x: cuánto cambia la excitación de la neurona de salida `target` si se
        mueve cada píxel (saliency). Se usa h y no O para que la saturación de la salida no
        aplaste el gradiente. Es el mismo backprop, llevado un paso más hasta la entrada."""
        self.forward(X, keep=True)
        hs, _, _ = self._cache
        delta = np.zeros_like(hs[-1])
        delta[:, target] = 1.0
        for l in range(len(self.W) - 1, 0, -1):
            delta = (delta @ self.W[l].T) * self.hidden.derivative(hs[l - 1])
        return delta @ self.W[0].T

    # ---------- guardar / cargar ----------

    def config(self) -> dict:
        return {'layers': self.layers, 'hidden': self.hidden.name, 'output': self.output.name,
                'beta': self.hidden.beta, 'loss': self.loss_name, 'init': self.init, 'seed': self.seed}

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f'W{i}': W for i, W in enumerate(self.W)} | {f'b{i}': b for i, b in enumerate(self.b)}
        meta = json.dumps({'config': self.config(), 'extra': extra or {}})
        np.savez_compressed(path, meta=np.array(meta), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> tuple['MLP', dict]:
        data = np.load(path)
        meta = json.loads(str(data['meta']))
        cfg = meta['config']
        model = cls(cfg['layers'], cfg['hidden'], cfg['output'], cfg['beta'], cfg['loss'],
                    cfg['seed'], cfg['init'])
        model.W = [data[f'W{i}'] for i in range(len(model.W))]
        model.b = [data[f'b{i}'] for i in range(len(model.b))]
        return model, meta['extra']
