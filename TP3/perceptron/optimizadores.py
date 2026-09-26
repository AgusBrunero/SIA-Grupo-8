"""Optimizadores (clase 12.1). Todos tienen la misma interfaz: step(params, grads).

Actualizan los parámetros in-place, así la comparación entre optimizadores es cambiar un
string en la configuración.
"""
import numpy as np


class GD:
    """Descenso por gradiente: Δw = −η ∂E/∂w."""

    def __init__(self, lr: float, weight_decay: float = 0.0):
        self.lr = lr
        # L2: E_reg = E + ½λ‖w‖²  →  ∂E_reg/∂w = ∂E/∂w + λw
        self.weight_decay = weight_decay

    def _regularize(self, params, grads):
        if self.weight_decay == 0:
            return grads
        # Solo se regularizan los pesos (índices pares), no los bias
        return [g + self.weight_decay * p if i % 2 == 0 else g
                for i, (p, g) in enumerate(zip(params, grads))]

    def step(self, params, grads):
        for p, g in zip(params, self._regularize(params, grads)):
            p -= self.lr * g


class Momentum(GD):
    """Δw(t+1) = −η ∂E/∂w + α Δw(t)."""

    def __init__(self, lr: float, alpha: float = 0.9, weight_decay: float = 0.0):
        super().__init__(lr, weight_decay)
        self.alpha = alpha
        self.velocity = None

    def step(self, params, grads):
        grads = self._regularize(params, grads)
        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p in params]
        for p, g, v in zip(params, grads, self.velocity):
            v *= self.alpha
            v -= self.lr * g
            p += v


class RMSProp(GD):
    """Divide η por un promedio móvil de g²: cada peso tiene su propia escala de paso."""

    def __init__(self, lr: float = 1e-3, rho: float = 0.9, eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(lr, weight_decay)
        self.rho, self.eps = rho, eps
        self.sq = None

    def step(self, params, grads):
        grads = self._regularize(params, grads)
        if self.sq is None:
            self.sq = [np.zeros_like(p) for p in params]
        for p, g, s in zip(params, grads, self.sq):
            s *= self.rho
            s += (1 - self.rho) * g * g
            p -= self.lr * g / (np.sqrt(s) + self.eps)


class Adam(GD):
    """Momentum (m) + RMSProp (v) con corrección de sesgo por arrancar m y v en cero."""

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(lr, weight_decay)
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m = self.v = None
        self.t = 0

    def step(self, params, grads):
        grads = self._regularize(params, grads)
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        c1 = 1 - self.beta1 ** self.t
        c2 = 1 - self.beta2 ** self.t
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.beta1
            m += (1 - self.beta1) * g
            v *= self.beta2
            v += (1 - self.beta2) * g * g
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)


class AdaptiveLR:
    """η adaptativo (clase 12.1), se aplica por época sobre cualquier optimizador:
    si el error baja K épocas seguidas, η += a; si sube, η −= b·η."""

    def __init__(self, optimizer: GD, a: float, b: float, k: int = 3):
        self.optimizer, self.a, self.b, self.k = optimizer, a, b, k
        self.prev, self.streak = None, 0

    def end_epoch(self, error: float):
        if self.prev is not None:
            if error < self.prev:
                self.streak += 1
                if self.streak >= self.k:
                    self.optimizer.lr += self.a
                    self.streak = 0
            elif error > self.prev:
                self.optimizer.lr -= self.b * self.optimizer.lr
                self.streak = 0
        self.prev = error


def build(name: str, lr: float, weight_decay: float = 0.0, **kwargs) -> GD:
    classes = {'gd': GD, 'momentum': Momentum, 'rmsprop': RMSProp, 'adam': Adam}
    return classes[name](lr=lr, weight_decay=weight_decay, **kwargs)
