"""Memoria asociativa bipolar: Hebb y recuperación asincrónica, sin umbrales."""
from dataclasses import dataclass

import numpy as np


@dataclass
class RecallResult:
    state: np.ndarray
    states: np.ndarray
    energies: np.ndarray
    sweeps: int
    converged: bool


class Hopfield:
    """W = Σ ξ ξᵀ / N, diagonal nula y empates que conservan el estado.

    Cada barrido actualiza todas las neuronas en una permutación aleatoria.
    Se guarda energía después de cada actualización, incluidos los pasos sin cambio.
    """

    def fit(self, patterns):
        patterns = np.asarray(patterns, dtype=float)
        if patterns.ndim != 2 or not all(patterns.shape):
            raise ValueError("patterns debe ser una matriz no vacía (patrones, neuronas)")
        if not np.isin(patterns, [-1, 1]).all():
            raise ValueError("Los patrones deben contener únicamente −1 y +1")
        self.patterns = patterns.copy()
        self.weights = patterns.T @ patterns / patterns.shape[1]
        np.fill_diagonal(self.weights, 0)
        return self

    def _state(self, state):
        if not hasattr(self, "weights"):
            raise RuntimeError("Primero se debe entrenar con fit(patterns)")
        state = np.asarray(state, dtype=float)
        if state.shape != (self.weights.shape[0],) or not np.isin(state, [-1, 1]).all():
            raise ValueError("El estado debe ser un vector bipolar de tamaño N")
        return state.copy()

    def energy(self, state):
        state = self._state(state)
        return float(-0.5 * state @ self.weights @ state)

    def is_fixed_point(self, state):
        state = self._state(state)
        # En campo cero se conserva el bit: también es un estado estable.
        return bool(np.all(state * (self.weights @ state) >= -1e-12))

    def recall(self, state, max_sweeps=100, seed=0):
        if not isinstance(max_sweeps, (int, np.integer)) or max_sweeps < 1:
            raise ValueError("max_sweeps debe ser un entero positivo")
        state = self._state(state)
        rng = np.random.default_rng(seed)
        states = [state.copy()]
        energies = [self.energy(state)]
        for sweep in range(1, max_sweeps + 1):
            changed = False
            for i in rng.permutation(len(state)):
                field = self.weights[i] @ state
                # Evita que un cero numérico de Hebb sea tratado como un campo real.
                new = 1 if field > 1e-12 else -1 if field < -1e-12 else state[i]
                changed |= new != state[i]
                # ΔE = −Δs_i h_i para W simétrica con diagonal cero.
                energies.append(energies[-1] - (new - state[i]) * field)
                state[i] = new
            states.append(state.copy())
            if not changed:
                return RecallResult(state.copy(), np.array(states), np.array(energies),
                                    sweep, True)
        return RecallResult(state.copy(), np.array(states), np.array(energies),
                            max_sweeps, self.is_fixed_point(state))
