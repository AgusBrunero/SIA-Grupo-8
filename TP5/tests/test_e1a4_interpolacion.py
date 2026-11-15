"""Interpolación fina o→e (experiments/e1a4_interpolacion.py).

Lee el resultado guardado: los extremos del camino son la "o" y la "e" del dataset, y el paso
de la letra elegida (D45) aparece con el mismo z y la misma distancia.
Correr desde TP5/:  .venv/bin/python -m unittest discover -s tests -t . -v
"""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]


class InterpolacionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((RAIZ / 'config' / 'e1a4_interpolacion.json').read_text(encoding='utf-8'))
        cls.pasos = pd.read_csv(RAIZ / 'results' / 'e1a4_interpolacion' / 'pasos.csv')

    def test_cantidad_de_pasos(self) -> None:
        self.assertEqual(len(self.pasos), self.config['pasos'])

    def test_los_extremos_son_las_letras_del_par(self) -> None:
        primero, ultimo = self.pasos.iloc[0], self.pasos.iloc[-1]
        self.assertEqual((primero.vecino, primero.distancia), (self.config['par'][0], 0))
        self.assertEqual((ultimo.vecino, ultimo.distancia), (self.config['par'][1], 0))

    def test_contiene_la_letra_elegida(self) -> None:
        elegida = pd.read_csv(RAIZ / 'results' / 'elegidas' / 'elegidas.csv').set_index('item').loc['1a.4']
        cerca = np.isclose(self.pasos.z1, elegida.z1) & np.isclose(self.pasos.z2, elegida.z2)
        self.assertEqual(cerca.sum(), 1)
        self.assertEqual(self.pasos[cerca].distancia.iloc[0], elegida.distancia)


if __name__ == '__main__':
    unittest.main()
