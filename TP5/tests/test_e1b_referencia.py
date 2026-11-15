"""Línea de referencia del DAE: decodificador "letra más cercana" (experiments/e1b_referencia.py).

Correr desde TP5/:  .venv/bin/python -m unittest discover -s tests -t . -v
"""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.e1b_referencia import correr, medir_entradas
from src.data.font import cargar_font

RAIZ = Path(__file__).resolve().parents[1]


class MedirEntradasTests(unittest.TestCase):
    def test_empate_vale_uno_sobre_k(self) -> None:
        X = np.array([[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]], dtype=float)
        # Letra 0 con un píxel invertido en la posición 0: queda a 1 de la letra 0 y a 1 de la 1
        ruidosas = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 1]], dtype=float)
        acierto, otra = medir_entradas(ruidosas, X)
        np.testing.assert_allclose(acierto, [0.5, 1.0, 1.0])
        np.testing.assert_array_equal(otra, [False, False, False])

    def test_estrictamente_mas_cerca_de_otra(self) -> None:
        X = np.array([[0, 0, 0, 0], [1, 1, 1, 0]], dtype=float)
        ruidosas = np.array([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=float)   # la 0 quedó a 2 de sí y a 1 de la otra
        acierto, otra = medir_entradas(ruidosas, X)
        np.testing.assert_allclose(acierto, [0.0, 1.0])
        np.testing.assert_array_equal(otra, [True, False])


class MismoRuidoQueElDAETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((RAIZ / 'config' / 'e1b.json').read_text(encoding='utf-8'))
        cls.X, _ = cargar_font()
        cls.semillas = correr(cls.config, cls.X)

    def test_sin_ruido_recupera_todo(self) -> None:
        cero = self.semillas[self.semillas.p_prueba == 0.0]
        self.assertTrue((cero.letra_cercana_exacta == 1.0).all())
        self.assertTrue((cero.mas_cerca_de_otra == 0.0).all())

    def test_los_pixeles_alterados_coinciden_con_los_del_dae(self) -> None:
        """Si el ruido es el mismo, la cantidad de píxeles alterados por (semilla, p) coincide
        con la que registró e1b para cualquier arquitectura y nivel de entrenamiento."""
        celdas = pd.read_csv(RAIZ / 'results' / 'e1b' / 'celdas.csv')
        dae = celdas[(celdas.arquitectura == 'B') & (celdas.p_entrenamiento == 0.1)].set_index(['semilla', 'p_prueba'])
        for fila in self.semillas.itertuples():
            self.assertAlmostEqual(fila.alterados_media, dae.loc[(fila.semilla, fila.p_prueba), 'alterados_media'], places=12)


if __name__ == '__main__':
    unittest.main()
