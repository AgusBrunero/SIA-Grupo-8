"""Las muestras elegidas por el grupo existen en los CSV de origen y son nuevas.

Correr desde TP5/:  .venv/bin/python -m unittest discover -s tests -t . -v
"""
import json
import unittest
from pathlib import Path

import pandas as pd

from experiments.elegidas import letra_elegida

RAIZ = Path(__file__).resolve().parents[1]


class ElegidasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((RAIZ / 'config' / 'elegidas.json').read_text(encoding='utf-8'))

    def test_la_letra_es_nueva_y_esta_entre_dos_del_dataset(self) -> None:
        filas = letra_elegida(self.config['letra'])
        self.assertEqual(list(filas.distancia.iloc[[0, 2]]), [0, 0])      # los extremos son letras del dataset
        self.assertGreater(filas.distancia.iloc[1], 0)
        self.assertEqual(len(filas.pixeles.iloc[1]), 35)

    def test_los_relojes_estan_en_candidatas(self) -> None:
        relojes = self.config['relojes']
        tabla = pd.read_csv(RAIZ / relojes['origen'])
        filas = tabla[(tabla['lambda'] == relojes['lambda']) & tabla.numero.isin(relojes['numeros'])]
        self.assertEqual(len(filas), len(relojes['numeros']))
        self.assertTrue((filas.distancia > 0).all())


if __name__ == '__main__':
    unittest.main()
