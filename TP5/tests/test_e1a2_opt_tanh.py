"""Optimizadores sobre la red elegida con latente tanh (config/e1a2_opt_tanh_*.json, D60).

Correr desde TP5/:  python3 -m unittest discover -s tests -t . -v
"""
import json
import unittest
from pathlib import Path

import pandas as pd

from experiments.e1a2_optimizadores import NOMBRES, SEMILLAS

RAIZ = Path(__file__).resolve().parents[1]
CLAVES_MODELO = ('capas_encoder', 'activacion_oculta', 'activacion_latente', 'activacion_salida', 'tamano_lote')


class ConfigTests(unittest.TestCase):
    def test_mismo_modelo_que_la_elegida_y_solo_cambian_optimizador_eta_y_costo(self) -> None:
        eleccion = pd.read_csv(RAIZ / 'results' / 'e1a2_arquitecturas' / 'eleccion.csv')
        elegida = eleccion[eleccion.elegida].iloc[0].arquitectura
        origen = json.loads((RAIZ / 'config' / f'e1a2_arquitecturas_bce_{elegida}.json').read_text())['escalones'][0]
        rutas = sorted((RAIZ / 'config').glob('e1a2_opt_tanh_*.json'))
        self.assertGreaterEqual(len(rutas), 2 * len(NOMBRES))
        for ruta in rutas:
            config = json.loads(ruta.read_text(encoding='utf-8'))
            costo = 'bce' if '_bce_' in ruta.name else 'ec'
            self.assertEqual(config['semillas'], list(range(SEMILLAS)))
            for escalon in config['escalones']:
                with self.subTest(config=ruta.name, escalon=escalon['id']):
                    for clave in CLAVES_MODELO:
                        self.assertEqual(escalon[clave], origen[clave], clave)
                    self.assertEqual(escalon['perdida'], 'bce' if costo == 'bce' else 'error_cuadratico')
                    optimizador = escalon['id'].split(' eta=')[0]
                    self.assertIn(optimizador, NOMBRES)
                    self.assertEqual(float(escalon['id'].split('eta=')[1]), escalon['optimizador']['eta'])
                    self.assertEqual(escalon['eta_adaptativo'] is not None, optimizador == 'eta_adaptativo')


if __name__ == '__main__':
    unittest.main()
