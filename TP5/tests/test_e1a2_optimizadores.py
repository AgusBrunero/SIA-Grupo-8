"""Comparación justa de optimizadores (experiments/e1a2_optimizadores.py).

Correr desde TP5/:  .venv/bin/python -m unittest discover -s tests -t . -v
"""
import json
import unittest
from pathlib import Path

import pandas as pd

from experiments.e1a2_optimizadores import SEMILLAS, agregar, mejor_eta

RAIZ = Path(__file__).resolve().parents[1]


def corridas(filas: list[tuple[str, int, bool, int]]) -> pd.DataFrame:
    return pd.DataFrame([{'escalon': e, 'semilla': s, 'logro': l, 'total_pixeles': px, 'max_pixeles': 1,
                          'error_final': 0.0} for e, s, l, px in filas])


class MejorEtaTests(unittest.TestCase):
    def tabla(self, celdas: dict[str, tuple[int, list[int]]]) -> pd.DataFrame:
        filas = []
        for escalon, (logradas, pixeles) in celdas.items():
            for semilla in range(SEMILLAS):
                filas.append((escalon, semilla, semilla < logradas, pixeles[semilla % len(pixeles)]))
        return agregar(corridas(filas), 'ec')

    def test_gana_el_que_mas_semillas_logra_aunque_tenga_mas_pixeles(self) -> None:
        tabla = self.tabla({'gd eta=0.01': (4, [30]), 'gd eta=0.03': (2, [5])})
        mejor = mejor_eta(tabla).iloc[0]
        self.assertEqual(mejor.eta, 0.01)

    def test_a_igualdad_de_logradas_desempata_por_pixeles_y_marca_el_empate(self) -> None:
        tabla = self.tabla({'adam eta=0.001': (0, [10, 12]), 'adam eta=0.003': (0, [11, 13]),
                            'adam eta=0.0003': (0, [40, 42])})
        mejor = mejor_eta(tabla).iloc[0]
        self.assertEqual(mejor.eta, 0.001)
        self.assertEqual(mejor.empata_con, '0.003')      # 1 px de diferencia, menos que el desvío

    def test_conteo_por_celda(self) -> None:
        with self.assertRaises(AssertionError):
            agregar(corridas([('gd eta=0.01', 0, True, 0)]), 'ec')


class ConfigTests(unittest.TestCase):
    def test_mismo_modelo_y_solo_cambian_optimizador_eta_y_costo(self) -> None:
        e1a2 = json.loads((RAIZ / 'config' / 'e1a2.json').read_text(encoding='utf-8'))
        e4 = next(e for e in e1a2['escalones'] if e['id'] == 'E4')
        for costo, perdida in (('ec', 'error_cuadratico'), ('bce', 'bce')):
            config = json.loads((RAIZ / 'config' / f'e1a2_opt_{costo}.json').read_text(encoding='utf-8'))
            self.assertEqual(len(config['semillas']), SEMILLAS)
            self.assertEqual(len(config['escalones']), 15)
            for escalon in config['escalones']:
                self.assertEqual(escalon['perdida'], perdida)
                for clave in ('capas_encoder', 'activacion_oculta', 'activacion_latente', 'activacion_salida', 'tamano_lote'):
                    self.assertEqual(escalon[clave], e4[clave], clave)


if __name__ == '__main__':
    unittest.main()
