"""Comparación de arquitecturas con latente no lineal (config/e1a2_arquitecturas_<costo>_<arq>.json).

Correr desde TP5/:  python3 -m unittest discover -s tests -t . -v
"""
import json
import unittest
from pathlib import Path

import pandas as pd

from experiments.e1a2_arquitecturas import elegir, mejor_eta
from experiments.e1a2_escalera import construir
from src.data.font import cargar_font
from src.nn.perdidas import PERDIDAS
from tests.gradcheck import TOLERANCIA, error_relativo, gradiente_numerico

RAIZ = Path(__file__).resolve().parents[1]
COSTOS = {'bce': 'bce', 'ec': 'error_cuadratico'}
ARQUITECTURAS = ('T1', 'T2', 'T3', 'T4', 'L2')


def leer_config(costo: str, arquitectura: str) -> dict:
    ruta = RAIZ / 'config' / f'e1a2_arquitecturas_{costo}_{arquitectura}.json'
    return json.loads(ruta.read_text(encoding='utf-8'))


def escalones(costo: str) -> list[dict]:
    """Los 15 escalones de un costo (5 arquitecturas × 3 η), en orden."""
    return [e for a in ARQUITECTURAS for e in leer_config(costo, a)['escalones']]


class ConfigTests(unittest.TestCase):
    def test_ninguna_capa_lineal(self) -> None:
        for costo in COSTOS:
            for escalon in escalones(costo):
                with self.subTest(costo=costo, escalon=escalon['id']):
                    self.assertIn(escalon['activacion_latente'], ('tanh', 'logistica'))
                    self.assertEqual(escalon['activacion_oculta'], 'tanh')
                    self.assertEqual(escalon['activacion_salida'], 'logistica')

    def test_solo_cambian_arquitectura_latente_y_eta(self) -> None:
        for costo, perdida in COSTOS.items():
            for arquitectura in ARQUITECTURAS:
                config = leer_config(costo, arquitectura)
                self.assertEqual(config['experimento'], f'e1a2_arquitecturas/{costo}_{arquitectura}')
                self.assertEqual(config['semillas'], list(range(10)))
                self.assertEqual(config['epocas'], 10000)
                self.assertEqual([e['optimizador']['eta'] for e in config['escalones']], [0.0003, 0.001, 0.003])
                self.assertTrue(all(e['id'].startswith(arquitectura + ' ') for e in config['escalones']))
            self.assertEqual(len(escalones(costo)), 15)
            for escalon in escalones(costo):
                with self.subTest(costo=costo, escalon=escalon['id']):
                    self.assertEqual(escalon['perdida'], perdida)
                    self.assertEqual(escalon['capas_encoder'][0], 35)
                    self.assertEqual(escalon['capas_encoder'][-1], 2)
                    self.assertEqual(escalon['tamano_lote'], 32)
                    self.assertIsNone(escalon['eta_adaptativo'])
                    self.assertEqual(escalon['optimizador']['nombre'], 'adam')

    def test_los_dos_costos_corren_las_mismas_redes(self) -> None:
        claves = ('id', 'capas_encoder', 'activacion_latente', 'optimizador')
        bce, ec = ([{k: e[k] for k in claves} for e in escalones(c)] for c in COSTOS)
        self.assertEqual(bce, ec)


class GradientCheckTests(unittest.TestCase):
    """CLAUDE.md §4: sin gradient check en verde no se entrena nada nuevo. Una red por
    arquitectura distinta de los configs (los η no cambian el gradiente)."""

    def test_autoencoder_completo(self) -> None:
        X, _ = cargar_font()
        X = X[:6]       # 6 patrones alcanzan para el chequeo y lo mantienen rápido
        config = leer_config('bce', 'T1')
        vistas = set()
        for escalon in escalones('bce'):
            clave = (tuple(escalon['capas_encoder']), escalon['activacion_latente'])
            if clave in vistas:
                continue
            vistas.add(clave)
            ae = construir({**config, 'amplitud_inicial': 0.5}, escalon, semilla=3)
            for perdida in COSTOS.values():
                _, analiticos = ae.gradientes(X, X, perdida)
                funcion = lambda: PERDIDAS[perdida](ae.reconstruir(X), X, 'suma')[0]
                for indice, W in enumerate(ae.params):
                    with self.subTest(red=clave, perdida=perdida, matriz=indice):
                        self.assertLess(error_relativo(analiticos[indice], gradiente_numerico(funcion, W)),
                                        TOLERANCIA)


def mejores(filas: dict[str, tuple]) -> pd.DataFrame:
    """filas: arquitectura → (logradas_bce, px_bce, epoca_mediana, epoca_desvio, pesos, logradas_ec)."""
    datos = []
    for arq, (log_bce, px, epoca, desvio, pesos, log_ec) in filas.items():
        base = {'arquitectura': arq, 'eta': 0.001, 'total_px_desvio': 1.0, 'epoca_media': epoca, 'n_pesos': pesos}
        datos.append({**base, 'costo': 'bce', 'logradas': log_bce, 'total_px_media': px,
                      'epoca_mediana': epoca, 'epoca_desvio': desvio})
        datos.append({**base, 'costo': 'ec', 'logradas': log_ec, 'total_px_media': 30.0,
                      'epoca_mediana': float('nan'), 'epoca_desvio': float('nan')})
    return pd.DataFrame(datos)


def ganadora(tabla: pd.DataFrame) -> str:
    return tabla[tabla.elegida].arquitectura.iloc[0]


class CriterioTests(unittest.TestCase):
    def test_mas_semillas_con_bce_gana_aunque_tenga_mas_pixeles(self) -> None:
        tabla = elegir(mejores({'A': (10, 0.5, 900, 50, 100, 0), 'B': (9, 0.0, 500, 50, 50, 0)}))
        self.assertEqual(ganadora(tabla), 'A')

    def test_a_igualdad_de_semillas_pesa_el_error_cuadratico(self) -> None:
        tabla = elegir(mejores({'A': (10, 0.0, 900, 50, 100, 2), 'B': (10, 0.0, 500, 50, 50, 0)}))
        self.assertEqual(ganadora(tabla), 'A')

    def test_epoca_mas_rapida_gana_si_supera_el_desvio(self) -> None:
        tabla = elegir(mejores({'A': (10, 0.0, 2000, 300, 100, 0), 'B': (10, 0.0, 800, 200, 300, 0)}))
        self.assertEqual(ganadora(tabla), 'B')
        self.assertEqual(tabla.set_index('arquitectura').loc['A', 'queda_afuera_en'], '4 · época en llegar con BCE')

    def test_si_la_epoca_empata_gana_la_de_menos_pesos(self) -> None:
        tabla = elegir(mejores({'A': (10, 0.0, 1000, 300, 100, 0), 'B': (10, 0.0, 800, 300, 300, 0)}))
        self.assertEqual(ganadora(tabla), 'A')

    def test_mejor_eta_desempata_por_epoca(self) -> None:
        todas = pd.DataFrame([{'costo': 'bce', 'arquitectura': 'A', 'eta': eta, 'logradas': 10, 'total_px_media': 0.0,
                               'epoca_mediana': epoca} for eta, epoca in ((0.001, 4000), (0.003, 2000), (0.0003, None))])
        self.assertEqual(mejor_eta(todas).eta.iloc[0], 0.003)


if __name__ == '__main__':
    unittest.main()
