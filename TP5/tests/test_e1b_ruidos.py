"""DAE rehecho (experiments/e1b_ruidos.py): modelo, protocolo y conteos.

Correr desde TP5/:  python3 -m unittest discover -s tests -t . -v
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from experiments import e1b_ruidos

RAIZ = Path(__file__).resolve().parents[1]


def leer_config() -> dict:
    return json.loads((RAIZ / 'config' / 'e1b_ruidos.json').read_text(encoding='utf-8'))


class ConfigTests(unittest.TestCase):
    def test_el_modelo_es_el_elegido_para_el_ae(self) -> None:
        """D57: el mismo modelo que 1a.3/1a.4, que es el elegido en eleccion.csv."""
        config = leer_config()
        ae = json.loads((RAIZ / 'config' / 'e1a3_e1a4.json').read_text(encoding='utf-8'))
        self.assertEqual(config['modelo'], ae['modelo'])
        self.assertEqual(config['modelo']['activacion_latente'], 'tanh')
        for clave in ('epocas', 'reduccion', 'beta', 'amplitud_inicial', 'codificacion'):
            self.assertEqual(config[clave], ae[clave], clave)

    def test_ruidos_niveles_y_semillas(self) -> None:
        config = leer_config()
        self.assertEqual(set(config['ruidos']), {'gaussiano', 'sal_y_pimienta'})
        self.assertGreaterEqual(len(config['semillas']), 5)
        self.assertEqual(config['epocas'] % config['evaluar_cada'], 0)
        for tipo, nivel in config['ejemplos']['niveles'].items():
            self.assertIn(nivel, config['ruidos'][tipo])


class CorridaChicaTests(unittest.TestCase):
    """Corre el experimento entero con un config diminuto en una carpeta temporal."""

    def test_prueba_con_el_mismo_ruido_y_conteos(self) -> None:
        config = dict(leer_config(), semillas=[0, 1], epocas=4, evaluar_cada=2, realizaciones_curva=2,
                      realizaciones_final=3, ruidos={'gaussiano': [0.3], 'sal_y_pimienta': [0.1]},
                      ejemplos={'semilla': 0, 'letras': ['a', 'o'], 'niveles': {'gaussiano': 0.3, 'sal_y_pimienta': 0.1}})
        with tempfile.TemporaryDirectory() as temporal, mock.patch.object(e1b_ruidos, 'SALIDA', Path(temporal)):
            for tipo, niveles in config['ruidos'].items():
                e1b_ruidos.correr_parte(config, tipo, niveles[0])
            corridas, curvas = e1b_ruidos.leer(config)
            self.assertEqual(len(corridas), 4)
            self.assertEqual(len(curvas), 4 * 3)                 # épocas 0, 2 y 4
            for (tipo, nivel), _ in corridas.groupby(['tipo', 'nivel']):
                self.assertIn(nivel, config['ruidos'][tipo])
            resumen = e1b_ruidos.resumir(config, corridas)
            self.assertEqual(resumen.semillas.tolist(), [2, 2])
            media = curvas.groupby(['tipo', 'nivel', 'epoca']).agg(mal_m=('mal_media', 'mean')).reset_index()
            texto = e1b_ruidos.texto(config, resumen, media)
            self.assertIn('mismo tipo y nivel', texto)
            e1b_ruidos.aplicar_estilo()
            e1b_ruidos.figura_barras(config, resumen, Path(temporal) / 'barras.png')
            e1b_ruidos.figura_curvas(config, curvas, Path(temporal) / 'curvas.png')
            e1b_ruidos.figura_ejemplos(config, Path(temporal) / 'ejemplos.png')
            self.assertEqual(len(pd.read_csv(Path(temporal) / 'gaussiano_0.3' / 'ejemplos.csv')), 6)
            # Doble conteo: una corrida repetida tiene que hacer fallar la lectura
            ruta = Path(temporal) / 'gaussiano_0.3' / 'corridas.csv'
            tabla = pd.read_csv(ruta)
            pd.concat([tabla, tabla.iloc[:1]]).to_csv(ruta, index=False)
            with self.assertRaises(AssertionError):
                e1b_ruidos.leer(config)


if __name__ == '__main__':
    unittest.main()
