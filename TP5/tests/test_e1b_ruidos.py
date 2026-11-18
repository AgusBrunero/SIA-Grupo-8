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
    def test_los_modelos_son_las_dos_mejores_del_ae(self) -> None:
        """D59: las dos primeras del orden de eleccion.csv; la primera es la misma red de 1a.3/1a.4."""
        config = leer_config()
        orden = pd.read_csv(RAIZ / 'results' / 'e1a2_arquitecturas' / 'eleccion.csv').sort_values('orden')
        self.assertEqual(list(config['modelos']), orden.arquitectura.head(2).tolist())
        ae = json.loads((RAIZ / 'config' / 'e1a3_e1a4.json').read_text(encoding='utf-8'))
        self.assertEqual(config['modelos'][orden.arquitectura.iloc[0]], ae['modelo'])
        for nombre, modelo in config['modelos'].items():
            fila = orden.set_index('arquitectura').loc[nombre]
            self.assertEqual(modelo['optimizador']['eta'], fila.eta_bce)
            self.assertNotEqual(modelo['activacion_latente'], 'identidad')
        for clave in ('epocas', 'reduccion', 'beta', 'amplitud_inicial', 'codificacion'):
            self.assertEqual(config[clave], ae[clave], clave)

    def test_ruidos_niveles_y_semillas(self) -> None:
        config = leer_config()
        self.assertEqual(set(config['ruidos']), {'gaussiano', 'sal_y_pimienta'})
        self.assertGreaterEqual(len(config['semillas']), 5)
        self.assertEqual(config['epocas'] % config['evaluar_cada'], 0)


class CorridaChicaTests(unittest.TestCase):
    """Corre el experimento entero con un config diminuto en una carpeta temporal."""

    def test_prueba_con_el_mismo_ruido_y_conteos(self) -> None:
        config = dict(leer_config(), semillas=[0, 1], epocas=4, evaluar_cada=2, realizaciones_curva=2,
                      realizaciones_final=3, ruidos={'gaussiano': [0.3], 'sal_y_pimienta': [0.1]})
        with tempfile.TemporaryDirectory() as temporal, mock.patch.object(e1b_ruidos, 'SALIDA', Path(temporal)):
            for modelo in config['modelos']:
                for tipo, niveles in config['ruidos'].items():
                    e1b_ruidos.correr_parte(config, modelo, tipo, niveles[0])
            corridas, curvas = e1b_ruidos.leer(config)
            self.assertEqual(len(corridas), 2 * 4)
            self.assertEqual(len(curvas), 2 * 4 * 3)             # épocas 0, 2 y 4
            for (modelo, tipo, nivel), _ in corridas.groupby(['modelo', 'tipo', 'nivel']):
                self.assertIn(nivel, config['ruidos'][tipo])
            resumen = e1b_ruidos.resumir(config, corridas)
            self.assertEqual(resumen.semillas.tolist(), [2] * 4)
            e1b_ruidos.aplicar_estilo()
            media = e1b_ruidos.figura_curvas(config, curvas, Path(temporal) / 'curvas.png')
            texto = e1b_ruidos.texto(config, resumen, media)
            self.assertIn('mismo tipo y nivel', texto)
            e1b_ruidos.figura_barras(config, resumen, Path(temporal) / 'barras.png')
            # Doble conteo: una corrida repetida tiene que hacer fallar la lectura
            ruta = Path(temporal) / list(config['modelos'])[0] / 'gaussiano_0.3' / 'corridas.csv'
            tabla = pd.read_csv(ruta)
            pd.concat([tabla, tabla.iloc[:1]]).to_csv(ruta, index=False)
            with self.assertRaises(AssertionError):
                e1b_ruidos.leer(config)


if __name__ == '__main__':
    unittest.main()
