"""Config y corrida de la escalera e1a2 (consigna 1a.2).

Correr desde TP5/:  .venv/bin/python -m unittest discover -s tests -t . -v
"""
import json
import unittest
from pathlib import Path

from experiments.e1a2_escalera import CLAVES_COMPARABLES, cargar_datos, correr

RAIZ = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((RAIZ / 'config' / 'e1a2.json').read_text(encoding='utf-8'))
        cls.escalones = {e['id']: e for e in cls.config['escalones']}

    def test_cantidad_de_escalones_y_semillas(self) -> None:
        self.assertLessEqual(len(self.escalones), 10)
        self.assertEqual(len(self.escalones), len(self.config['escalones']))     # ids únicos
        self.assertGreaterEqual(len(set(self.config['semillas'])), 5)

    def test_cada_escalon_cambia_una_sola_cosa(self) -> None:
        for escalon in self.config['escalones']:
            self.assertTrue(escalon['sosten'], f"{escalon['id']} sin referencia a la cátedra")
            if escalon['respecto_de'] is None:
                continue
            base = self.escalones[escalon['respecto_de']]
            distintas = [clave for clave in CLAVES_COMPARABLES if escalon[clave] != base[clave]]
            with self.subTest(escalon=escalon['id']):
                self.assertEqual(len(distintas), 1, f"{escalon['id']} difiere de {base['id']} en {distintas}")

    def test_latente_de_dos_dimensiones(self) -> None:
        for escalon in self.config['escalones']:
            self.assertEqual(escalon['capas_encoder'][0], 35)
            self.assertEqual(escalon['capas_encoder'][-1], 2)


class CorridaTests(unittest.TestCase):
    def test_una_corrida_corta_por_escalon(self) -> None:
        """Cada escalón del config corre de punta a punta (pocas épocas) y devuelve sus tres tablas."""
        config = json.loads((RAIZ / 'config' / 'e1a2.json').read_text(encoding='utf-8'))
        config['epocas'], config['curva_cada'] = 4, 2
        X, etiquetas = cargar_datos(config['codificacion'])
        for escalon in config['escalones']:
            if escalon['optimizador']['nombre'] == 'powell':
                escalon = dict(escalon, optimizador={'nombre': 'powell', 'max_evaluaciones': 400})
            with self.subTest(escalon=escalon['id']):
                fila, curva, patrones = correr(config, escalon, 0, X, etiquetas)
                self.assertEqual(fila['escalon'], escalon['id'])
                self.assertEqual(fila['logro'], fila['max_pixeles'] <= 1)
                self.assertEqual(len(patrones), 32)
                self.assertEqual(int(patrones['pixeles_mal'].sum()), fila['total_pixeles'])
                self.assertEqual(int(patrones['pixeles_mal'].max()), fila['max_pixeles'])
                if escalon['optimizador']['nombre'] == 'powell':
                    self.assertGreater(fila['evaluaciones'], 0)
                else:
                    self.assertEqual(list(curva['epoca']), [1, 2, 4])


if __name__ == '__main__':
    unittest.main()
