"""Los números de la presentación salen de los CSV y el manifiesto guardado está al día.

Correr desde TP5/:  .venv/bin/python -m unittest discover -s tests -t . -v
"""
import json
import re
import unittest
from pathlib import Path

from experiments.presentacion import numeros
from experiments.presentacion_html import auditar

PRESENTACION = Path(__file__).resolve().parents[1] / 'docs' / 'presentacion'


class PresentacionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.numeros = numeros()

    def test_el_manifiesto_guardado_coincide_con_los_csv(self) -> None:
        guardado = json.loads((PRESENTACION / 'manifest.json').read_text(encoding='utf-8'))['numeros']
        self.assertEqual(guardado, self.numeros)

    def test_cada_numero_tiene_texto_y_origen(self) -> None:
        for clave, dato in self.numeros.items():
            with self.subTest(clave=clave):
                self.assertRegex(dato['texto'], r'\d')
                self.assertTrue(dato['origen'])

    def test_las_laminas_solo_usan_claves_del_manifiesto(self) -> None:
        laminas = (PRESENTACION / 'laminas.json').read_text(encoding='utf-8')
        usadas = set(re.findall(r'\{\{(\w+)\}\}', laminas))
        self.assertGreater(len(usadas), 20)
        self.assertEqual(sorted(usadas - set(self.numeros)), [])

    def test_las_laminas_cumplen_las_reglas(self) -> None:
        laminas = json.loads((PRESENTACION / 'laminas.json').read_text(encoding='utf-8'))
        self.assertEqual(auditar(laminas, self.numeros), [])
        self.assertEqual(len(laminas), 23)

    def test_la_auditoria_detecta_las_fallas(self) -> None:
        larga = {'tipo': 'resultado', 'titulo': 'palabra ' * 36, 'lectura': ''}
        a_mano = {'tipo': 'resultado', 'titulo': 'Quedan 12 píxeles', 'lectura': ''}
        codigo = {'tipo': 'resultado', 'titulo': 'Sale de resumen.csv', 'lectura': ''}
        sin_clave = {'tipo': 'resultado', 'titulo': 'Vale {{no_existe}}', 'lectura': ''}
        fallas = auditar([larga, a_mano, codigo, sin_clave], self.numeros)
        self.assertEqual(len(fallas), 4)
        for regla, falla in zip(('R1', 'R10', 'R8', 'manifest'), fallas):
            self.assertIn(regla, falla)


if __name__ == '__main__':
    unittest.main()
