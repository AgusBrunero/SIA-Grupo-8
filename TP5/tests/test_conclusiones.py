"""Texto contra datos (CLAUDE.md §4): todo número con decimales de docs/CONCLUSIONES.md
tiene que estar, tal cual, en docs/RESULTADOS.md (que se genera desde los CSV).

Correr desde TP5/:  .venv/bin/python -m unittest discover -s tests -t . -v
"""
import re
import unittest
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / 'docs'


class ConclusionesTests(unittest.TestCase):
    def test_los_numeros_salen_de_resultados(self) -> None:
        conclusiones = (DOCS / 'CONCLUSIONES.md').read_text(encoding='utf-8')
        resultados = (DOCS / 'RESULTADOS.md').read_text(encoding='utf-8')
        en_resultados = set(re.findall(r'\d+,\d+', resultados))
        numeros = set(re.findall(r'\d+,\d+', conclusiones))
        self.assertGreater(len(numeros), 40)
        self.assertEqual(sorted(numeros - en_resultados), [])

    def test_los_conteos_salen_de_resultados(self) -> None:
        conclusiones = (DOCS / 'CONCLUSIONES.md').read_text(encoding='utf-8')
        resultados = (DOCS / 'RESULTADOS.md').read_text(encoding='utf-8')
        for conteo in set(re.findall(r'\d+ de \d+', conclusiones)):
            with self.subTest(conteo=conteo):
                self.assertIn(conteo, resultados)


if __name__ == '__main__':
    unittest.main()
