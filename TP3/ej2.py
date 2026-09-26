"""Ej. 2: estudio del MLP sobre digits.csv (configs/ej2.json).

Uso (desde TP3/):
    python ej2.py barridos     # corre las variantes (se pueden cortar y retomar)
    python ej2.py final        # entrena la config elegida y la evalúa una vez en digits_test
    python analisis_digitos.py # gráficos y tablas
"""
from experimentos import main

if __name__ == '__main__':
    main('ej2')
