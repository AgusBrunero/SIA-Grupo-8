"""Ej. 3: llegar a accuracy ≥ 98 % con digits + more_digits (configs/ej3.json).

Uso (desde TP3/):
    python ej3.py barridos                     # ablación de técnicas (se puede cortar y retomar)
    python ej3.py final --extra final_ruido    # modelo final + variante entrenada con ruido
    python analisis_digitos.py                 # gráficos y tablas
"""
from experimentos import main

if __name__ == '__main__':
    main('ej3')
