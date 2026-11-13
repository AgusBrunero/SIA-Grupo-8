"""Estilo de las figuras y escritura de docs/RESULTADOS.md, compartidos por todos los experimentos.

CLAUDE.md §5: figuras sin título interno, ejes en español y con unidades, leyenda legible
y el mismo color para el mismo modelo en todo el TP.
"""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parents[1]
RESULTADOS_MD = RAIZ / 'docs' / 'RESULTADOS.md'

# Un color por modelo, fijo en todo el TP
COLOR = {'AE': '#2a78d6', 'DAE': '#eb6834', 'VAE': '#1baf7a', 'PCA': '#eda100'}
TINTA, TINTA_SECUNDARIA, GRILLA, EJE = '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7'
# Rampa secuencial de un solo tono (el del AE), de claro a oscuro
RAMPA_AE = ['#f4f8fe', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']


def aplicar_estilo() -> None:
    plt.rcParams.update({
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 11,
        'text.color': TINTA, 'axes.labelcolor': TINTA, 'xtick.color': TINTA_SECUNDARIA,
        'ytick.color': TINTA_SECUNDARIA, 'axes.edgecolor': EJE, 'axes.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRILLA, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
        'legend.frameon': False, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
    })


def escribir_seccion(clave: str, texto: str, ruta: Path = RESULTADOS_MD) -> None:
    """Reemplaza (o agrega) en RESULTADOS.md el bloque de un experimento, entre marcadores.

    Cada experimento genera su bloque desde sus CSV: el texto con números nunca se edita a mano.
    """
    inicio, fin = f'<!-- {clave}:inicio -->', f'<!-- {clave}:fin -->'
    bloque = f'{inicio}\n{texto.strip()}\n{fin}'
    encabezado = ('# TP5 · Resultados\n\nObservaciones factuales generadas por los scripts de '
                  '`experiments/` a partir de los CSV de `results/`. No editar a mano entre los '
                  'marcadores: se regenera. Las conclusiones van en `CONCLUSIONES.md` y las escribe el grupo.\n')
    contenido = ruta.read_text(encoding='utf-8') if ruta.exists() else encabezado
    if inicio in contenido:
        antes, resto = contenido.split(inicio, 1)
        contenido = antes + bloque + resto.split(fin, 1)[1]
    else:
        contenido = contenido.rstrip('\n') + '\n\n' + bloque + '\n'
    ruta.write_text(contenido, encoding='utf-8')
