"""Consigna 1a.2 rehecha · Optimizadores sobre la red elegida con latente tanh (D60).

Pregunta: con la red elegida en e1a2_arquitecturas (la de results/e1a2_arquitecturas/eleccion.csv,
ocultas y latente tanh, salida logística; hoy T8, 35–30–20–2–20–30–35), dándole a cada optimizador su mejor η entre tres (hasta cinco si quedaba en el borde), ¿cuál aprende las
32 letras con a lo sumo 1 píxel incorrecto por letra, con entropía cruzada y con error
cuadrático? Es la comparación de D49 (e1a2_optimizadores.py), que se hizo con latente lineal,
repetida sobre la red nueva.

No entrena: lee los corridas.csv que escribe experiments.e1a2_escalera con
config/e1a2_opt_tanh_<costo>_<optimizador>.json y, donde el mejor η de los tres quedó en el
borde, con config/e1a2_opt_tanh_borde_<costo>_<optimizador>.json (un η más) y, si seguía en el borde,
config/e1a2_opt_tanh_borde2_<costo>_<optimizador>.json (otro más). Mejor η: el criterio de D49 (más semillas que
llegan; luego menos píxeles; empata si la diferencia no supera el desvío), reusado de
experiments.e1a2_optimizadores.
Sostén: Clase 11 diap. 52–54 (descenso por gradiente); Clase 12.1 diap. 5–13 (momentum,
η adaptativo, RMSProp, Adam); Autoencoders diap. 51 (entropía cruzada).

Uso, desde TP5/:
    python3 -m experiments.e1a2_optimizadores_tanh

Salida en results/e1a2_opt_tanh/: todas.csv, mejor_eta.csv, fig_optimizador_costo.png y el
bloque e1a2_opt_tanh de docs/RESULTADOS.md.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.comun import RAIZ, aplicar_estilo, escribir_seccion
from experiments.e1a2_optimizadores import COLOR_COSTO, COSTOS, NOMBRES, SEMILLAS, agregar, mejor_eta

SALIDA = RAIZ / 'results' / 'e1a2_opt_tanh'


def red() -> str:
    """La red de los configs ('35–30–20–2–20–30–35'), para no escribirla a mano en textos y figuras."""
    capas = json.loads((RAIZ / 'config' / 'e1a2_opt_tanh_bce_adam.json').read_text())['escalones'][0]['capas_encoder']
    return '–'.join(str(c) for c in capas + capas[-2::-1])


def leer() -> pd.DataFrame:
    tablas = []
    for costo in COSTOS:
        partes = [pd.read_csv(SALIDA / f'{costo}_{opt}' / 'corridas.csv') for opt in NOMBRES]
        # Un η más allá del borde donde el mejor de los tres quedó en el borde (esquema de D49)
        # (y un segundo paso donde con el primero seguía en el borde: borde2)
        partes += [pd.read_csv(SALIDA / f'{prefijo}_{costo}_{opt}' / 'corridas.csv') for opt in NOMBRES
                   for prefijo in ('borde', 'borde2') if (SALIDA / f'{prefijo}_{costo}_{opt}' / 'corridas.csv').exists()]
        tablas.append(agregar(pd.concat(partes, ignore_index=True), costo))
    tabla = pd.concat(tablas, ignore_index=True)
    # Conteo: 3 η por optimizador y costo, más uno o dos si se probaron η más allá del borde; cada η con
    # SEMILLAS corridas y sin repetir (lo verifica agregar)
    assert tabla.groupby(['costo', 'optimizador']).size().between(3, 5).all()
    return tabla


def figura(mejores: pd.DataFrame, ruta) -> None:
    """Dos paneles con el mejor η de cada optimizador: semillas que llegan y píxeles mal."""
    optimizadores = list(NOMBRES)
    x = np.arange(len(optimizadores))
    ancho = 0.38
    fig, ejes = plt.subplots(1, 2, figsize=(13, 4.6))
    for k, costo in enumerate(COSTOS):
        m = mejores[mejores.costo == costo].set_index('optimizador').loc[optimizadores]
        dx = (k - 0.5) * ancho
        barras = ejes[0].bar(x + dx, m.logradas, ancho, color=COLOR_COSTO[costo], label=COSTOS[costo])
        for barra, eta in zip(barras, m.eta):
            ejes[0].text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.15, f'η = {eta:g}'.replace('.', ','),
                         ha='center', va='bottom', fontsize=7.5, rotation=90)
        abajo = np.minimum(m.total_px_desvio, m.total_px_media)
        ejes[1].bar(x + dx, m.total_px_media, ancho, yerr=[abajo, m.total_px_desvio], capsize=3, color=COLOR_COSTO[costo])
    for ax in ejes:
        ax.set_xticks(x, [NOMBRES[o] for o in optimizadores], fontsize=9)
    ejes[0].set_ylim(0, SEMILLAS + 2.5)
    ejes[0].set_ylabel(f'Semillas que aprenden las 32 letras\ncon ≤ 1 píxel incorrecto (de {SEMILLAS})')
    ejes[1].set_yscale('symlog', linthresh=1)
    ejes[1].set_ylim(0, 2000)
    ejes[1].set_ylabel('Píxeles incorrectos en total, de 1120\n(media ± desvío; lineal hasta 1, log arriba)')
    fig.legend(*ejes[0].get_legend_handles_labels(), loc='upper center', ncol=2, fontsize=10, bbox_to_anchor=(0.5, 1.06))
    fig.supxlabel(f'Red {red()} con latente tanh; cada barra es el mejor η de ese optimizador entre tres y cinco, con 10 semillas',
                  fontsize=10)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def coma(valor) -> str:
    """η o lista de η ('0.001 / 0.003') con coma decimal."""
    return (f'{valor:g}' if isinstance(valor, float) else str(valor)).replace('.', ',')


def pm(m: float, d: float, decimales: int = 1) -> str:
    return f'{m:.{decimales}f} ± {d:.{decimales}f}'.replace('.', ',')


def texto(tabla: pd.DataFrame, mejores: pd.DataFrame) -> str:
    lineas = ['| Costo | Optimizador | η | Semillas con ≤ 1 px | Px mal en total | Máx. px por letra (media) |',
              '|---|---|---|---|---|---|']
    for f in tabla.sort_values(['costo', 'optimizador', 'eta']).itertuples():
        lineas.append(f"| {COSTOS[f.costo]} | {NOMBRES[f.optimizador].replace(chr(10), ' ')} | {coma(f.eta)} | "
                      f'{int(f.logradas)} de {int(f.semillas)} | {pm(f.total_px_media, f.total_px_desvio)} | '
                      f"{f'{f.max_px_media:.1f}'.replace('.', ',')} |")
    mejor = ['| Costo | Optimizador | Mejor η | Semillas con ≤ 1 px | Px mal en total | Empata con η | En el borde |',
             '|---|---|---|---|---|---|---|']
    for f in mejores.itertuples():
        mejor.append(f"| {COSTOS[f.costo]} | {NOMBRES[f.optimizador].replace(chr(10), ' ')} | {coma(f.eta)} | {int(f.logradas)} de {SEMILLAS} | "
                     f"{pm(f.total_px_media, f.total_px_desvio)} | {coma(f.empata_con) if isinstance(f.empata_con, str) and f.empata_con else '—'} | {'sí' if f.en_borde else 'no'} |")
    return f"""## 1a.2 rehecha · Optimizadores sobre la red con latente tanh (e1a2_opt_tanh)

Red elegida en 1a.1 (D57, D59, D62): {red()}, ocultas y latente tanh, salida logística. 5 optimizadores × 3 η (los de `config/e1a2.json` × ⅓, 1 y 3; más uno o dos donde el mejor quedó en el borde) × 2 costos × {SEMILLAS} semillas, 10000 épocas, batch de 32. Datos en `results/e1a2_opt_tanh/`, figura `fig_optimizador_costo.png`.

### Mejor η de cada optimizador (`mejor_eta.csv`)

{chr(10).join(mejor)}

### Todas las celdas (`todas.csv`)

{chr(10).join(lineas)}
"""


def main() -> None:
    aplicar_estilo()
    tabla = leer()
    mejores = mejor_eta(tabla)
    tabla.to_csv(SALIDA / 'todas.csv', index=False)
    mejores.to_csv(SALIDA / 'mejor_eta.csv', index=False)
    figura(mejores, SALIDA / 'fig_optimizador_costo.png')
    escribir_seccion('e1a2_opt_tanh', texto(tabla, mejores))
    print(mejores[['costo', 'optimizador', 'eta', 'logradas', 'total_px_media', 'total_px_desvio', 'empata_con', 'en_borde']]
          .to_string(index=False))


if __name__ == '__main__':
    main()
