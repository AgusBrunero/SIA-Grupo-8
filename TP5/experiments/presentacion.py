"""Insumos de la presentación (sesión 8): los números que van en pantalla y las figuras
hechas a medida de una lámina.

No corre ningún experimento: lee los CSV de results/. Escribe
  docs/presentacion/manifest.json   cada número en pantalla, con su texto y el CSV del que sale
                                    (regla R10 del plan de presentación del TP4);
  docs/presentacion/figuras/*.png   figuras con rótulos en palabras, sin códigos de escalón (R8).

Uso, desde TP5/:
    .venv/bin/python -m experiments.presentacion
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.comun import COLOR, RAIZ, TINTA_SECUNDARIA, aplicar_estilo
from src.data.font import cargar_font, distancias_hamming

RESULTADOS = RAIZ / 'results'
SALIDA = RAIZ / 'docs' / 'presentacion'

# Rótulos en palabras de los escalones de e1a2 (qué cambia cada uno está en config/e1a2.json)
ARQUITECTURAS = {'E1': 'Sin capa oculta\n(35–2–35)', 'E2': 'Con capa oculta\nde 20', 'E3': 'Latente lineal',
                 'E4': 'Ocultas tanh'}
OPTIMIZADORES = {'E4': 'Descenso\npor gradiente', 'E5': 'Momentum', 'E6': 'η\nadaptativo', 'E7': 'RMSProp',
                 'E8': 'Adam', 'E10': 'Powell'}


def num(valor: float, decimales: int = 2) -> str:
    return f'{valor:.{decimales}f}'.replace('.', ',')


def leer(ruta: str, **kw) -> pd.DataFrame:
    return pd.read_csv(RESULTADOS / ruta, **kw)


def numeros() -> dict:
    """clave → {texto, origen}. El texto es exactamente lo que aparece en la lámina."""
    m: dict[str, dict] = {}

    def poner(clave: str, texto: str, origen: str) -> None:
        m[clave] = {'texto': texto, 'origen': origen}

    e1a2 = leer('e1a2/resumen.csv').set_index('escalon')
    n = int(e1a2.semillas.iloc[0])
    assert (e1a2.semillas == n).all()
    poner('px_base', num(e1a2.loc['E1', 'total_px_media'], 0), 'e1a2/resumen.csv · E1 · total_px_media')
    poner('px_latente_lineal', num(e1a2.loc['E3', 'total_px_media'], 0), 'e1a2/resumen.csv · E3 · total_px_media')
    poner('logro_gd', f"{int(e1a2.loc['E4', 'logradas'])} de {n}", 'e1a2/resumen.csv · E4 · logradas y semillas')
    poner('logro_bce', f"{int(e1a2.loc['E9', 'logradas'])} de {n}", 'e1a2/resumen.csv · E9 · logradas y semillas')
    poner('px_bce', num(e1a2.loc['E9', 'total_px_media'], 0), 'e1a2/resumen.csv · E9 · total_px_media')
    tanh = e1a2.loc[['E4', 'E5', 'E6', 'E7', 'E8']]
    poner('saturados', f'{int(tanh.px_mal_saturados.sum())} de {int(tanh.px_mal_total.sum())}',
          'e1a2/resumen.csv · E4 a E8 · px_mal_saturados y px_mal_total')

    X, etiquetas = cargar_font()
    poner('n_letras', str(len(X)), 'docs/tp5/font.h · cantidad de patrones')
    hamming = distancias_hamming(X)
    poner('dist_l_barra', str(int(hamming[etiquetas.index('l'), etiquetas.index('|')])),
          'docs/tp5/font.h · distancia de Hamming entre l y |')

    pca = leer('e1a3_e1a4/pca.csv').iloc[0]
    poner('pca_px', str(int(pca.total_pixeles)), 'e1a3_e1a4/pca.csv · total_pixeles')
    poner('pca_var', f'{num(100 * (pca.varianza_c1 + pca.varianza_c2), 0)} %', 'e1a3_e1a4/pca.csv · varianza_c1 + varianza_c2')
    estabilidad = leer('e1a3_e1a4/estabilidad.csv')
    par = estabilidad[(estabilidad.tipo == 'AE–AE') & (estabilidad.a == 'semilla 0') & (estabilidad.b == 'semilla 1')]
    poner('corr_semillas', num(par.correlacion.iloc[0]), 'e1a3_e1a4/estabilidad.csv · AE–AE · semilla 0 y semilla 1')

    elegidas = leer('elegidas/elegidas.csv')
    poner('letra_dist', str(int(elegidas[elegidas['item'] == '1a.4'].distancia.iloc[0])), 'elegidas/elegidas.csv · 1a.4 · distancia')

    e1b = leer('e1b/resumen.csv')

    def celda(arquitectura: str, p_entrenamiento: float, p_prueba: float) -> pd.Series:
        fila = e1b[(e1b.arquitectura == arquitectura) & np.isclose(e1b.p_entrenamiento, p_entrenamiento)
                   & np.isclose(e1b.p_prueba, p_prueba)]
        assert len(fila) == 1
        return fila.iloc[0]

    for arquitectura in 'ABC':
        poner(f'mal_{arquitectura}', num(celda(arquitectura, 0.1, 0.1).mal_media, 1),
              f'e1b/resumen.csv · {arquitectura} · entrenada y probada con p = 0,1 · mal_media')
    poner('entran_bajo', num(celda('A', 0.0, 0.05).alterados, 2), 'e1b/resumen.csv · p_prueba 0,05 · alterados')
    poner('salen_sin_ruido', num(celda('A', 0.0, 0.05).mal_media, 1), 'e1b/resumen.csv · A · p_entrenamiento 0 · p_prueba 0,05 · mal_media')
    poner('hasta1_bajo', f"{num(100 * celda('B', 0.1, 0.1).hasta_1px_media, 0)} %",
          'e1b/resumen.csv · B · entrenada y probada con p = 0,1 · hasta_1px_media')
    poner('hasta1_alto', f"{num(100 * celda('B', 0.1, 0.3).hasta_1px_media, 0)} %",
          'e1b/resumen.csv · B · entrenada con p = 0,1, probada con p = 0,3 · hasta_1px_media')

    e2b_config = json.loads((RESULTADOS / 'e2b' / 'config.json').read_text(encoding='utf-8'))
    poner('n_caritas', str(e2b_config['dataset']['n']), 'e2b/config.json · dataset.n')
    from src.data.emojis import PARAMETROS
    poner('n_parametros', str(len(PARAMETROS)), 'src/data/emojis.py · cantidad de parámetros')
    curvas = leer('e2b/curvas.csv')
    primera, ultima = curvas[curvas.epoca == 1], curvas[curvas.epoca == curvas.epoca.max()]
    assert len(primera) == len(ultima) == curvas.semilla.nunique() >= 5
    poner('L_inicial', num(primera.L.mean(), 0), 'e2b/curvas.csv · época 1 · media de L entre semillas')
    poner('L_final', num(ultima.L.mean(), 0), 'e2b/curvas.csv · última época · media de L entre semillas')
    poner('R_final', num(ultima.R.mean(), 1), 'e2b/curvas.csv · última época · media de R entre semillas')

    e2c = leer('e2c/corridas.csv')
    ae, vae = e2c[e2c.modelo == 'AE'], e2c[(e2c.modelo == 'VAE') & np.isclose(e2c['lambda'], 1.0)]
    assert len(ae) == len(vae) >= 5
    poner('desvio_ae', num(ae.codigo_desvio.mean(), 1), 'e2c/corridas.csv · AE · media de codigo_desvio')
    poner('desvio_vae', num(vae.codigo_desvio.mean(), 1), 'e2c/corridas.csv · VAE λ = 1 · media de codigo_desvio')
    poner('dist_generada', num(vae.distancia_vecino_media.mean(), 3), 'e2c/corridas.csv · VAE λ = 1 · media de distancia_vecino_media')
    from experiments.e2c_latente_generacion import vecino_por_pixel
    from src.data.emojis import generar_caritas
    caritas = generar_caritas(e2b_config['dataset']['n'], e2b_config['dataset']['semilla']).X
    poner('dist_real', num(vecino_por_pixel(caritas, caritas, excluir_identicas=True)[1].mean(), 3),
          'src/data/emojis.py · distancia media de cada carita a la otra más cercana (igual que en RESULTADOS.md)')
    return m


def figura_arquitecturas(ruta) -> None:
    e = leer('e1a2/resumen.csv').set_index('escalon').loc[list(ARQUITECTURAS)]
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    ax.bar(range(len(e)), e.total_px_media, yerr=e.total_px_desvio, color=COLOR['AE'], width=0.6, capsize=5,
           error_kw={'ecolor': TINTA_SECUNDARIA, 'elinewidth': 1.2})
    for i, (media, desvio) in enumerate(zip(e.total_px_media, e.total_px_desvio)):
        ax.text(i, media + desvio + 6, num(media, 0), ha='center', fontsize=13)
    ax.set_xticks(range(len(e)), ARQUITECTURAS.values(), fontsize=12)
    ax.set_ylabel('Píxeles incorrectos en las 32 letras\n(media ± desvío, 10 semillas)')
    ax.set_xlabel('Cada barra agrega un cambio a la anterior (error cuadrático, descenso por gradiente)')
    ax.grid(axis='x', visible=False)
    fig.savefig(ruta)
    plt.close(fig)


def figura_logro(con_bce: bool, ruta) -> None:
    e = leer('e1a2/resumen.csv').set_index('escalon')
    rotulos = dict(OPTIMIZADORES, **({'E9': 'Adam +\nentropía cruzada'} if con_bce else {}))
    e = e.loc[list(rotulos)]
    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    ax.bar(range(len(e)), e.logradas, color=COLOR['AE'], width=0.6)
    for i, (logradas, semillas) in enumerate(zip(e.logradas, e.semillas)):
        ax.text(i, logradas + 0.25, f'{int(logradas)} de {int(semillas)}', ha='center', fontsize=13)
    ax.set_xticks(range(len(e)), rotulos.values(), fontsize=9 if con_bce else 10)
    ax.set_ylim(0, 11.5)
    ax.set_ylabel('Semillas (de 10) con todas las letras\na 1 píxel incorrecto o menos')
    ax.set_xlabel('Optimizador' + (' (el último cambia además la función de costo)' if con_bce else ' (todos con error cuadrático)'))
    ax.grid(axis='x', visible=False)
    fig.savefig(ruta)
    plt.close(fig)


def figura_latente(ruta) -> None:
    latente = leer('e1a3_e1a4/latente.csv')
    latente = latente[latente.modelo == 'AE']
    fig, ejes = plt.subplots(1, 2, figsize=(8.6, 3.9))
    for ax, semilla in zip(ejes, (0, 1)):
        g = latente[latente.semilla == semilla]
        assert len(g) == 32
        ax.scatter(g.z1, g.z2, s=14, color=COLOR['AE'])
        for fila in g.itertuples():
            ax.annotate(str(fila.etiqueta), (fila.z1, fila.z2), xytext=(3, 3), textcoords='offset points', fontsize=10)
        ax.set_xlabel(f'z₁ · entrenamiento con la semilla {semilla}')
    ejes[0].set_ylabel('z₂')
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def figura_ruido(ruta) -> None:
    e1b = leer('e1b/resumen.csv')
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    niveles = sorted(e1b.p_prueba.unique())
    ax.plot(niveles, [35 * p for p in niveles], color=TINTA_SECUNDARIA, linestyle='--', linewidth=1.5,
            label='Entrada con ruido (sin red)')
    for arquitectura, p_entrenamiento, modelo, etiqueta in (('A', 0.0, 'AE', 'Autoencoder entrenado sin ruido'),
                                                            ('B', 0.1, 'DAE', 'DAE entrenado con p = 0,1')):
        g = e1b[(e1b.arquitectura == arquitectura) & np.isclose(e1b.p_entrenamiento, p_entrenamiento)].sort_values('p_prueba')
        assert len(g) == len(niveles) and (g.semillas >= 5).all()
        ax.errorbar(g.p_prueba, g.mal_media, yerr=g.mal_desvio, color=COLOR[modelo], marker='o', linewidth=2, capsize=4, label=etiqueta)
    ax.set_xticks(niveles, [num(p, 2) for p in niveles])
    ax.set_xlabel('p: probabilidad de invertir cada píxel de la entrada')
    ax.set_ylabel('Píxeles incorrectos por letra\n(media ± desvío, 5 semillas)')
    ax.legend(fontsize=11)
    fig.savefig(ruta)
    plt.close(fig)


ESQUEMAS = {
    'esquema_ae': ('AE', [('X', False), ('Encoder f', True), ('Z', False), ('Decoder g', True), ('X′', False)]),
    'esquema_saturacion': ('AE', [('salida\nsaturada', False), ('θ′(h) ≈ 0', True), ('δ ≈ 0', False),
                                  ('el peso\nno cambia', False)]),
    'esquema_dae': ('DAE', [('X', False), ('ruido', False), ('X̃', False), ('Encoder', True), ('Z', False), ('Decoder', True),
                            ('X′', False)]),
    'esquema_vae': ('VAE', [('x', False), ('Encoder', True), ('μ(x), σ(x)', False), ('z = μ + σ·ϵ', False), ('Decoder', True),
                            ('x′', False)]),
}


def figura_esquema(modelo: str, cajas: list[tuple[str, bool]], ruta) -> None:
    """Cajas y flechas de izquierda a derecha; las redes van con el color del modelo."""
    ancho, alto, hueco = 1.9, 1.0, 0.6
    total = len(cajas) * ancho + (len(cajas) - 1) * hueco
    fig, ax = plt.subplots(figsize=(total, alto + 0.2))
    for i, (texto, red) in enumerate(cajas):
        x = i * (ancho + hueco)
        ax.add_patch(plt.Rectangle((x, 0), ancho, alto, facecolor=COLOR[modelo] if red else '#f4f3ee',
                                   edgecolor=COLOR[modelo] if red else '#c3c2b7', linewidth=1.2))
        ax.text(x + ancho / 2, alto / 2, texto, ha='center', va='center', fontsize=15 if len(texto) > 7 else 20,
                color='white' if red else '#0b0b0b', fontweight='bold' if red else 'normal')
        if i < len(cajas) - 1:
            ax.annotate('', xy=(x + ancho + hueco - 0.06, alto / 2), xytext=(x + ancho + 0.06, alto / 2),
                        arrowprops={'arrowstyle': '-|>', 'color': TINTA_SECUNDARIA, 'linewidth': 1.6})
    ax.set_xlim(-0.05, total + 0.05)
    ax.set_ylim(-0.1, alto + 0.1)
    ax.axis('off')
    fig.savefig(ruta)
    plt.close(fig)


def figura_confusiones(ruta, arquitectura: str = 'B', p: float = 0.1, cuantas: int = 6) -> None:
    """Las confusiones más frecuentes de una arquitectura entrenada y probada con el mismo p."""
    _, etiquetas = cargar_font()
    c = leer('e1b/confusion.csv')
    c = c[(c.arquitectura == arquitectura) & np.isclose(c.p_entrenamiento, p) & np.isclose(c.p_prueba, p)]
    por_letra = c.groupby('patron').cantidad.sum()
    assert por_letra.nunique() == 1                       # la misma cantidad de reconstrucciones por letra
    errores = c[c.patron != c.predicho].sort_values('cantidad', ascending=False).head(cuantas).iloc[::-1]
    rotulos = [f'“{etiquetas[int(f.patron)]}” leída como “{etiquetas[int(f.predicho)]}”' for f in errores.itertuples()]
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    ax.barh(range(len(errores)), errores.cantidad, color=COLOR['DAE'], height=0.6)
    for i, cantidad in enumerate(errores.cantidad):
        ax.text(cantidad + 0.4, i, str(int(cantidad)), va='center', fontsize=12)
    ax.set_yticks(range(len(errores)), rotulos, fontsize=12)
    ax.set_xlabel(f'Veces, de {int(por_letra.iloc[0])} reconstrucciones de esa letra (5 semillas, p = {num(p, 1)})')
    ax.grid(axis='y', visible=False)
    fig.savefig(ruta)
    plt.close(fig)


def main() -> None:
    aplicar_estilo()
    (SALIDA / 'figuras').mkdir(parents=True, exist_ok=True)
    figura_arquitecturas(SALIDA / 'figuras' / 'arquitecturas.png')
    figura_logro(False, SALIDA / 'figuras' / 'optimizadores.png')
    figura_logro(True, SALIDA / 'figuras' / 'entropia_cruzada.png')
    figura_latente(SALIDA / 'figuras' / 'latente_dos_semillas.png')
    figura_ruido(SALIDA / 'figuras' / 'ruido.png')
    figura_confusiones(SALIDA / 'figuras' / 'confusiones.png')
    for nombre, (modelo, cajas) in ESQUEMAS.items():
        figura_esquema(modelo, cajas, SALIDA / 'figuras' / f'{nombre}.png')
    manifest = {'nota': 'Generado por experiments/presentacion.py desde results/. No editar a mano.', 'numeros': numeros()}
    (SALIDA / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    for clave, dato in manifest['numeros'].items():
        print(f"{clave:20s} {dato['texto']:12s} {dato['origen']}")
    from experiments import presentacion_html
    presentacion_html.main()


if __name__ == '__main__':
    main()
