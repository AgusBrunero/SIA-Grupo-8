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

from experiments.comun import COLOR, RAIZ, RAMPA_AE, TINTA, TINTA_SECUNDARIA, aplicar_estilo
from src.data.font import cargar_font, distancias_hamming

RESULTADOS = RAIZ / 'results'
SALIDA = RAIZ / 'docs' / 'presentacion'

# Rótulos en palabras de los escalones de e1a2 (qué cambia cada uno está en config/e1a2.json)
ARQUITECTURAS = {'E1': 'Sin capa oculta\n(35–2–35)', 'E2': 'Con capa oculta\nde 20', 'E3': 'Latente lineal'}
# Optimizadores de la comparación justa (results/e1a2_optimizadores), en el orden de la lámina
OPTIMIZADORES = {'gd': 'Descenso\npor gradiente', 'momentum': 'Momentum', 'eta_adaptativo': 'η\nadaptativo',
                 'rmsprop': 'RMSProp', 'adam': 'Adam'}
COSTOS = {'ec': ('Error cuadrático', RAMPA_AE[3]), 'bce': ('Entropía cruzada', RAMPA_AE[6])}


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
    poner('logro_e3', f"{int(e1a2.loc['E3', 'logradas'])} de {n}", 'e1a2/resumen.csv · E3 · logradas y semillas')
    poner('px_bce', num(e1a2.loc['E9', 'total_px_media'], 0), 'e1a2/resumen.csv · E9 · total_px_media')
    mejores = leer('e1a2_optimizadores/mejor_eta.csv')
    assert len(mejores) == 2 * len(OPTIMIZADORES) and (mejores.semillas == n).all()
    poner('max_logro_ec', f"{int(mejores[mejores.costo == 'ec'].logradas.max())} de {n}",
          'e1a2_optimizadores/mejor_eta.csv · error cuadrático · máximo de logradas')
    bce_mejores = mejores[mejores.costo == 'bce']
    poner('bce_llegan', f'{int((bce_mejores.logradas == n).sum())} de {len(bce_mejores)}',
          'e1a2_optimizadores/mejor_eta.csv · entropía cruzada · optimizadores con logradas = semillas')
    profundas = leer('e1a2_optimizadores/profunda.csv').set_index('epocas')
    for clave, epocas in (('logro_profunda', 10000), ('logro_profunda_larga', 40000)):
        fila = profundas.loc[epocas]
        poner(clave, f'{int(fila.logradas)} de {int(fila.semillas)}', f'e1a2_optimizadores/profunda.csv · {epocas} épocas · logradas y semillas')
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
    pasos = leer('e1a4_interpolacion/pasos.csv')
    assert pasos.distancia.max() == int(elegidas[elegidas['item'] == '1a.4'].distancia.iloc[0])
    poner('pasos_nuevos', str(int((pasos.distancia > 0).sum())), 'e1a4_interpolacion/pasos.csv · pasos con distancia > 0')
    poner('pasos_total', str(len(pasos)), 'e1a4_interpolacion/pasos.csv · cantidad de pasos')
    minima = leer('e1a4_grilla/minima.csv').iloc[0]
    assert minima.distancia_minima == int(np.where(np.eye(len(X), dtype=bool), np.inf, hamming).min())
    poner('dist_minima_letras', str(int(minima.distancia_minima)), 'e1a4_grilla/minima.csv · distancia_minima')
    unicas = leer('e1a4_grilla/unicas.csv')
    poner('unicas_total', str(len(unicas)), 'e1a4_grilla/unicas.csv · salidas únicas que no son letras')
    poner('unicas_2px', str(int((unicas.distancia >= 2).sum())), 'e1a4_grilla/unicas.csv · con distancia ≥ 2')
    poner('dist_candidata', str(int(letras_elegidas().distancia.min())),
          'e1a4_grilla/unicas.csv · menor distancia de las elegidas en config/elegidas.json (letras_grilla)')

    e1b = leer('e1b/resumen.csv')
    niveles = json.loads((RESULTADOS / 'e1b' / 'config.json').read_text(encoding='utf-8'))['niveles']
    for clave, p in (('p_005', 0.05), ('p_01', 0.1), ('p_03', 0.3)):
        assert p in niveles
        poner(clave, f'{p:g}'.replace('.', ','), 'e1b/config.json · niveles')
    referencia = leer('e1b/referencia.csv').set_index('p_prueba')
    for clave, p in (('01', 0.1), ('03', 0.3)):
        fila = referencia.loc[p]
        poner(f'exacto_dae_{clave}', f'{num(100 * fila.dae_B_exactos_media, 0)} %',
              f'e1b/referencia.csv · p = {num(p, 1)} · dae_B_exactos_media (de e1b/resumen.csv)')
        poner(f'exacto_ref_{clave}', f'{num(100 * fila.letra_cercana_media, 0)} %',
              f'e1b/referencia.csv · p = {num(p, 1)} · letra_cercana_media')
    poner('otra_01', f'{num(100 * referencia.loc[0.1].mas_cerca_de_otra_media, 1)} %', 'e1b/referencia.csv · p = 0,1 · mas_cerca_de_otra_media')
    replicas = leer('e1b_replicas/resumen.csv')
    replicas = replicas[np.isclose(replicas.p_prueba, 0.1)]
    larga = replicas.loc[replicas.exactos_media.idxmax()]
    assert larga.replicas == 5 and larga.epocas == 40000 and larga.semillas == 5
    poner('exacto_dae_max_01', f'{num(100 * larga.exactos_media, 0)} %',
          'e1b_replicas/resumen.csv · ×5 réplicas, 40000 épocas · p = 0,1 · exactos_media')
    corta = replicas[(replicas.replicas == 1) & (replicas.epocas == 10000)].iloc[0]
    poner('hasta1_dae_01', f'{num(100 * corta.hasta_1px_media, 0)} %',
          'e1b_replicas/resumen.csv · ×1, 10000 épocas · p = 0,1 · hasta_1px_media')
    poner('hasta1_dae_max_01', f'{num(100 * larga.hasta_1px_media, 0)} %',
          'e1b_replicas/resumen.csv · ×5 réplicas, 40000 épocas · p = 0,1 · hasta_1px_media')
    poner('otra_03', f'{num(100 * referencia.loc[0.3].mas_cerca_de_otra_media, 0)} %', 'e1b/referencia.csv · p = 0,3 · mas_cerca_de_otra_media')

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

    e2b_config = json.loads((RESULTADOS / 'e2b' / 'config.json').read_text(encoding='utf-8'))
    poner('n_relojes', str(e2b_config['dataset']['n']), 'e2b/config.json · dataset.n')
    from src.data.relojes import PARAMETROS
    poner('n_parametros', str(len(PARAMETROS)), 'src/data/relojes.py · cantidad de parámetros')
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
    poner('horas_vae', num(vae.horas_cubiertas.mean(), 1), 'e2c/corridas.csv · VAE λ = 1 · media de horas_cubiertas')
    poner('horas_ae', num(ae.horas_cubiertas.mean(), 1), 'e2c/corridas.csv · AE · media de horas_cubiertas')
    poner('dist_generada', num(vae.distancia_vecino_media.mean(), 3), 'e2c/corridas.csv · VAE λ = 1 · media de distancia_vecino_media')
    from experiments.e2c_latente_generacion import vecino_por_pixel
    from src.data.relojes import generar_relojes
    relojes = generar_relojes(e2b_config['dataset']['n'], e2b_config['dataset']['semilla']).X
    poner('dist_real', num(vecino_por_pixel(relojes, relojes, excluir_identicas=True)[1].mean(), 3),
          'src/data/relojes.py · distancia media de cada reloj al otro más cercano (igual que en RESULTADOS.md)')
    from src.nn.perdidas import bce
    # Piso de L: la BCE de los datos en grises consigo mismos (la misma cuenta que e2b escribe en RESULTADOS.md)
    poner('L_piso', num(bce(np.clip(relojes, 1e-12, 1 - 1e-12), relojes, 'media')[0], 1),
          'src/data/relojes.py · BCE de los 500 relojes consigo mismos (igual que en RESULTADOS.md, e2b)')
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


def figura_optimizador_costo(ruta) -> None:
    """Lámina 4: semillas que llegan al objetivo con el mejor η de cada optimizador, por función de costo."""
    mejores = leer('e1a2_optimizadores/mejor_eta.csv')
    x = np.arange(len(OPTIMIZADORES))
    ancho = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 3.7))
    for k, (costo, (nombre, color)) in enumerate(COSTOS.items()):
        m = mejores[mejores.costo == costo].set_index('optimizador').loc[list(OPTIMIZADORES)]
        assert (m.semillas == 10).all()
        barras = ax.bar(x + (k - 0.5) * ancho, m.logradas, ancho, color=color, label=nombre)
        for barra, logradas in zip(barras, m.logradas):
            ax.text(barra.get_x() + barra.get_width() / 2, logradas + 0.2, str(int(logradas)), ha='center', fontsize=12)
    ax.set_xticks(x, OPTIMIZADORES.values(), fontsize=11)
    ax.set_ylim(0, 11.5)
    ax.set_ylabel('Semillas (de 10) con todas las letras\na 1 píxel incorrecto o menos')
    ax.set_xlabel('Cada optimizador con su mejor η entre tres o cuatro valores (mismo modelo)')
    # La leyenda va arriba, fuera del área de las barras, para no tapar ninguna barra ni su valor
    ax.legend(fontsize=11, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.grid(axis='x', visible=False)
    fig.savefig(ruta)
    plt.close(fig)


def separar_etiquetas(puntos: np.ndarray, distancia: float = 0.05, iteraciones: int = 300) -> np.ndarray:
    """Posiciones de etiquetas (en coordenadas normalizadas a [0, 1]) que no se pisan entre sí ni con
    los puntos: arrancan arriba a la derecha de su punto, se repelen y vuelven hacia su punto."""
    etiquetas = puntos + np.array([0.012, 0.018])
    for _ in range(iteraciones):
        empuje = np.zeros_like(etiquetas)
        for otros in (etiquetas, puntos):
            diferencia = etiquetas[:, None, :] - otros[None, :, :]
            d = np.linalg.norm(diferencia, axis=2) + np.eye(len(etiquetas)) * (otros is etiquetas) * 1e9 + 1e-9
            cerca = d < distancia
            empuje += (diferencia / d[..., None] * ((distancia - d) * cerca)[..., None]).sum(axis=1)
        etiquetas += 0.5 * empuje + 0.02 * (puntos + np.array([0.012, 0.018]) - etiquetas)
    return etiquetas


def figura_latente(ruta) -> None:
    latente = leer('e1a3_e1a4/latente.csv')
    latente = latente[latente.modelo == 'AE']
    fig, ejes = plt.subplots(1, 2, figsize=(8.6, 3.9))
    for ax, semilla in zip(ejes, (0, 1)):
        g = latente[latente.semilla == semilla]
        assert len(g) == 32
        ax.scatter(g.z1, g.z2, s=14, color=COLOR['AE'], zorder=3)
        z = g[['z1', 'z2']].to_numpy()
        minimo, rango = z.min(axis=0), np.ptp(z, axis=0)
        posiciones = separar_etiquetas((z - minimo) / rango) * rango + minimo
        for (z1, z2), (e1, e2), etiqueta in zip(z, posiciones, g.etiqueta):
            lejos = np.hypot(*(((e1, e2) - np.array([z1, z2])) / rango)) > 0.035
            ax.annotate(str(etiqueta), (z1, z2), xytext=(e1, e2), textcoords='data', fontsize=10, ha='left', va='bottom',
                        arrowprops={'arrowstyle': '-', 'color': TINTA_SECUNDARIA, 'linewidth': 0.6} if lejos else None)
        ax.set_xlabel(f'z₁ · entrenamiento con la semilla {semilla}')
    ejes[0].set_ylabel('z₂')
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def figura_interpolacion(ruta, cada: int = 5) -> None:
    """Lámina 9: un paso de cada `cada` entre la “o” y la “e”, con su distancia, y el camino en el latente."""
    pasos = leer('e1a4_interpolacion/pasos.csv', dtype={'pixeles': str})
    config = json.loads((RESULTADOS / 'e1a4_interpolacion' / 'config.json').read_text(encoding='utf-8'))
    latente = leer('e1a3_e1a4/latente.csv')
    latente = latente[(latente.modelo == 'AE') & (latente.semilla == config['semilla'])]
    assert len(latente) == 32 and len(pasos) == config['pasos']
    elegidos = pasos[pasos.paso % cada == 0]
    fig = plt.figure(figsize=(11, 4.2))
    grilla = fig.add_gridspec(2, len(elegidos), height_ratios=[1, 2.4], hspace=0.55)
    for i, fila in enumerate(elegidos.itertuples()):
        ax = fig.add_subplot(grilla[0, i])
        ax.imshow(np.array(list(fila.pixeles), dtype=int).reshape(7, 5), cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(visible=False)
        for borde in ax.spines.values():
            borde.set_visible(True)
            borde.set_color(COLOR['AE'] if fila.distancia > 0 else '#c3c2b7')
            borde.set_linewidth(2 if fila.distancia > 0 else 0.8)
        ax.set_xlabel(f'{fila.distancia} px de “{fila.vecino}”', fontsize=9, color=TINTA_SECUNDARIA)
    ax = fig.add_subplot(grilla[1, :])
    ax.scatter(latente.z1, latente.z2, s=12, color=TINTA_SECUNDARIA, zorder=2)
    for fila in latente.itertuples():
        ax.annotate(str(fila.etiqueta), (fila.z1, fila.z2), xytext=(3, 2), textcoords='offset points', fontsize=9, color=TINTA)
    ax.plot(pasos.z1, pasos.z2, color=COLOR['AE'], linewidth=1.2, zorder=3)
    nuevos = pasos[pasos.distancia > 0]
    ax.scatter(nuevos.z1, nuevos.z2, s=40, color=COLOR['AE'], zorder=4, label='Pasos que no son una letra del dataset')
    ax.scatter(elegidos.z1, elegidos.z2, s=14, facecolors='white', edgecolors=COLOR['AE'], zorder=5,
               label='Pasos mostrados arriba')
    ax.set_xlabel('z₁')
    ax.set_ylabel('z₂')
    ax.legend(fontsize=9, loc='upper right')
    fig.savefig(ruta)
    plt.close(fig)


def figura_dae_entrenamiento(ruta) -> None:
    """Lámina 13: % de reconstrucciones exactas del DAE B con el entrenamiento corto y el largo, contra la letra más cercana."""
    resumen = leer('e1b_replicas/resumen.csv')
    assert (resumen.semillas == 5).all()
    curvas = (((1, 10000), 'DAE B, entrenamiento corto (10.000 épocas)', '#f6b294'),
              ((5, 40000), 'DAE B, entrenamiento largo (×5 réplicas, 40.000 épocas)', COLOR['DAE']))
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    for (replicas, epocas), nombre, color in curvas:
        g = resumen[(resumen.replicas == replicas) & (resumen.epocas == epocas)].sort_values('p_prueba')
        assert len(g) == 5
        ax.errorbar(g.p_prueba, 100 * g.exactos_media, yerr=100 * g.exactos_desvio, color=color, marker='o', linewidth=2.4,
                    capsize=3, label=nombre)
    ref = resumen.drop_duplicates('p_prueba').sort_values('p_prueba')
    ax.plot(ref.p_prueba, 100 * ref.letra_cercana_media, color=TINTA_SECUNDARIA, linestyle='--', marker='s', linewidth=2,
            label='Letra más cercana a la entrada ruidosa')
    ax.set_xticks(ref.p_prueba, [num(p, 2).rstrip('0').rstrip(',') for p in ref.p_prueba])
    ax.set_xlabel('p de prueba (probabilidad de invertir cada píxel)')
    ax.set_ylabel('% de reconstrucciones exactas\n(media ± desvío, 5 semillas)')
    ax.set_ylim(0, 102)
    ax.legend(fontsize=10, loc='lower left')
    fig.savefig(ruta)
    plt.close(fig)


def letras_elegidas() -> pd.DataFrame:
    """Las salidas de la grilla fina elegidas para la lámina 8 (config/elegidas.json · letras_grilla)."""
    config = json.loads((RAIZ / 'config' / 'elegidas.json').read_text(encoding='utf-8'))['letras_grilla']
    unicas = pd.read_csv(RAIZ / config['origen'], dtype={'pixeles': str}).set_index('numero')
    elegidas = unicas.loc[config['numeros']].reset_index()
    assert 1 <= len(elegidas) <= 3 and (elegidas.distancia >= 2).all()
    return elegidas


def figura_letras_nuevas(ruta) -> None:
    """Lámina 8: las salidas elegidas, cada una junto a su letra más cercana, y dónde caen en el latente."""
    elegidas = letras_elegidas()
    X, etiquetas = cargar_font()
    latente = leer('e1a3_e1a4/latente.csv')
    latente = latente[(latente.modelo == 'AE') & (latente.semilla == 0)]
    fig = plt.figure(figsize=(11.5, 4.4))
    grilla = fig.add_gridspec(2, 2 * len(elegidas) + 5, width_ratios=[1, 1] * len(elegidas) + [1.1, 2, 2, 2, 2], hspace=0.35)
    for i, fila in enumerate(elegidas.itertuples()):
        for j, (pixeles, rotulo, color) in enumerate((
                (np.array(list(fila.pixeles), dtype=int), f'#{fila.numero}\nnueva', COLOR['AE']),
                (X[etiquetas.index(fila.vecino)], f'“{fila.vecino}”\na {fila.distancia} px', '#c3c2b7'))):
            ax = fig.add_subplot(grilla[:, 2 * i + j])
            ax.imshow(pixeles.reshape(7, 5), cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(visible=False)
            for borde in ax.spines.values():
                borde.set_visible(True)
                borde.set_color(color)
                borde.set_linewidth(2.2 if j == 0 else 0.8)
            ax.set_xlabel(rotulo, fontsize=10, color=TINTA if j == 0 else TINTA_SECUNDARIA)
    ax = fig.add_subplot(grilla[:, 2 * len(elegidas) + 1:])
    ax.scatter(latente.z1, latente.z2, s=10, color=TINTA_SECUNDARIA, zorder=2)
    for fila in latente.itertuples():
        ax.annotate(str(fila.etiqueta), (fila.z1, fila.z2), xytext=(2, 2), textcoords='offset points', fontsize=8, color=TINTA_SECUNDARIA)
    ax.scatter(elegidas.z1, elegidas.z2, s=70, marker='*', color=COLOR['AE'], zorder=4)
    for fila in elegidas.itertuples():
        ax.annotate(f'#{fila.numero}', (fila.z1, fila.z2), xytext=(5, -10), textcoords='offset points', fontsize=10,
                    color=COLOR['AE'], fontweight='bold')
    ax.set_xlabel('z₁')
    ax.set_ylabel('z₂')
    fig.savefig(ruta)
    plt.close(fig)


def piso_L() -> float:
    """BCE de los relojes en grises consigo mismos: lo mínimo que puede valer L (igual que e2b)."""
    from src.data.relojes import generar_relojes
    from src.nn.perdidas import bce
    config = json.loads((RESULTADOS / 'e2b' / 'config.json').read_text(encoding='utf-8'))
    relojes = generar_relojes(config['dataset']['n'], config['dataset']['semilla']).X
    return float(bce(np.clip(relojes, 1e-12, 1 - 1e-12), relojes, 'media')[0])


def figura_curvas_vae(ruta) -> None:
    """Lámina de las curvas del VAE: L y R por época, con el piso de L como línea punteada."""
    curvas = leer('e2b/curvas.csv')
    assert curvas.semilla.nunique() >= 5
    piso = piso_L()
    fig, ejes = plt.subplots(1, 2, figsize=(10, 3.8))
    for columna, ax in (('L', ejes[0]), ('R', ejes[1])):
        for _, curva in curvas.groupby('semilla'):
            ax.plot(curva.epoca, curva[columna], color=COLOR['VAE'], linewidth=1.2, alpha=0.7)
        ax.set_xlabel('Época (una línea por semilla)')
    ejes[0].axhline(piso, color=TINTA, linestyle='--', linewidth=1.4, label=f'Piso de L: {num(piso, 1)}')
    ejes[0].set_yscale('log')
    ejes[0].yaxis.set_major_formatter(lambda valor, _: f'{valor:g}'.replace('.', ','))
    ejes[0].yaxis.set_minor_formatter(lambda valor, _: f'{valor:g}' if valor in (60, 80, 200, 300) else '')
    ejes[0].set_ylabel('Reconstrucción L (BCE por carita,\nescala log)')
    ejes[1].set_ylabel('Regularización R (KL por carita)')
    ejes[0].legend(fontsize=11)
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
                                                            ('B', 0.1, 'DAE', 'DAE entrenado con p = 0,1 (entrenamiento corto)')):
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
    figura_optimizador_costo(SALIDA / 'figuras' / 'optimizador_costo.png')
    figura_latente(SALIDA / 'figuras' / 'latente_dos_semillas.png')
    figura_interpolacion(SALIDA / 'figuras' / 'interpolacion.png')
    figura_curvas_vae(SALIDA / 'figuras' / 'curvas_vae.png')
    figura_letras_nuevas(SALIDA / 'figuras' / 'letras_nuevas.png')
    figura_dae_entrenamiento(SALIDA / 'figuras' / 'dae_entrenamiento.png')
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
