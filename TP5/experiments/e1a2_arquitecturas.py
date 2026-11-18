"""Consigna 1a.1 rehecha: arquitecturas del autoencoder con latente no lineal (nunca identidad).

Pregunta: con latente tanh (o logístico), ocultas tanh (o logísticas), salida logística y Adam, ¿qué
arquitectura aprende las 32 letras de font.h con a lo sumo 1 píxel incorrecto por letra, con
entropía cruzada y con error cuadrático? ¿Cuál elegimos para el AE?
Arquitecturas (encoder; el decoder es el espejo): T1 35-20-2, T2 35-20-10-2, T3 35-30-15-2,
T4 35-30-20-10-2 (latente tanh) y L2 35-20-10-2 (latente logístico). Ampliación D59 (feedback de la
cátedra: variar dimensiones y activaciones): T5 35-10-2, T6 35-30-2, T8 35-30-20-2, T9 35-30-30-15-2
(ocultas y latente tanh), G1 35-20-2 y G3 35-30-15-2 (ocultas logísticas, latente tanh). Cada una con
4 η de Adam (0,0003; 0,001; 0,003 y, por D62, 0,01) y 10 semillas. El η = 0,01 se agregó después de la
auditoría (8 de 11 redes tenían su mejor η en el borde, 0,003); vive en config/e1a2_arquitecturas_<costo>_<arq>_eta001.json.
Donde el mejor quedó en 0,01 con BCE se probó también 0,03 (config/e1a2_arquitecturas_bce_<arq>_eta003.json).
El orden completo (columna `orden` de eleccion.csv) sale de aplicar el criterio, sacar a la elegida y
volver a aplicarlo sobre las que quedan.

Criterio de elección, fijado antes de correr (docs/DECISIONES.md, D57):
  mejor η de cada arquitectura y costo: más semillas con ≤ 1 px; luego menos píxeles en total;
  luego menor mediana de época en llegar.
  Entre arquitecturas, en orden: (1) semillas logradas con BCE; (2) semillas logradas con error
  cuadrático; (3) píxeles en total con BCE; (4) mediana de época en llegar con BCE; (5) menos pesos.
  En (3) y (4) dos arquitecturas empatan si la diferencia no supera el desvío entre semillas.

No entrena: lee los corridas.csv que escribe experiments.e1a2_escalera con
config/e1a2_arquitecturas_<costo>_<arq>.json.
Sostén: Autoencoders diap. 3–6 (encoder y decoder con capas ocultas); Clase 10.2 diap. 23
(tanh); Clase 12.1 diap. 11–13 (Adam); Autoencoders diap. 51 (entropía cruzada).

Uso, desde TP5/:
    python3 -m experiments.e1a2_arquitecturas

Salida en results/e1a2_arquitecturas/: todas.csv, mejor_eta.csv, eleccion.csv,
fig_arquitecturas.png y el bloque e1a2_arquitecturas de docs/RESULTADOS.md.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.comun import RAIZ, RAMPA_AE, TINTA_SECUNDARIA, aplicar_estilo, escribir_seccion

SALIDA = RAIZ / 'results' / 'e1a2_arquitecturas'
SEMILLAS = 10
ETAS = [0.0003, 0.001, 0.003, 0.01]
COSTOS = {'bce': 'Entropía cruzada', 'ec': 'Error cuadrático'}
COLOR_COSTO = {'ec': RAMPA_AE[3], 'bce': RAMPA_AE[6]}
ARQUITECTURAS = {'T1': '35–20–2–20–35', 'T2': '35–20–10–2–10–20–35', 'T3': '35–30–15–2–15–30–35',
                 'T4': '35–30–20–10–2–10–20–30–35', 'L2': '35–20–10–2–10–20–35',
                 'T5': '35–10–2–10–35', 'T6': '35–30–2–30–35', 'T8': '35–30–20–2–20–30–35',
                 'T9': '35–30–30–15–2–15–30–30–35', 'G1': '35–20–2–20–35', 'G3': '35–30–15–2–15–30–35'}
LATENTE = {a: 'logística' if a == 'L2' else 'tanh' for a in ARQUITECTURAS}
OCULTAS = {a: 'logística' if a.startswith('G') else 'tanh' for a in ARQUITECTURAS}
ABREV = {'tanh': 'tanh', 'logística': 'logíst.'}
# Referencia que ya no se usa (latente identidad): escalón E9 de results/e1a2
REFERENCIA_E9 = 'E9'


def leer() -> pd.DataFrame:
    partes = []
    for costo in COSTOS:
        for arq in ARQUITECTURAS:
            # Grilla de 4 η (D62) y, donde el mejor quedó en 0,01, uno más allá del borde (0,03)
            for carpeta in (f'{costo}_{arq}', f'{costo}_{arq}_eta001', f'{costo}_{arq}_eta003'):
                if carpeta.endswith('eta003') and not (SALIDA / carpeta / 'corridas.csv').exists():
                    continue
                corridas = pd.read_csv(SALIDA / carpeta / 'corridas.csv')
                partes.append(corridas.assign(costo=costo))
    datos = pd.concat(partes, ignore_index=True)
    partes_id = datos.escalon.str.extract(r'^(?P<arquitectura>\w+) eta=(?P<eta>[\d.e-]+)$')
    assert partes_id.notna().all().all(), 'escalón con un id inesperado'
    return datos.assign(arquitectura=partes_id.arquitectura, eta=partes_id.eta.astype(float))


def agregar(datos: pd.DataFrame) -> pd.DataFrame:
    """Una fila por (costo, arquitectura, η). Las épocas son solo de las corridas que terminan
    con ≤ 1 px (logro): una que llegó y después lo perdió no cuenta."""
    datos = datos.assign(epoca_objetivo=datos.epoca_objetivo.where(datos.logro))
    tabla = datos.groupby(['costo', 'arquitectura', 'eta']).agg(
        semillas=('semilla', 'size'), logradas=('logro', 'sum'),
        total_px_media=('total_pixeles', 'mean'), total_px_desvio=('total_pixeles', 'std'),
        max_px_media=('max_pixeles', 'mean'), error_media=('error_final', 'mean'),
        epoca_mediana=('epoca_objetivo', 'median'), epoca_media=('epoca_objetivo', 'mean'),
        epoca_desvio=('epoca_objetivo', 'std'), n_pesos=('n_pesos', 'first'),
        segundos_media=('segundos', 'mean')).reset_index()
    # Conteo: exactamente SEMILLAS corridas por celda, sin repetir semilla
    assert (tabla.semillas == SEMILLAS).all(), tabla[tabla.semillas != SEMILLAS]
    assert not datos.duplicated(['costo', 'arquitectura', 'eta', 'semilla']).any()
    # Cada (costo, arquitectura) tiene los 4 η de la grilla, y a lo sumo uno más (0,03) por el borde
    etas = tabla.groupby(['costo', 'arquitectura']).eta.apply(lambda e: tuple(sorted(e)))
    assert etas.isin([tuple(ETAS), tuple(ETAS + [0.03])]).all(), etas
    return tabla


def mejor_eta(tabla: pd.DataFrame) -> pd.DataFrame:
    """Por (costo, arquitectura): más logradas, luego menos píxeles, luego menor mediana de época."""
    orden = tabla.assign(_epoca=tabla.epoca_mediana.fillna(np.inf)).sort_values(
        ['costo', 'arquitectura', 'logradas', 'total_px_media', '_epoca'],
        ascending=[True, True, False, True, True], kind='stable')
    return orden.groupby(['costo', 'arquitectura'], sort=False).head(1).drop(columns='_epoca').reset_index(drop=True)


def no_peor(valores: pd.Series, desvios: pd.Series) -> pd.Series:
    """Quiénes empatan con el mejor (el menor): diferencia ≤ el mayor de los dos desvíos."""
    mejor = valores.idxmin()
    return (valores - valores[mejor]) <= np.maximum(desvios.fillna(0), desvios.fillna(0)[mejor])


def elegir(mejores: pd.DataFrame) -> pd.DataFrame:
    """Aplica el criterio por pasos. Devuelve una fila por arquitectura con el paso en que quedó afuera."""
    bce = mejores[mejores.costo == 'bce'].set_index('arquitectura')
    ec = mejores[mejores.costo == 'ec'].set_index('arquitectura')
    tabla = pd.DataFrame({
        'eta_bce': bce.eta, 'logradas_bce': bce.logradas, 'logradas_ec': ec.logradas.reindex(bce.index),
        'total_px_bce': bce.total_px_media, 'total_px_bce_desvio': bce.total_px_desvio,
        'total_px_ec': ec.total_px_media.reindex(bce.index), 'total_px_ec_desvio': ec.total_px_desvio.reindex(bce.index),
        'eta_ec': ec.eta.reindex(bce.index),
        'epoca_mediana_bce': bce.epoca_mediana, 'epoca_media_bce': bce.epoca_media, 'epoca_desvio_bce': bce.epoca_desvio,
        'n_pesos': bce.n_pesos})
    tabla['queda_afuera_en'] = ''
    vivas = tabla.index.tolist()

    def descartar(quedan: list[str], paso: str) -> None:
        nonlocal vivas
        for a in vivas:
            if a not in quedan:
                tabla.loc[a, 'queda_afuera_en'] = paso
        vivas = [a for a in vivas if a in quedan]

    t = tabla.loc[vivas]
    descartar(t.index[t.logradas_bce == t.logradas_bce.max()].tolist(), '1 · semillas con BCE')
    t = tabla.loc[vivas]
    descartar(t.index[t.logradas_ec == t.logradas_ec.max()].tolist(), '2 · semillas con error cuadrático')
    t = tabla.loc[vivas]
    descartar(t.index[no_peor(t.total_px_bce, t.total_px_bce_desvio)].tolist(), '3 · píxeles con BCE')
    t = tabla.loc[vivas]
    if t.epoca_mediana_bce.notna().any():
        epocas = t.epoca_mediana_bce.fillna(np.inf)
        descartar(t.index[no_peor(epocas, t.epoca_desvio_bce)].tolist(), '4 · época en llegar con BCE')
    t = tabla.loc[vivas]
    descartar(t.index[t.n_pesos == t.n_pesos.min()].tolist(), '5 · cantidad de pesos')
    assert len(vivas) >= 1
    tabla['elegida'] = tabla.index == vivas[0]
    assert tabla.elegida.sum() == 1
    return tabla.reset_index().rename(columns={'index': 'arquitectura'})


def ordenar(mejores: pd.DataFrame) -> pd.DataFrame:
    """elegir() más la columna `orden`: 1 la elegida, 2 la que gana entre las que quedan, etc."""
    tabla = elegir(mejores).set_index('arquitectura')
    quedan = list(tabla.index)
    for puesto in range(1, len(quedan) + 1):
        ronda = elegir(mejores[mejores.arquitectura.isin(quedan)])
        ganadora = ronda[ronda.elegida].arquitectura.iloc[0]
        tabla.loc[ganadora, 'orden'] = puesto
        quedan.remove(ganadora)
    tabla['orden'] = tabla.orden.astype(int)
    assert tabla.loc[tabla.elegida, 'orden'].iloc[0] == 1
    return tabla.sort_values('orden').reset_index()


def figura(mejores: pd.DataFrame, datos: pd.DataFrame, ruta) -> None:
    """Tres paneles con el mejor η de cada arquitectura y costo: semillas, píxeles y época en llegar."""
    arqs = list(ARQUITECTURAS)
    x = np.arange(len(arqs))
    ancho = 0.38
    fig, ejes = plt.subplots(3, 1, figsize=(15, 14))
    for k, costo in enumerate(COSTOS):
        m = mejores[mejores.costo == costo].set_index('arquitectura').loc[arqs]
        dx = (k - 0.5) * ancho
        barras = ejes[0].bar(x + dx, m.logradas, ancho, color=COLOR_COSTO[costo], label=COSTOS[costo])
        for barra, eta in zip(barras, m.eta):
            ejes[0].text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.15, f'η = {eta:g}'.replace('.', ','),
                         ha='center', va='bottom', fontsize=7.5, color=TINTA_SECUNDARIA, rotation=90)
        abajo = np.minimum(m.total_px_desvio, m.total_px_media)
        ejes[1].bar(x + dx, m.total_px_media, ancho, yerr=[abajo, m.total_px_desvio], capsize=3, color=COLOR_COSTO[costo])
    # Época en que cada semilla llega al objetivo, con BCE y el mejor η de cada arquitectura
    mb = mejores[mejores.costo == 'bce'].set_index('arquitectura')
    for i, a in enumerate(arqs):
        sel = datos[(datos.costo == 'bce') & (datos.arquitectura == a) & (datos.eta == mb.loc[a, 'eta']) & datos.logro]
        jitter = np.linspace(-0.12, 0.12, len(sel)) if len(sel) > 1 else np.zeros(len(sel))
        ejes[2].scatter(i + jitter, sel.epoca_objetivo, s=18, color=COLOR_COSTO['bce'])
        if len(sel):
            ejes[2].hlines(sel.epoca_objetivo.median(), i - 0.25, i + 0.25, color=TINTA_SECUNDARIA, lw=1.5)
    # Solo el encoder: el decoder es el espejo
    rotulos = [f"{a}\n{ARQUITECTURAS[a].split('–2–')[0]}–2\n{ABREV[OCULTAS[a]]} / {ABREV[LATENTE[a]]}" for a in arqs]
    for ax in ejes:
        ax.set_xticks(x, rotulos, fontsize=9)
    ejes[0].set_ylim(0, SEMILLAS + 3)
    ejes[0].set_ylabel(f'Semillas con ≤ 1 píxel incorrecto\npor letra (de {SEMILLAS})')
    ejes[1].set_yscale('symlog', linthresh=1)
    ejes[1].set_ylim(0, 200)
    ejes[1].set_ylabel('Píxeles incorrectos en total, de 1120\n(media ± desvío; lineal hasta 1, log arriba)')
    ejes[2].set_ylabel('Época en que llega a ≤ 1 px por letra\n(entropía cruzada; línea: mediana)')
    ejes[2].set_ylim(0, 10000)
    fig.legend(*ejes[0].get_legend_handles_labels(), loc='upper center', ncol=2, fontsize=10, bbox_to_anchor=(0.5, 1.02))
    fig.supxlabel('Encoder de cada arquitectura (el decoder es el espejo) y activación de ocultas / latente. Cada barra o punto '
                  'usa el mejor η de esa arquitectura entre 0,0003, 0,001, 0,003 y 0,01 (Adam), con 10 semillas',
                  fontsize=10)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def pm(media: float, desvio: float, decimales: int = 1) -> str:
    return f'{media:.{decimales}f} ± {desvio:.{decimales}f}'.replace('.', ',')


def epoca(valor: float) -> str:
    """Mediana de épocas: entera, o con ,5 cuando cae entre dos (sin redondear al par)."""
    if pd.isna(valor):
        return '—'
    return f'{valor:.0f}' if valor == int(valor) else f'{valor:.1f}'.replace('.', ',')


def num(valor: float, decimales: int = 0) -> str:
    return '—' if pd.isna(valor) else f'{valor:.{decimales}f}'.replace('.', ',')


def texto(todas: pd.DataFrame, eleccion: pd.DataFrame) -> str:
    lineas = ['| Arquitectura | Ocultas | Latente | Pesos | Costo | η | Semillas con ≤ 1 px | Px mal en total | Máx. px por letra (media) | Época en que llega (mediana; media ± desvío) | Tiempo (s) |',
              '|---|---|---|---|---|---|---|---|---|---|---|']
    for f in todas.itertuples():
        lineas.append(f'| {f.arquitectura} {ARQUITECTURAS[f.arquitectura]} | {OCULTAS[f.arquitectura]} | {LATENTE[f.arquitectura]} | {f.n_pesos} | '
                      f'{COSTOS[f.costo]} | {num(f.eta, 4).rstrip("0")} | {f.logradas} de {f.semillas} | '
                      f'{pm(f.total_px_media, f.total_px_desvio)} | {num(f.max_px_media, 1)} | '
                      f'{epoca(f.epoca_mediana)}' + (f' ({pm(f.epoca_media, f.epoca_desvio, 0)})' if f.logradas > 1 else '') +
                      f' | {num(f.segundos_media, 1)} |')
    pasos = ['| Puesto | Arquitectura | η (BCE) | Semillas BCE | Semillas error cuadrático | Px BCE | Px error cuadrático (η) | Época BCE (mediana; media ± desvío) | Pesos | Queda afuera en |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for f in eleccion.itertuples():
        pasos.append(f'| {f.orden} | {f.arquitectura} {ARQUITECTURAS[f.arquitectura]} ({OCULTAS[f.arquitectura]}/{LATENTE[f.arquitectura]}) | {num(f.eta_bce, 4).rstrip("0")} | {f.logradas_bce} de {SEMILLAS} | {f.logradas_ec} de {SEMILLAS} | '
                     f'{pm(f.total_px_bce, f.total_px_bce_desvio)} | {pm(f.total_px_ec, f.total_px_ec_desvio)} ({num(f.eta_ec, 4).rstrip("0")}) | '
                     f'{epoca(f.epoca_mediana_bce)} ({pm(f.epoca_media_bce, f.epoca_desvio_bce, 0) if f.logradas_bce > 1 else "—"}) | '
                     f'{f.n_pesos} | {f.queda_afuera_en or "**elegida**"} |')
    elegida = eleccion[eleccion.elegida].iloc[0]
    e9 = pd.read_csv(RAIZ / 'results' / 'e1a2' / 'resumen.csv').set_index('escalon').loc[REFERENCIA_E9]
    ec_logradas = int(todas[todas.costo == 'ec'].logradas.sum())
    return f"""## 1a.1 rehecha · Arquitecturas con latente no lineal (e1a2_arquitecturas)

Pedido de la cátedra (2026-11-17): el latente no puede ser lineal (D57), y hay que variar capas, dimensiones y activaciones (D59). {len(ARQUITECTURAS)} arquitecturas × {len(ETAS)} η de Adam (0,0003; 0,001; 0,003; 0,01; el último agregado por D62) × 2 costos × {SEMILLAS} semillas (0 a 9), 10000 épocas, batch de 32, ocultas tanh (logísticas en G1 y G3), salida logística, init ±0,1. Datos en `results/e1a2_arquitecturas/`, figura `fig_arquitecturas.png`. Media ± desvío entre semillas; "época en que llega" es entre las corridas que terminan con ≤ 1 px.

### Todas las celdas (`todas.csv`)

{chr(10).join(lineas)}

### Elección (`eleccion.csv`, criterio de D57; "queda afuera en" es contra la elegida, "puesto" repite el criterio sin las anteriores; si dos empatan en todos los pasos, como L2 y T2, el orden entre ellas es alfabético)

{chr(10).join(pasos)}

- Elegida: **{elegida.arquitectura} {ARQUITECTURAS[elegida.arquitectura]}**, latente {LATENTE[elegida.arquitectura]}, entropía cruzada, Adam con η = {num(elegida.eta_bce, 4).rstrip('0')}: {elegida.logradas_bce} de {SEMILLAS} semillas con ≤ 1 px, {num(elegida.total_px_bce, 1)} px mal en total, llega en la época {epoca(elegida.epoca_mediana_bce)} (mediana).
- Con error cuadrático llegan al objetivo {ec_logradas} de {len(todas[todas.costo == 'ec']) * SEMILLAS} corridas.
- Referencia que ya no se usa (latente identidad, E9 de `results/e1a2`, 35–20–2–20–35, BCE, η = 0,001): {int(e9.logradas)} de {int(e9.semillas)} semillas, {pm(e9.total_px_media, e9.total_px_desvio)} px, época {epoca(e9.epoca_objetivo_mediana)} (mediana).
"""


def main() -> None:
    aplicar_estilo()
    datos = leer()
    todas = agregar(datos)
    mejores = mejor_eta(todas)
    eleccion = ordenar(mejores)
    todas.to_csv(SALIDA / 'todas.csv', index=False)
    mejores.to_csv(SALIDA / 'mejor_eta.csv', index=False)
    eleccion.to_csv(SALIDA / 'eleccion.csv', index=False)
    figura(mejores, datos, SALIDA / 'fig_arquitecturas.png')
    escribir_seccion('e1a2_arquitecturas', texto(todas, eleccion))
    print(eleccion.to_string(index=False))


if __name__ == '__main__':
    main()
