"""Presentación en HTML (sesión 8): un único archivo, una lámina por pantalla, sin controles.

Arma docs/presentacion/presentacion.html con
  - laminas.json     el contenido de las 23 láminas (el mismo que usa el respaldo en PowerPoint);
  - manifest.json    los números en pantalla, cada uno con el CSV del que sale (regla R10);
  - plantilla.html   el estilo y la navegación con teclado;
  - vendor/katex     KaTeX embebido, para que las fórmulas se vean sin internet.
Las figuras se embeben en el archivo. También escribe guion_borrador.md con las notas de cada
lámina, para que quien presenta las reescriba (R14).

Antes de escribir audita las reglas del plan de presentación del TP4 que se pueden verificar a
máquina: R1 (≤ 35 palabras), R15 (≤ 3 números), R10 (ningún número escrito a mano), R4 (sin verbos
de interfaz) y R8 (sin identificadores de código).

Uso, desde TP5/ (lo llama experiments.presentacion):
    .venv/bin/python -m experiments.presentacion_html
"""
import base64
import html
import json
import re

from experiments.comun import RAIZ

CARPETA = RAIZ / 'docs' / 'presentacion'
TOKEN = re.compile(r'\{\{(\w+)\}\}')
VERBOS = re.compile(r'\b(elegí|tocá|pasá|entrenalo|cambiá|pintá|mové|hacé clic)\b', re.IGNORECASE)
CODIGO = re.compile(r'\w+\.(py|csv|json)\b|\b[a-z]+_[a-z_]+\b|--')
MAX_PALABRAS, MAX_NUMEROS = 35, 3


def textos_en_pantalla(lamina: dict) -> list[str]:
    """El texto que se ve en la lámina, sin contar la figura ni las notas."""
    campos = {'portada': ('titulo', 'subtitulo', 'pie'), 'resultado': ('titulo', 'lectura'),
              'teoria': ('titulo', 'formulas', 'lectura'), 'cierre': ('titulo', 'frases')}[lamina['tipo']]
    textos: list[str] = []
    for campo in campos:
        valor = lamina.get(campo)
        if valor:
            textos += valor if isinstance(valor, list) else [valor]
    return textos


def resolver(texto: str, numeros: dict) -> str:
    return TOKEN.sub(lambda m: numeros[m.group(1)]['texto'], texto)


def auditar(laminas: list[dict], numeros: dict) -> list[str]:
    fallas = []
    for n, lamina in enumerate(laminas, start=1):
        crudo = ' '.join(textos_en_pantalla(lamina))
        claves = TOKEN.findall(crudo)
        faltan = sorted(set(claves) - set(numeros))
        if faltan:
            fallas.append(f'lámina {n}: claves que no están en manifest.json: {faltan}')
            continue
        texto = resolver(crudo, numeros)
        palabras = sum(1 for p in texto.split() if re.search(r'\w', p))
        if palabras > MAX_PALABRAS:
            fallas.append(f'lámina {n}: {palabras} palabras (R1)')
        if len(claves) > MAX_NUMEROS:
            fallas.append(f'lámina {n}: {len(claves)} números (R15)')
        if lamina['tipo'] not in ('teoria', 'portada') and re.search(r'\d', TOKEN.sub('', crudo)):
            fallas.append(f'lámina {n}: número escrito a mano (R10)')
        if VERBOS.search(texto):
            fallas.append(f'lámina {n}: verbo de interfaz (R4)')
        if CODIGO.search(texto):
            fallas.append(f'lámina {n}: identificador de código (R8)')
    return fallas


def imagen(relativa: str) -> str:
    datos = (RAIZ / relativa).read_bytes()
    return 'data:image/png;base64,' + base64.b64encode(datos).decode('ascii')


def katex_embebido() -> str:
    """KaTeX con las fuentes woff2 como data URI (misma técnica que el recuperatorio del TP4)."""
    carpeta = CARPETA / 'vendor' / 'katex'
    css = (carpeta / 'katex.min.css').read_text(encoding='utf-8')
    css = re.sub(r',url\(fonts/[^)]+\.(?:woff|ttf)\) format\("[^"]+"\)', '', css)
    css = re.sub(r'url\(fonts/([^)]+\.woff2)\)',
                 lambda m: 'url(data:font/woff2;base64,' + base64.b64encode((carpeta / 'fonts' / m.group(1)).read_bytes()).decode('ascii') + ')',
                 css)
    assert 'url(fonts/' not in css, 'quedó una fuente de KaTeX sin embeber'
    return f"<style>{css}</style>\n<script>{(carpeta / 'katex.min.js').read_text(encoding='utf-8')}</script>"


def seccion(n: int, lamina: dict, numeros: dict) -> str:
    def t(texto: str) -> str:
        return html.escape(resolver(texto, numeros))

    tipo, modelo = lamina['tipo'], lamina.get('modelo', '')
    presenta = lamina.get('presenta', '')
    prefijo = f'<b>Presenta {html.escape(presenta)}.</b> ' if presenta else ''
    notas = f'<aside class="notas">{prefijo}{t(lamina["notas"])}</aside>' if lamina.get('notas') else ''
    quien = f'<span class="presenta">{html.escape(presenta)}</span> · ' if presenta else ''
    pagina = f'<div class="pagina">{quien}{n}</div>'
    if tipo == 'portada':
        cuerpo = (f'<div><h1>{t(lamina["titulo"])}</h1><p class="sub">{t(lamina["subtitulo"])}</p>'
                  '<div class="barras"><i style="background:var(--AE)"></i><i style="background:var(--DAE)"></i>'
                  f'<i style="background:var(--VAE)"></i></div><p class="pie">{t(lamina["pie"])}</p></div>')
    elif tipo == 'cierre':
        cuerpo = f'<div><h2>{t(lamina["titulo"])}</h2>' + ''.join(f'<p>{t(frase)}</p>' for frase in lamina['frases']) + '</div>'
    else:
        if tipo == 'teoria':
            figura = (f'<img src="{imagen(lamina["esquema"])}" alt="">'
                      + ''.join(f'<div class="tex" data-tex="{html.escape(tex)}"></div>' for tex in lamina['formulas_tex']))
        else:
            figura = f'<img src="{imagen(lamina["figura"])}" alt="">'
        lectura = f'<p class="lectura">{t(lamina["lectura"])}</p>' if lamina.get('lectura') else '<p class="lectura"></p>'
        cuerpo = (f'<div class="item">{t(lamina.get("item", ""))}</div><h2>{t(lamina["titulo"])}</h2>'
                  f'<div class="fig">{figura}</div>{lectura}')
    return f'<section class="slide {tipo} {modelo}" id="l{n}">{cuerpo}{pagina}{notas}</section>'


def guion(laminas: list[dict], numeros: dict) -> str:
    lineas = ['# TP5 · Borrador del guion', '',
              'Generado desde `laminas.json`. Es un borrador: quien presenta cada lámina lo reescribe con sus palabras.', '']
    for n, lamina in enumerate(laminas, start=1):
        quien = f' · presenta {lamina["presenta"]}' if lamina.get('presenta') else ''
        lineas += [f'## {n}. {resolver(lamina["titulo"], numeros)}{quien}', '', resolver(lamina.get('notas', ''), numeros), '']
    return '\n'.join(lineas)


def main() -> None:
    laminas = json.loads((CARPETA / 'laminas.json').read_text(encoding='utf-8'))
    numeros = json.loads((CARPETA / 'manifest.json').read_text(encoding='utf-8'))['numeros']
    fallas = auditar(laminas, numeros)
    if fallas:
        raise SystemExit('\n'.join(fallas))
    plantilla = (CARPETA / 'plantilla.html').read_text(encoding='utf-8')
    assert '<!--__KATEX__-->' in plantilla and '<!--__LAMINAS__-->' in plantilla
    secciones = '\n'.join(seccion(n, lamina, numeros) for n, lamina in enumerate(laminas, start=1))
    salida = plantilla.replace('<!--__KATEX__-->', katex_embebido()).replace('<!--__LAMINAS__-->', secciones)
    (CARPETA / 'presentacion.html').write_text(salida, encoding='utf-8')
    (CARPETA / 'guion_borrador.md').write_text(guion(laminas, numeros), encoding='utf-8')
    print(f'{len(laminas)} láminas → {CARPETA / "presentacion.html"} ({len(salida) / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
