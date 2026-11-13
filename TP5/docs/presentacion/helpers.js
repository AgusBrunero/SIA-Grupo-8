/**
 * Plantilla de la presentación del TP5: cuatro layouts (portada, teoría, resultado, cierre)
 * y la auditoría de las reglas R1, R4, R8, R10 y R15 del plan de presentación del TP4.
 *
 * Ningún número se escribe a mano en una lámina: el texto usa {{clave}} y el valor sale de
 * manifest.json, que genera experiments/presentacion.py desde los CSV de results/.
 */
'use strict';

const fs = require('fs');
const path = require('path');
const sizeOf = require('image-size');

const RAIZ = path.resolve(__dirname, '..', '..');
const MANIFEST = JSON.parse(fs.readFileSync(path.join(__dirname, 'manifest.json'), 'utf8')).numeros;

// Los mismos colores que experiments/comun.py: un color por modelo en todo el TP
const COLOR = { tinta: '0B0B0B', tenue: '52514E', linea: 'C3C2B7', papel: 'FFFFFF', fondo: 'F4F3EE',
                AE: '2A78D6', DAE: 'EB6834', VAE: '1BAF7A' };
const FUENTE = 'Helvetica Neue';
const ANCHO = 13.333, ALTO = 7.5, MARGEN = 0.6;

const auditoria = [];      // una entrada por lámina: {numero, texto, claves}

/** Reemplaza {{clave}} por el texto del manifiesto y anota qué claves usó la lámina. */
function resolver(texto, claves) {
  return texto.replace(/\{\{(\w+)\}\}/g, (_, clave) => {
    if (!MANIFEST[clave]) throw new Error(`la clave {{${clave}}} no está en manifest.json`);
    claves.push(clave);
    return MANIFEST[clave].texto;
  });
}

function figura(relativa) {
  const ruta = path.isAbsolute(relativa) ? relativa : path.join(RAIZ, relativa);
  if (!fs.existsSync(ruta)) throw new Error(`falta la figura ${ruta}`);
  return ruta;
}

/** Ubica una imagen dentro de una caja conservando su proporción, centrada. */
function encajar(ruta, caja) {
  const { width, height } = sizeOf(ruta);
  const escala = Math.min(caja.w / width, caja.h / height);
  const w = width * escala, h = height * escala;
  return { path: ruta, x: caja.x + (caja.w - w) / 2, y: caja.y + (caja.h - h) / 2, w, h };
}

function nueva(pptx, datos, textos) {
  const slide = pptx.addSlide();
  slide.background = { color: COLOR.papel };
  const claves = [];
  const resueltos = textos.map((t) => resolver(t, claves));
  auditoria.push({ numero: auditoria.length + 1, tipo: datos.tipo, crudo: textos.join(' '), texto: resueltos.join(' '), claves });
  if (datos.notas) slide.addNotes(resolver(datos.notas, []));
  if (datos.item) {
    slide.addText(datos.item, { x: MARGEN, y: 0.32, w: 3, h: 0.3, fontFace: FUENTE, fontSize: 12, bold: true,
                                color: COLOR[datos.modelo] || COLOR.tenue });
  }
  slide.addText(String(auditoria.length), { x: ANCHO - 1.1, y: ALTO - 0.45, w: 0.6, h: 0.3, fontFace: FUENTE, fontSize: 10,
                                            color: COLOR.tenue, align: 'right' });
  return { slide, resueltos };
}

function titulo(slide, texto, modelo) {
  slide.addShape('rect', { x: 0, y: 0, w: 0.18, h: ALTO, fill: { color: COLOR[modelo] || COLOR.tenue }, line: { color: COLOR[modelo] || COLOR.tenue } });
  slide.addText(texto, { x: MARGEN, y: 0.62, w: ANCHO - 2 * MARGEN, h: 1.15, fontFace: FUENTE, fontSize: 28, bold: true,
                         color: COLOR.tinta, valign: 'top', fit: 'shrink' });
}

function portada(pptx, d) {
  const { slide, resueltos } = nueva(pptx, { tipo: 'portada', notas: d.notas }, [d.titulo, d.subtitulo, d.pie]);
  slide.background = { color: COLOR.fondo };
  slide.addText(resueltos[0], { x: MARGEN + 0.4, y: 2.3, w: 11, h: 1.0, fontFace: FUENTE, fontSize: 44, bold: true, color: COLOR.tinta });
  slide.addText(resueltos[1], { x: MARGEN + 0.4, y: 3.3, w: 11, h: 0.7, fontFace: FUENTE, fontSize: 24, color: COLOR.tenue });
  slide.addText(resueltos[2], { x: MARGEN + 0.4, y: 5.6, w: 11, h: 0.5, fontFace: FUENTE, fontSize: 16, color: COLOR.tenue });
  ['AE', 'DAE', 'VAE'].forEach((m, i) => slide.addShape('rect', { x: MARGEN + 0.4 + i * 1.3, y: 4.5, w: 1.1, h: 0.12,
                                                                  fill: { color: COLOR[m] }, line: { color: COLOR[m] } }));
}

/** Lámina de resultado: título-afirmación, una figura protagonista y una línea de lectura. */
function resultado(pptx, d) {
  const { slide, resueltos } = nueva(pptx, { tipo: 'resultado', ...d }, [d.titulo, d.lectura]);
  titulo(slide, resueltos[0], d.modelo);
  slide.addImage(encajar(figura(d.figura), { x: MARGEN, y: 1.85, w: ANCHO - 2 * MARGEN, h: 4.35 }));
  slide.addText(resueltos[1], { x: MARGEN, y: 6.3, w: ANCHO - 2 * MARGEN - 0.8, h: 0.75, fontFace: FUENTE, fontSize: 18,
                                color: COLOR.tinta, valign: 'top' });
}

/** Lámina de teoría: un esquema (figura de experiments/presentacion.py) más las fórmulas, en notación de la cátedra. */
function teoria(pptx, d) {
  const { slide, resueltos } = nueva(pptx, { tipo: 'teoria', ...d }, [d.titulo, ...d.formulas, ...(d.lectura ? [d.lectura] : [])]);
  titulo(slide, resueltos[0], d.modelo);
  slide.addImage(encajar(figura(d.esquema), { x: MARGEN, y: 2.1, w: ANCHO - 2 * MARGEN, h: 1.7 }));
  d.formulas.forEach((_, i) => {
    slide.addText(resueltos[1 + i], { x: MARGEN + 0.6, y: 4.15 + i * 0.72, w: ANCHO - 2 * MARGEN - 1.2, h: 0.62, fontFace: FUENTE,
                                      fontSize: 22, color: COLOR.tinta, align: 'center' });
  });
  if (d.lectura) {
    slide.addText(resueltos[resueltos.length - 1], { x: MARGEN, y: 6.3, w: ANCHO - 2 * MARGEN - 0.8, h: 0.75, fontFace: FUENTE,
                                                     fontSize: 18, color: COLOR.tinta, valign: 'top' });
  }
}

/** Cierre de parte: la consecuencia en grande y hasta tres frases cortas. */
function cierre(pptx, d) {
  const { slide, resueltos } = nueva(pptx, { tipo: 'cierre', ...d }, [d.titulo, ...d.frases]);
  slide.background = { color: COLOR.fondo };
  slide.addShape('rect', { x: 0, y: 0, w: 0.18, h: ALTO, fill: { color: COLOR[d.modelo] || COLOR.tenue }, line: { color: COLOR[d.modelo] || COLOR.tenue } });
  slide.addText(resueltos[0], { x: MARGEN + 0.4, y: 1.5, w: ANCHO - 2 * MARGEN - 0.8, h: 2.0, fontFace: FUENTE, fontSize: 36, bold: true,
                                color: COLOR.tinta, valign: 'middle' });
  d.frases.forEach((_, i) => {
    slide.addText(resueltos[1 + i], { x: MARGEN + 0.4, y: 3.9 + i * 0.75, w: ANCHO - 2 * MARGEN - 0.8, h: 0.65, fontFace: FUENTE,
                                      fontSize: 22, color: COLOR.tenue });
  });
}

/**
 * Auditoría de las reglas que se pueden verificar a máquina. Devuelve la lista de fallas.
 *   R1  ≤ 35 palabras en pantalla (título y texto; no cuenta la figura)
 *   R15 ≤ 3 números en el texto
 *   R10 todo número viene de manifest.json: no hay dígitos escritos a mano (salvo en teoría y portada)
 *   R4  sin verbos de interfaz      R8  sin identificadores de código
 */
function auditar() {
  const fallas = [];
  for (const l of auditoria) {
    const palabras = l.texto.split(/\s+/).filter((p) => /[\p{L}\p{N}]/u.test(p)).length;
    if (palabras > 35) fallas.push(`lámina ${l.numero}: ${palabras} palabras (R1)`);
    if (l.claves.length > 3) fallas.push(`lámina ${l.numero}: ${l.claves.length} números (R15)`);
    const sinClaves = l.crudo.replace(/\{\{\w+\}\}/g, '');
    if (!['teoria', 'portada'].includes(l.tipo) && /\d/.test(sinClaves)) fallas.push(`lámina ${l.numero}: número escrito a mano (R10)`);
    if (/\b(elegí|tocá|pasá|entrenalo|cambiá|pintá|mové|hacé clic)\b/i.test(l.texto)) fallas.push(`lámina ${l.numero}: verbo de interfaz (R4)`);
    if (/\w+\.(py|csv|json)\b|\b[a-z]+_[a-z_]+\b|--/.test(l.texto)) fallas.push(`lámina ${l.numero}: identificador de código (R8)`);
  }
  return fallas;
}

module.exports = { portada, resultado, teoria, cierre, auditar, auditoria, MANIFEST };
