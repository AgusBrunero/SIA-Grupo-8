/**
 * Genera el respaldo en PowerPoint de la presentación del TP5 (la principal es presentacion.html).
 *
 *     .venv/bin/python -m experiments.presentacion     (desde TP5/: números, figuras y el HTML)
 *     node docs/presentacion/build.js
 *
 * El contenido de las láminas está en laminas.json, compartido con el HTML. Los títulos y los
 * cierres salen de docs/CONCLUSIONES.md. Las notas son un borrador del guion (R14).
 */
'use strict';

const fs = require('fs');
const path = require('path');
const PptxGenJS = require('pptxgenjs');
const h = require('./helpers');

const SALIDA = path.join(__dirname, 'TP5_Grupo8.pptx');
const LAMINAS = JSON.parse(fs.readFileSync(path.join(__dirname, 'laminas.json'), 'utf8'));

function main() {
  const pptx = new PptxGenJS();
  pptx.layout = 'LAYOUT_WIDE';
  pptx.author = 'Grupo 8 · SIA 2026';
  pptx.title = 'TP5 · Deep Learning';
  for (const lamina of LAMINAS) h[lamina.tipo](pptx, lamina);

  const fallas = h.auditar();
  fs.writeFileSync(path.join(__dirname, 'auditoria.json'), JSON.stringify(
    h.auditoria.map((l) => ({ lamina: l.numero, tipo: l.tipo, texto: l.texto,
                              numeros: l.claves.map((c) => ({ clave: c, ...h.MANIFEST[c] })) })), null, 1));
  if (fallas.length) {
    console.error(fallas.join('\n'));
    process.exit(1);
  }
  return pptx.writeFile({ fileName: SALIDA }).then(() => console.log(`${h.auditoria.length} láminas → ${SALIDA}`));
}

main();
