# TP5 · Resultados

Observaciones factuales generadas por los scripts de `experiments/` a partir de los CSV de `results/`. No editar a mano entre los marcadores: se regenera. Las conclusiones van en `CONCLUSIONES.md` y las escribe el grupo.

<!-- e1a2:inicio -->
## 1a.2 · Arquitecturas y técnicas de optimización (e1a2)

### Escalera oficial (`results/e1a2/`)

10 escalones × 10 semillas (0 a 9), 10000 épocas en los escalones entrenados por épocas, datos en {0, 1}, salida logística, latente de 2 dimensiones. Media ± desvío entre semillas. "Época en que llega" es la mediana entre las corridas que terminan con ≤ 1 px.

| Escalón | Respecto de | Qué cambia | Semillas con ≤ 1 px | Máx. px por patrón | Px mal en total | Error final | Época en que llega (mediana) | Tiempo (s) |
|---|---|---|---|---|---|---|---|---|
| E1 | — | Base: 35-2-35, logística en todas las capas, error cuadrático, GD en batch | 0 de 10 | 18,9 ± 2,5 | 210,0 ± 11,0 | 77,93 ± 1,81 | — | 1,1 |
| E2 | E1 | Arquitectura: una capa oculta de 20 (35-20-2-20-35) | 0 de 10 | 9,2 ± 2,9 | 91,4 ± 41,8 | 39,04 ± 12,84 | — | 1,9 |
| E3 | E2 | Activación del latente: identidad | 0 de 10 | 2,7 ± 1,9 | 11,7 ± 3,9 | 7,03 ± 1,63 | — | 1,7 |
| E4 | E3 | Activación oculta: tanh | 4 de 10 | 1,6 ± 0,5 | 10,0 ± 2,1 | 5,22 ± 1,06 | 1727 | 1,5 |
| E5 | E4 | Optimizador: momentum (alfa 0,9) | 0 de 10 | 2,5 ± 0,5 | 16,7 ± 5,2 | 8,54 ± 2,59 | — | 1,5 |
| E6 | E4 | eta adaptativo sobre GD | 0 de 10 | 2,2 ± 0,4 | 11,4 ± 3,7 | 6,50 ± 1,81 | — | 1,6 |
| E7 | E4 | Optimizador: RMSProp | 1 de 10 | 2,3 ± 0,8 | 14,6 ± 3,3 | 7,30 ± 1,64 | 789 | 1,7 |
| E8 | E4 | Optimizador: Adam | 0 de 10 | 3,0 ± 1,3 | 18,4 ± 3,5 | 9,22 ± 1,75 | — | 1,8 |
| E9 | E8 | Pérdida: BCE | 10 de 10 | 0,0 ± 0,0 | 0,0 ± 0,0 | 0,61 ± 0,26 | 2205 | 2,0 |
| E10 | E1 | Optimizador: Powell (scipy) | 0 de 10 | 29,5 ± 4,4 | 339,8 ± 25,8 | 108,45 ± 6,84 | — | 0,8 |

Figuras: `fig_logro.png` (fracción de semillas por escalón), `fig_pixeles.png` (píxeles incorrectos por corrida), `fig_curvas.png` (error por época; el panel de E9 es BCE y no es comparable en valor con los de error cuadrático), `fig_fallas.png` (qué patrones fallan en cada escalón). Tablas: `resumen.csv`, `fallas.csv`.

Observaciones:

- Escalones con alguna semilla que termina con ≤ 1 px por patrón: E4 (4 de 10), E7 (1 de 10), E9 (10 de 10). El resto, 0 de 10.
- E9 (BCE) termina con 0 píxeles incorrectos en las 10 semillas; llega a ≤ 1 px entre las épocas 2024 y 4789 (mediana 2205).
- En los escalones con error cuadrático entrenados por épocas (E1 a E8), 1277 de 3842 píxeles incorrectos (33,2 %) tienen |O − X| > 0,99. De E4 a E8: 709 de 711.
- Píxeles incorrectos en total: E1 210,0 ± 11,0; E2 91,4 ± 41,8; E3 11,7 ± 3,9; E4 10,0 ± 2,1.
- Entre E4, E5, E6, E7 y E8 (mismo modelo, distinto optimizador, error cuadrático) el total de píxeles incorrectos va de 10,0 ± 2,1 (E4) a 18,4 ± 3,5 (E8).
- Corridas que en alguna época tuvieron ≤ 1 px y terminaron con más: E6: 1.
- E10 (Powell sobre 35–2–35, 177 pesos): 0 de 10. scipy informa convergencia en las 10 corridas, con entre 2 y 28 iteraciones y entre 3049 y 51051 evaluaciones de la pérdida (tope: 200000); tarda entre 0,13 y 2,24 s. Error final 108,45 ± 6,84, contra 77,93 ± 1,81 de E1 (misma red, GD). 5 de 10 corridas se detienen en 2 iteraciones con error 114,48.

Patrones que más fallan (más de 1 px mal) y sus vecinos, en los escalones con error cuadrático y capa oculta tanh:

- **E4:** '}' en 4 de 10 (vecino en el latente más frecuente: 'j'; más cercano por Hamming: '`', a 8 px); 'q' en 2 de 10 (vecino en el latente más frecuente: 'v'; más cercano por Hamming: 'a', a 10 px); 'DEL' en 2 de 10 (vecino en el latente más frecuente: 'd'; más cercano por Hamming: 'a', a 19 px).
- **E5:** 'q' en 6 de 10 (vecino en el latente más frecuente: 'a'; más cercano por Hamming: 'a', a 10 px); 'v' en 4 de 10 (vecino en el latente más frecuente: 'o'; más cercano por Hamming: 'u', a 5 px); '}' en 4 de 10 (vecino en el latente más frecuente: 'j'; más cercano por Hamming: '`', a 8 px).
- **E6:** '}' en 4 de 10 (vecino en el latente más frecuente: 'j'; más cercano por Hamming: '`', a 8 px); 's' en 3 de 10 (vecino en el latente más frecuente: 'd'; más cercano por Hamming: 'e', a 8 px); 'j' en 2 de 10 (vecino en el latente más frecuente: 'c'; más cercano por Hamming: '}', a 8 px).
- **E7:** 'DEL' en 3 de 10 (vecino en el latente más frecuente: 'g'; más cercano por Hamming: 'a', a 19 px); 'f' en 3 de 10 (vecino en el latente más frecuente: '{'; más cercano por Hamming: '{', a 6 px); '}' en 3 de 10 (vecino en el latente más frecuente: 'j'; más cercano por Hamming: '`', a 8 px).
- **E8:** 's' en 4 de 10 (vecino en el latente más frecuente: 'd'; más cercano por Hamming: 'e', a 8 px); 'x' en 4 de 10 (vecino en el latente más frecuente: 'c'; más cercano por Hamming: 'z', a 8 px); 'v' en 4 de 10 (vecino en el latente más frecuente: 'u'; más cercano por Hamming: 'u', a 5 px).

### Hallazgos del piloto (`results/e1a2_piloto/`, `results/e1a2_piloto_eta/`)

Piloto para elegir η por optimizador: 37 variantes × 2 semillas (las 100 y 101, distintas de las oficiales), 10000 épocas, error cuadrático y salida logística. Más 8 combinaciones de las constantes del η adaptativo.

- **Ninguna de las 74 corridas del piloto de η llega a ≤ 1 píxel por patrón** (0 de 74). El mínimo de máximo-por-patrón es 2 px; la corrida con menos píxeles mal en total tiene 5 (E4_gd_0.01, semilla 101).
- En las 24 corridas con RMSProp o Adam, el error final dividido por la cantidad de píxeles incorrectos vale entre 0,500 y 0,578 (mediana 0,500). Un píxel con la salida en el valor opuesto aporta ½·1² = 0,5 al error cuadrático.
- En esas corridas, el 99,2 % de los píxeles incorrectos (499 de 503) tiene |O − X| > 0,99, y la mediana entre corridas de la θ′ de salida más grande entre sus píxeles incorrectos es 4.8e-06. Con error cuadrático el delta de salida es (ζ − O)·θ′(h) (Clase 11, diap. 40–43).
- En el piloto del η adaptativo, 3 de 16 corridas llegan a ≤ 1 px; las 3 son de combinaciones con k′ = 3.

### Hallazgos del diagnóstico (`results/e1a2_diagnostico/`)

Mismas 2 semillas del piloto, 10000 épocas. Cada variante cambia lo indicado sobre la referencia.

| Variante (cambio sobre la referencia) | Semillas con ≤ 1 px | Px mal en total, por semilla | Px mal saturados |
|---|---|---|---|
| Referencia: 35–20–2–20–35, Adam η = 0,001, error cuadrático | 0 de 2 | 15 y 28 | 43 de 43 |
| BCE | 2 de 2 | 0 y 0 | 0 de 0 |
| Salida identidad | 0 de 2 | 9 y 4 | 0 de 13 |
| Más profunda: 35–20–10–2 | 0 de 2 | 11 y 16 | 27 de 27 |
| Más ancha: 35–30–2 | 0 de 2 | 11 y 19 | 30 de 30 |
| BCE y 35–20–10–2 | 2 de 2 | 0 y 0 | 0 de 0 |
| BCE con GD η = 0,01 | 0 de 2 | 256 y 425 | 280 de 681 |
| BCE con Adam η = 0,003 | 2 de 2 | 0 y 0 | 0 de 0 |
| BCE y 35–30–2 | 2 de 2 | 0 y 0 | 0 de 0 |

- Con error cuadrático, ninguna de las 8 corridas llega a ≤ 1 px (0 de 8), tampoco con más capas, más neuronas o salida identidad.
- Con BCE y Adam, 8 de 8 corridas llegan, todas con 0 píxeles incorrectos. Con BCE el delta de salida es 2β(O − ζ) (derivación nuestra, verificada en `tests/test_activaciones_perdidas.py`).
- BCE con GD y η = 0,01 no llega (0 de 2) y termina con error 594 y 2867.
- Son 2 semillas por variante: sirvió para decidir la escalera, no es un resultado.
<!-- e1a2:fin -->

<!-- e1a3_e1a4:inicio -->
## 1a.3 y 1a.4 · Espacio latente y generación (e1a3_e1a4)

Modelo: escalón E9 de e1a2, 10000 épocas, semillas 0, 1, 2, 3, 4. Datos en `results/e1a3_e1a4/`.

### Entrenamiento

- Píxeles incorrectos al reconstruir los 32 patrones: semilla 0: máximo 0 px, total 0 px; semilla 1: máximo 0 px, total 0 px; semilla 2: máximo 0 px, total 0 px; semilla 3: máximo 0 px, total 0 px; semilla 4: máximo 0 px, total 0 px. Las 5 de 5 semillas cumplen ≤ 1 px por patrón.

### 1a.3 · Latente (`fig_latente_ae.png`, `latente.csv`)

- La posición de cada patrón cambia de una semilla a otra. Para comparar sin depender de rotaciones, reflejos ni traslaciones se usan las 496 distancias entre pares de patrones en el latente (`estabilidad.csv`).
- Correlación de Pearson de esas distancias entre semillas: semilla 0 y semilla 1: 0,47, semilla 0 y semilla 2: 0,40, semilla 0 y semilla 3: 0,70, semilla 0 y semilla 4: 0,34, semilla 1 y semilla 2: 0,30, semilla 1 y semilla 3: 0,64, semilla 1 y semilla 4: 0,40, semilla 2 y semilla 3: 0,39, semilla 2 y semilla 4: 0,14, semilla 3 y semilla 4: 0,45.
- Patrones (de 32) cuyo vecino más cercano en el latente es el mismo en las dos semillas: semilla 0 y semilla 1: 5, semilla 0 y semilla 2: 8, semilla 0 y semilla 3: 7, semilla 0 y semilla 4: 7, semilla 1 y semilla 2: 9, semilla 1 y semilla 3: 9, semilla 1 y semilla 4: 3, semilla 2 y semilla 3: 10, semilla 2 y semilla 4: 3, semilla 3 y semilla 4: 2.
- Correlación entre la distancia en el latente y la distancia de Hamming entre los patrones: semilla 0: 0,45, semilla 1: 0,40, semilla 2: 0,45, semilla 3: 0,54, semilla 4: 0,33.

### Línea base: PCA 2D con SVD (`fig_latente_pca.png`, `pca.csv`)

- Las dos primeras componentes explican el 15,2 % y el 11,9 % de la varianza (27,0 % en total).
- Reconstruyendo con 2 componentes y umbral 0,5: máximo 15 px incorrectos en un patrón, 230 en total, y 32 de 32 patrones con más de 1 px mal. El autoencoder, en la peor de sus semillas: 0 en total.
- Correlación de las distancias entre pares: PCA con Hamming 0,65; AE con PCA semilla 0: 0,62, semilla 1: 0,52, semilla 2: 0,54, semilla 3: 0,64, semilla 4: 0,40.

### 1a.4 · Grilla decodificada (`fig_grilla_semilla*.png`, `grilla.csv`)

Grilla regular de 10 × 10 sobre el rectángulo que cubre los 32 códigos (algoritmo de Autoencoders diap. 34). Cada muestra se umbraliza en 0,5 y se compara con los 32 patrones.

- **Semilla 0** (100 muestras; rango z1 [-31,13; 29,94], z2 [-31,50; 22,60]): 39 son idénticas a una letra del dataset y 61 no. Distancia a la letra más cercana: 39 a 0 px, 10 a 1 px, 5 a 2 px, 9 a 3 px, 4 a 4 px, 4 a 5 px, 7 a 6 px, 8 a 7 px, 6 a 8 px, 5 a 9 px, 1 a 10 px, 2 a 11 px. Letras distintas que aparecen como la más cercana: 23 de 32.
- **Semilla 1** (100 muestras; rango z1 [-23,01; 26,32], z2 [-14,87; 27,32]): 37 son idénticas a una letra del dataset y 63 no. Distancia a la letra más cercana: 37 a 0 px, 16 a 1 px, 14 a 2 px, 7 a 3 px, 5 a 4 px, 9 a 5 px, 4 a 6 px, 5 a 7 px, 1 a 9 px, 2 a 10 px. Letras distintas que aparecen como la más cercana: 26 de 32.
- **Semilla 2** (100 muestras; rango z1 [-18,78; 26,00], z2 [-29,50; 32,46]): 40 son idénticas a una letra del dataset y 60 no. Distancia a la letra más cercana: 40 a 0 px, 4 a 1 px, 4 a 2 px, 3 a 3 px, 10 a 4 px, 10 a 5 px, 5 a 6 px, 7 a 7 px, 2 a 8 px, 1 a 9 px, 3 a 10 px, 2 a 11 px, 4 a 12 px, 2 a 13 px, 3 a 14 px. Letras distintas que aparecen como la más cercana: 26 de 32.
- **Semilla 3** (100 muestras; rango z1 [-23,34; 23,72], z2 [-19,12; 26,97]): 33 son idénticas a una letra del dataset y 67 no. Distancia a la letra más cercana: 33 a 0 px, 9 a 1 px, 9 a 2 px, 5 a 3 px, 14 a 4 px, 7 a 5 px, 7 a 6 px, 10 a 7 px, 2 a 8 px, 3 a 9 px, 1 a 10 px. Letras distintas que aparecen como la más cercana: 24 de 32.
- **Semilla 4** (100 muestras; rango z1 [-21,87; 36,30], z2 [-18,13; 25,05]): 27 son idénticas a una letra del dataset y 73 no. Distancia a la letra más cercana: 27 a 0 px, 10 a 1 px, 13 a 2 px, 11 a 3 px, 4 a 4 px, 6 a 5 px, 6 a 6 px, 3 a 7 px, 2 a 8 px, 8 a 9 px, 6 a 10 px, 4 a 11 px. Letras distintas que aparecen como la más cercana: 23 de 32.

### 1a.4 · Interpolaciones (`fig_interpolaciones_semilla*.png`, `interpolaciones.csv`)

9 puntos sobre el segmento entre los códigos de dos letras (Autoencoders diap. 33). En cada punto: distancia en píxeles a la letra más cercana · esa letra. Distancia de Hamming entre las letras de cada par: n–r: 3, o–e: 5, k–q: 24.

- Semilla 0, n→r: 0·n → 3·m → 3·m → 3·m → 0·m → 4·m → 0·e → 0·e → 0·r.
- Semilla 0, o→e: 0·o → 0·o → 0·o → 2·o → 0·e → 0·e → 0·e → 0·e → 0·e.
- Semilla 0, k→q: 0·k → 0·k → 0·k → 0·r → 0·r → 3·r → 5·o → 0·y → 0·q.
- Semilla 1, n→r: 0·n → 0·n → 0·n → 0·n → 0·n → 0·n → 1·n → 0·r → 0·r.
- Semilla 1, o→e: 0·o → 0·o → 0·o → 0·o → 2·o → 1·e → 0·e → 0·e → 0·e.
- Semilla 1, k→q: 0·k → 0·k → 1·k → 0·p → 0·p → 0·p → 5·s → 4·s → 0·q.
- Semilla 2, n→r: 0·n → 0·n → 0·n → 0·n → 1·n → 0·r → 0·r → 0·r → 0·r.
- Semilla 2, o→e: 0·o → 0·o → 1·o → 0·e → 0·e → 0·e → 0·e → 0·e → 0·e.
- Semilla 2, k→q: 0·k → 0·k → 0·b → 3·r → 5·v → 6·c → 5·~ → 6·` → 0·q.
- Semilla 3, n→r: 0·n → 0·n → 0·n → 1·r → 0·r → 0·r → 0·r → 0·r → 0·r.
- Semilla 3, o→e: 0·o → 0·o → 2·o → 1·e → 0·e → 0·e → 0·e → 0·e → 0·e.
- Semilla 3, k→q: 0·k → 0·k → 3·k → 6·b → 1·r → 4·o → 4·o → 6·u → 0·q.
- Semilla 4, n→r: 0·n → 0·n → 0·n → 0·n → 0·n → 1·n → 0·r → 0·r → 0·r.
- Semilla 4, o→e: 0·o → 0·o → 0·o → 1·e → 0·e → 0·e → 0·e → 0·e → 0·e.
- Semilla 4, k→q: 0·k → 0·k → 0·b → 0·b → 4·b → 5·a → 4·a → 3·a → 0·q.

No se señala ninguna muestra como "letra nueva": lo decide el grupo mirando las grillas.
<!-- e1a3_e1a4:fin -->

<!-- e1b:inicio -->
## 1b · Denoising Autoencoder (e1b)

3 arquitecturas × 5 niveles de entrenamiento × 5 semillas (10000 épocas, receta de E9), cada una probada con 5 niveles de ruido. Ruido salt-and-pepper: cada píxel se invierte con probabilidad p. Cada celda son 5 semillas × 32 patrones × 20 realizaciones de ruido nuevas = 3200 reconstrucciones; se informa media ± desvío entre semillas. Datos en `results/e1b/`.

Figuras: `fig_hasta_1px.png`, `fig_exactos.png`, `fig_pixeles_mal.png` (matrices entrenamiento × prueba), `fig_ganancia.png` (píxeles mal a la salida contra píxeles alterados a la entrada), `fig_confusion.png`.

Píxeles alterados a la entrada, en promedio por patrón: p = 0,0: 0,00, p = 0,05: 1,75, p = 0,1: 3,50, p = 0,2: 7,00, p = 0,3: 10,50 (35·p).

### Sin ruido en la prueba

| Arquitectura | p de entrenamiento | Corridas con ≤ 1 px en los 32 patrones limpios | Px mal en total (media ± desvío) |
|---|---|---|---|
| A (35–20–2–20–35) | 0,0 | 5 de 5 | 0,0 ± 0,0 |
| A (35–20–2–20–35) | 0,05 | 2 de 5 | 5,8 ± 1,9 |
| A (35–20–2–20–35) | 0,1 | 0 de 5 | 47,6 ± 9,8 |
| A (35–20–2–20–35) | 0,2 | 0 de 5 | 158,6 ± 7,1 |
| A (35–20–2–20–35) | 0,3 | 0 de 5 | 229,2 ± 19,9 |
| B (35–20–10–20–35) | 0,0 | 5 de 5 | 0,0 ± 0,0 |
| B (35–20–10–20–35) | 0,05 | 5 de 5 | 0,4 ± 0,9 |
| B (35–20–10–20–35) | 0,1 | 5 de 5 | 1,0 ± 1,2 |
| B (35–20–10–20–35) | 0,2 | 0 de 5 | 21,2 ± 8,1 |
| B (35–20–10–20–35) | 0,3 | 0 de 5 | 71,2 ± 3,3 |
| C (35–50–35) | 0,0 | 5 de 5 | 0,0 ± 0,0 |
| C (35–50–35) | 0,05 | 5 de 5 | 0,0 ± 0,0 |
| C (35–50–35) | 0,1 | 5 de 5 | 0,0 ± 0,0 |
| C (35–50–35) | 0,2 | 4 de 5 | 6,6 ± 1,5 |
| C (35–50–35) | 0,3 | 0 de 5 | 35,2 ± 3,3 |

### Arquitectura A: 35–20–2–20–35

% de reconstrucciones con ≤ 1 píxel incorrecto:

| p de entrenamiento \ p de prueba | 0,0 | 0,05 | 0,1 | 0,2 | 0,3 |
|---|---|---|---|---|---|
| 0,0 | 100,0 ± 0,0 | 36,7 ± 1,6 | 18,5 ± 1,8 | 8,3 ± 1,6 | 3,9 ± 1,1 |
| 0,05 | 96,2 ± 4,1 | 82,3 ± 2,9 | 62,2 ± 2,8 | 24,6 ± 1,8 | 8,2 ± 0,7 |
| 0,1 | 57,5 ± 11,4 | 55,5 ± 7,2 | 46,4 ± 3,9 | 20,9 ± 1,7 | 6,7 ± 1,3 |
| 0,2 | 17,5 ± 2,8 | 18,3 ± 2,1 | 17,4 ± 2,5 | 11,2 ± 1,3 | 4,6 ± 1,0 |
| 0,3 | 8,1 ± 3,6 | 9,0 ± 2,2 | 9,2 ± 2,4 | 8,1 ± 0,9 | 4,6 ± 0,5 |

Píxeles incorrectos a la salida, por patrón:

| p de entrenamiento \ p de prueba | 0,0 | 0,05 | 0,1 | 0,2 | 0,3 |
|---|---|---|---|---|---|
| 0,0 | 0,00 ± 0,00 | 6,97 ± 0,25 | 9,62 ± 0,27 | 12,33 ± 0,06 | 14,33 ± 0,43 |
| 0,05 | 0,18 ± 0,06 | 0,82 ± 0,08 | 2,29 ± 0,19 | 6,73 ± 0,31 | 10,33 ± 0,29 |
| 0,1 | 1,49 ± 0,31 | 1,74 ± 0,23 | 2,63 ± 0,18 | 6,14 ± 0,24 | 9,78 ± 0,33 |
| 0,2 | 4,96 ± 0,22 | 4,92 ± 0,12 | 5,01 ± 0,10 | 6,23 ± 0,13 | 8,49 ± 0,24 |
| 0,3 | 7,16 ± 0,62 | 6,88 ± 0,46 | 6,70 ± 0,25 | 7,09 ± 0,06 | 8,44 ± 0,23 |

### Arquitectura B: 35–20–10–20–35

% de reconstrucciones con ≤ 1 píxel incorrecto:

| p de entrenamiento \ p de prueba | 0,0 | 0,05 | 0,1 | 0,2 | 0,3 |
|---|---|---|---|---|---|
| 0,0 | 100,0 ± 0,0 | 56,9 ± 1,9 | 29,1 ± 2,4 | 8,7 ± 2,3 | 2,6 ± 0,5 |
| 0,05 | 100,0 ± 0,0 | 96,2 ± 0,8 | 81,2 ± 1,2 | 37,8 ± 1,8 | 11,8 ± 1,9 |
| 0,1 | 100,0 ± 0,0 | 96,9 ± 0,5 | 85,2 ± 1,9 | 44,9 ± 1,3 | 15,4 ± 1,6 |
| 0,2 | 83,8 ± 7,8 | 79,8 ± 5,6 | 71,2 ± 5,0 | 40,5 ± 4,5 | 14,6 ± 1,7 |
| 0,3 | 44,4 ± 4,1 | 39,9 ± 6,0 | 34,1 ± 5,4 | 20,1 ± 3,3 | 8,1 ± 1,0 |

Píxeles incorrectos a la salida, por patrón:

| p de entrenamiento \ p de prueba | 0,0 | 0,05 | 0,1 | 0,2 | 0,3 |
|---|---|---|---|---|---|
| 0,0 | 0,00 ± 0,00 | 2,39 ± 0,15 | 4,82 ± 0,23 | 8,49 ± 0,38 | 11,78 ± 0,35 |
| 0,05 | 0,01 ± 0,03 | 0,17 ± 0,06 | 0,96 ± 0,09 | 4,43 ± 0,29 | 8,43 ± 0,16 |
| 0,1 | 0,03 ± 0,04 | 0,15 ± 0,03 | 0,68 ± 0,12 | 3,50 ± 0,11 | 7,34 ± 0,17 |
| 0,2 | 0,66 ± 0,25 | 0,83 ± 0,19 | 1,20 ± 0,21 | 3,18 ± 0,27 | 6,76 ± 0,12 |
| 0,3 | 2,23 ± 0,10 | 2,36 ± 0,14 | 2,67 ± 0,16 | 4,22 ± 0,18 | 6,79 ± 0,11 |

### Arquitectura C: 35–50–35

% de reconstrucciones con ≤ 1 píxel incorrecto:

| p de entrenamiento \ p de prueba | 0,0 | 0,05 | 0,1 | 0,2 | 0,3 |
|---|---|---|---|---|---|
| 0,0 | 100,0 ± 0,0 | 70,9 ± 1,0 | 35,0 ± 2,2 | 5,4 ± 1,2 | 0,6 ± 0,3 |
| 0,05 | 100,0 ± 0,0 | 84,2 ± 1,9 | 52,1 ± 1,1 | 10,1 ± 1,4 | 1,3 ± 0,3 |
| 0,1 | 100,0 ± 0,0 | 83,5 ± 2,5 | 53,0 ± 1,2 | 12,2 ± 1,4 | 1,9 ± 0,4 |
| 0,2 | 99,4 ± 1,4 | 74,4 ± 2,9 | 47,5 ± 2,0 | 12,0 ± 1,4 | 2,2 ± 0,5 |
| 0,3 | 68,8 ± 5,4 | 43,7 ± 2,3 | 28,4 ± 1,7 | 9,2 ± 1,7 | 1,5 ± 0,4 |

Píxeles incorrectos a la salida, por patrón:

| p de entrenamiento \ p de prueba | 0,0 | 0,05 | 0,1 | 0,2 | 0,3 |
|---|---|---|---|---|---|
| 0,0 | 0,00 ± 0,00 | 1,15 ± 0,02 | 2,69 ± 0,09 | 6,33 ± 0,12 | 10,30 ± 0,12 |
| 0,05 | 0,00 ± 0,00 | 0,65 ± 0,04 | 1,80 ± 0,03 | 4,99 ± 0,06 | 8,44 ± 0,13 |
| 0,1 | 0,00 ± 0,00 | 0,68 ± 0,04 | 1,74 ± 0,04 | 4,61 ± 0,05 | 7,99 ± 0,11 |
| 0,2 | 0,21 ± 0,05 | 0,96 ± 0,06 | 1,92 ± 0,06 | 4,40 ± 0,06 | 7,50 ± 0,08 |
| 0,3 | 1,10 ± 0,10 | 1,87 ± 0,05 | 2,65 ± 0,11 | 4,76 ± 0,11 | 7,31 ± 0,04 |

### Píxeles incorrectos a la salida por nivel de prueba

Para cada arquitectura, el nivel de entrenamiento con menos píxeles mal en ese nivel de prueba (entre paréntesis).

| p de prueba | Entrada (px alterados) | AE de 1a (A, p = 0) | A: mejor nivel de entrenamiento | B: mejor nivel de entrenamiento | C: mejor nivel de entrenamiento |
|---|---|---|---|---|---|
| 0,05 | 1,75 | 6,97 ± 0,25 | 0,82 ± 0,08 (p = 0,05) | 0,15 ± 0,03 (p = 0,1) | 0,65 ± 0,04 (p = 0,05) |
| 0,1 | 3,50 | 9,62 ± 0,27 | 2,29 ± 0,19 (p = 0,05) | 0,68 ± 0,12 (p = 0,1) | 1,74 ± 0,04 (p = 0,1) |
| 0,2 | 7,00 | 12,33 ± 0,06 | 6,14 ± 0,24 (p = 0,1) | 3,18 ± 0,27 (p = 0,2) | 4,40 ± 0,06 (p = 0,2) |
| 0,3 | 10,50 | 14,33 ± 0,43 | 8,44 ± 0,23 (p = 0,3) | 6,76 ± 0,12 (p = 0,2) | 7,31 ± 0,04 (p = 0,3) |

### Confusiones entre letras (entrenado con p = 0,1, probado con p = 0,1)

Se asigna a cada salida umbralizada la letra del dataset más cercana por distancia de Hamming. 'l' y '|' difieren en 2 píxeles.

- **A:** 984 de 3200 reconstrucciones quedan más cerca de otra letra que de la original. Más frecuentes: 'w' → 'u': 53 de 100; 'v' → 'u': 44 de 100; 'r' → 'n': 44 de 100; 'e' → 'o': 38 de 100; 'l' → '|': 35 de 100. 'l' → '|': 35 de 100; '|' → 'l': 33 de 100.
- **B:** 228 de 3200 reconstrucciones quedan más cerca de otra letra que de la original. Más frecuentes: '|' → 'l': 25 de 100; 'l' → '|': 13 de 100; '~' → '`': 13 de 100; 'r' → 'n': 11 de 100; 'n' → 'm': 7 de 100. 'l' → '|': 13 de 100; '|' → 'l': 25 de 100.
- **C:** 138 de 3200 reconstrucciones quedan más cerca de otra letra que de la original. Más frecuentes: '|' → 'l': 18 de 100; '~' → '`': 15 de 100; 'r' → 'n': 7 de 100; 'v' → 'u': 7 de 100; 'w' → 'u': 5 de 100. 'l' → '|': 3 de 100; '|' → 'l': 18 de 100.
<!-- e1b:fin -->

<!-- e2b:inicio -->
## 2a y 2b · Dataset de relojes y entrenamiento del VAE (e2b)

Datos en `results/e2b/`. Figuras: `fig_dataset.png`, `fig_curvas.png`, `fig_reconstrucciones.png`.

### 2a · Dataset

- 500 relojes de agujas de 16 × 16 (256 píxeles) en grises en [0, 1], construidos en código (`src/data/relojes.py`) con la semilla 2026. La esfera es fija; lo único que cambia es la hora, sorteada uniforme entre las 12:00 y las 11:59, que fija el ángulo de las dos agujas como en un reloj real.
- Valor medio de un píxel: 0,182. El 36,1 % de los píxeles es gris (ni 0 ni 1). Imágenes distintas: 494 de 500 (las repetidas son relojes que marcan el mismo minuto; lo verifica `tests/test_relojes.py`).

### 2b · Entrenamiento

- Encoder 256–64–(μ y s de 2), decoder 2–64–256, ocultas tanh, salida logística. λ = 1,0, adam con η = 0,001, lotes de 50, 2000 épocas, semillas 0, 1, 2, 3, 4.
- L y R durante el entrenamiento (con el ϵ sorteado en cada actualización):

| Época | L (media ± desvío entre semillas) | R (media ± desvío entre semillas) |
|---|---|---|
| 1 | 173,83 ± 1,22 | 0,19 ± 0,12 |
| 10 | 65,15 ± 0,15 | 5,00 ± 0,13 |
| 100 | 61,68 ± 0,50 | 0,72 ± 0,29 |
| 1000 | 56,41 ± 0,45 | 2,64 ± 0,17 |
| 2000 | 54,37 ± 0,20 | 3,50 ± 0,08 |

- En el último 10 % de las épocas, L cambia -0,22 ± 0,12 (media ± desvío entre semillas del valor final menos el inicial del tramo).
- Piso de L para estos datos: 48,45. Con datos en grises la BCE no llega a 0 ni con una salida idéntica al dato (vale la entropía de los píxeles grises).
- Al terminar, decodificando z = μ(x) sin ruido: L = 53,42 ± 0,28; error absoluto por píxel = 0,020 ± 0,001; R = 3,50 ± 0,10.
- Códigos de los 500 patrones: desvío de μ = 1,03 ± 0,01 (el prior tiene desvío 1); σ media = 0,201 ± 0,008.
- Tiempo por corrida: 52,0 ± 0,2 s.
<!-- e2b:fin -->

<!-- e2c:inicio -->
## 2b y 2c · Latente del AE contra el del VAE, y generación (e2c)

Datos en `results/e2c/`. Figuras: `fig_latente.png`, `fig_grilla.png`, `fig_muestras.png`, `fig_lambda.png`, `fig_candidatas_lambda_*.png`.

- Mismo dataset (500 relojes) y misma arquitectura base (256–64–2 y decoder espejo, ocultas tanh, salida logística, BCE, adam, lotes de 50, 2000 épocas) para el AE común y para el VAE con λ = 0,1, 1, 10.
- 5 semillas por modelo; cada celda es media ± desvío entre semillas. El código de un dato es z para el AE y μ(x) para el VAE.
- Generación: 200 valores de z ~ N(0, I), los mismos para todos los modelos, decodificados. La distancia entre dos imágenes es la media sobre los píxeles de la diferencia absoluta (0 = idénticas).

| Medida | AE | VAE, λ = 0,1 | VAE, λ = 1 | VAE, λ = 10 |
|---|---|---|---|---|
| L con z = código | 50,92 ± 0,32 | 51,89 ± 0,29 | 53,42 ± 0,28 | 62,19 ± 0,00 |
| Error abs. por píxel | 0,012 ± 0,001 | 0,015 ± 0,001 | 0,020 ± 0,001 | 0,040 ± 0,000 |
| Desvío de los códigos | 11,89 ± 1,44 | 1,56 ± 0,05 | 1,03 ± 0,01 | 0,00 ± 0,00 |
| Mayor valor absoluto de una coordenada de un código | 39,49 ± 2,35 | 5,07 ± 0,23 | 2,57 ± 0,15 | 0,01 ± 0,00 |
| Códigos con ‖z‖ ≤ 3 | 0,15 ± 0,05 | 0,82 ± 0,02 | 1,00 ± 0,00 | 1,00 ± 0,00 |
| z ~ N(0, I) dentro del rectángulo de los códigos | 1,00 ± 0,00 | 1,00 ± 0,00 | 0,98 ± 0,00 | 0,00 ± 0,00 |
| Distancia media al más cercano | 0,013 ± 0,001 | 0,015 ± 0,001 | 0,018 ± 0,000 | 0,034 ± 0,000 |
| Distancia mínima al más cercano | 0,007 ± 0,001 | 0,007 ± 0,002 | 0,010 ± 0,001 | 0,034 ± 0,000 |
| Horas (de 12) en las que cae el más cercano de alguna muestra | 7,4 ± 1,5 | 12,0 ± 0,0 | 12,0 ± 0,0 | 1,0 ± 0,0 |
| R | — | 7,84 ± 0,17 | 3,50 ± 0,10 | 0,00 ± 0,00 |
| σ media | — | 0,049 ± 0,002 | 0,201 ± 0,008 | 1,000 ± 0,000 |

- Referencia dentro del dataset: la distancia de cada reloj al otro reloj más cercano es 0,006 en promedio (mínimo 0,000, máximo 0,023).
- Ninguna de las 4000 muestras generadas (todos los modelos y semillas) es idéntica a una imagen del dataset: la menor distancia es 0,004.
- `fig_candidatas_lambda_*.png` trae 40 candidatas numeradas por cada λ (semilla 0), con su z, su imagen más cercana y la distancia en `candidatas.csv`. La elección queda para el grupo.
<!-- e2c:fin -->
