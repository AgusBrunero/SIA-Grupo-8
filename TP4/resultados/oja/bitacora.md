# Ej. 1.3 · Bitácora: paso a paso, ideas probadas y descartadas

Registro de cómo se llegó a `indice.py`, `models/oja.py` y `conclusiones.md` (15/10/2026).
Las pruebas rápidas se hicieron en scripts sueltos; lo que quedó está en `indice.py`.

## 1. Punto de partida

- La tarea pide: implementar Oja sobre los mismos datos estandarizados del 1.1, mostrar la
  convergencia de ‖w‖ según η y épocas, interpretar el índice, decir si sirve elegir el máximo
  automáticamente y comparar con PC1 (pesos y ranking, cuidando el signo).
- Se reutilizan `datos.py` (27 variables, z-score) y `models/pca.py` (ya calcula el
  autovector 1 con numpy). Convención de `Kohonen` para η: `learning_rate_f(t, T)`.
- Decisión: **no** cambiar las variables ni agregar log en las de tamaño. El 1.2 y el 1.1 usan
  las mismas 27 estandarizadas; si Oja usara otras, la comparación con PC1 no tendría sentido.

## 2. Primera prueba: η constante

Prototipo de 15 líneas: `w` uniforme en [−1, 1] normalizado, por época una permutación de los
50 creadores, `w += η y (x − y w)`. Grilla η ∈ {1e−4, 1e−3, 5e−3, 1e−2, 2e−2, 5e−2, 1e−1} ×
épocas ∈ {5, 20, 100, 500} × 5 semillas. Se mide ‖w‖ y |cos| con PC1.

| η | resultado |
|---|---|
| 1e−4 | ‖w‖ = 1,0003 pero cos 0,42 a las 20 épocas; recién 0,999 a las 100 |
| 1e−3 | cos 0,9999 desde las 20 épocas; ‖w‖ 1,003 |
| 5e−3 | cos 0,99; no mejora con más épocas |
| 1e−2 | cos 0,96–0,97, ‖w‖ 1,03–1,04; tampoco mejora con épocas |
| 2e−2 | cos 0,89–0,93 |
| 5e−2 | diverge en 3 épocas en las 5 semillas |
| 1e−1 | diverge en la primera época |

Ideas que salieron de acá:

- **Más épocas no arreglan un η grande**: el error final es un ruido estacionario, no un
  transitorio. Con η = 0,01 el cos a las 500 épocas (0,969) es el mismo que a las 20 (0,965).
- **Por qué diverge**: calculé λ₁ = 13,4 y el máximo de y² = (x·v₁)² sobre los datos = 47,6.
  El factor radial `1 − η y²` puede cambiar de signo cerca de η = 0,021 si w es PC1
  unitaria. Sirve como referencia de sobrecorrección: no es una cota de estabilidad de
  la actualización completa, porque y cambia con w y también interviene η y x. La
  evidencia medida es que 0,02 oscila y 0,05 diverge en estas corridas.
- **Descartado**: 500 épocas como valor por defecto. Con 100 ya está todo lo que hay que ver
  y el barrido tarda 10 s en vez de 50.

## 3. Segunda prueba: η decreciente

Hipótesis: con η que arranca grande y termina chico se tiene velocidad y precisión. Probé:

| esquema | cos a las 100 épocas | mínimo entre semillas |
|---|---|---|
| constante 0,001 | 0,99987 | 0,99977 |
| lineal 0,01 → 0,0001 | 1,00000 | 0,99999 |
| 1/t: 0,01 / (1 + t/50) | 1,00000 | 1,00000 |
| constante 0,01 | 0,972 | 0,954 |

Elegí **1/t** (τ = una época): es la forma clásica de las condiciones de Robbins–Monro
(Σ η = ∞, Σ η² < ∞), no depende del total de épocas y a las 100 épocas vale 0,01/101 ≈ 1e−4.
El lineal anda igual de bien pero necesita saber T de antemano. Descartado: el esquema lineal,
para no tener dos opciones equivalentes en el gráfico.

## 4. El gráfico de ‖w‖ era plano

Problema: con `w₀` normalizado a 1 la curva de ‖w‖ arranca en 1 y se queda en 1; no muestra
nada. Dos ideas:

- Mostrar ‖w‖ − 1 en log. Descartada: menos legible y con η grande la norma oscila por arriba y
  por abajo.
- **Variar la norma inicial** (0,01, 1, 3, 10) con η = 0,001. Adoptada: se ve que la regla
  "normaliza" desde los dos lados. Con ‖w₀‖ = 10 la primera época ya la baja a 3,4 y con 0,01
  sube como una sigmoide (al principio y es chico, así que el crecimiento es lento). En ~15
  épocas todas en 1. Chequeé que con ‖w₀‖ = 10 y η = 0,001 no divergiera (no lo hace; con
  η = 0,01 sí sería riesgoso porque y² ∝ ‖w‖²).

Para el panel de ‖w‖ según η probé escala log y quedó con ticks ilegibles (1,2×10⁰…); pasé
ese panel a lineal y dejé log solo donde el rango es grande (norma inicial, ángulos).

## 5. El signo

Oja converge a +v₁ o a −v₁ según el `w₀`: en 10 semillas, 5 de cada uno. Opciones:

- Alinear `w` con PC1 (`w *= sign(w·v₁)`). Descartada: se prefirió una convención independiente para el índice publicado. Alinear el signo
  con PC1 sí sería válido para evaluar dos direcciones cuyo signo es arbitrario.
- **Misma convención que `PCA`**: el peso de mayor módulo queda positivo (`fijar_signo`).
  Adoptada. Con esto `average_caption_words` queda +0,256 en los dos, sin mirar a PC1.
- Para la métrica de convergencia se usa |cos|, así el ángulo no depende del signo.

Consecuencia para la interpretación: índice alto = profundo, índice bajo = masivo. Pensé en
invertirlo para que "más índice = más alcance" (más intuitivo para marketing), pero mantener
la convención de PCA hace que `indice.csv` y `pca_proyecciones.csv` sean comparables sin
transformar. Queda aclarado en las conclusiones que es la misma recta.

## 6. Comparación con PC1 y ranking

- Pesos: barras Oja/PCA por variable, ordenadas por PC1. Diferencia máxima 0,0002; nada que
  discutir, pero sirve como verificación visual de la implementación.
- Ranking: Pearson y Spearman entre el índice y la proyección sobre PC1. **scipy no está en
  `requirements.txt`**, y `pandas.corr(method="spearman")` lo necesita. Descartado agregar la
  dependencia; Spearman = Pearson sobre los rangos, 1 línea.
- Con la configuración final: 0 posiciones distintas de 50. Para mostrar que esto depende de η,
  anoté también los casos η fijo: 4 posiciones distintas con 0,001 y 19 con 0,01.

## 7. Qué representa el índice y la pregunta del "máximo automático"

Ideas para mostrarlo, en orden:

1. Correlación del índice con **todas** las variables numéricas, incluidas las 4 derivadas que
   no entran al modelo (sirve para interpretar: precio en pesos −0,87, alcance estimado en
   Argentina −0,73). Adoptada: `correlacion_indice_variables.csv`.
2. Índice vs followers en log con nombres: deja a la vista que es casi una recta decreciente
   (Spearman −0,92) pero con dispersión vertical asociada también a las otras variables de interacción y formato; no demuestra calidad
   ni conversiones.
3. Índice por tema con la línea del top 10: 7 de 10 son de perros y no hay arte. Es el argumento
   más directo contra elegir el máximo para cualquier campaña.
4. Descartado: un gráfico de "top 5 por tema según el índice". Es útil para el 1.4 (selección
   por cliente), no para el 1.3; se puede sacar de `indice.csv` filtrando por tema.

Conclusión que quedó: el índice es PC1, o sea el eje masivo ↔ profundo; un solo eje, ciego al
tema, relativo al dataset. Sirve como coordenada dentro de una selección por tema, no como
criterio único.

## 8. Cosas que quedaron fuera a propósito

- **Regla de Sanger / varias componentes**: la consigna pide un índice (una neurona).
- **Oja sobre datos sin estandarizar**: PC1 sería "followers y precio" por la escala, y la
  comparación con el 1.1 se rompe. Lo menciono en conclusiones como razón de estandarizar.
- **Mini-batch o versión matricial**: con 50 creadores la actualización por muestra es
  suficiente y permite observar directamente la dinámica de la regla pedida.
- **Optimizar conversiones o ROI**: faltan ventas, adopciones, márgenes y presupuesto.
  Se interpretan perfiles y se propone un piloto, sin atribuir resultados comerciales.

## 9. Cierre y verificación (15/10/2026)

- Se volvió a ejecutar `python indice.py` completo: cinco semillas por configuración,
  100 épocas, más el modelo final con semilla 0. Los CSV y gráficos se regeneraron.
- Modelo final: norma 1,00035, ángulo 0,030° y diferencia máxima por peso cercana a 0,0002.
  Pearson y Spearman redondean a 1,00000; no cambia ninguna de las 50 posiciones.
- Se corrigió «exactamente PC1»: los valores son aproximados aunque el ranking coincida.
  La varianza del índice es 13,430 y λ₁ = 13,421; el peso no tiene norma exactamente 1.
- La norma cercana a 1 se observó en las inicializaciones probadas; no es una garantía para
  cualquier w₀. Con paso fijo queda un sesgo y w₀ = 0 no aprendería.
- Se precisó que el primer cruce del ángulo bajo 1° no implica permanecer por debajo.
- Se retiraron las afirmaciones de conversiones observadas y de una ley −log(tamaño).
  Tampoco se interpreta el umbral radial aproximado como garantía de estabilidad.
- Se completó el [informe del ejercicio 1](../conclusiones_ejercicio1.md): PCA y decisión de
  dimensionalidad, hiperparámetros y regiones del SOM, matriz U, selección para los cuatro
  clientes, comparación con seguidores y mantenimiento. Se actualizaron las casillas de
  `TAREAS.md`; el ejercicio 2 y los entregables generales conservan su estado pendiente.

La bitácora conserva los ensayos exploratorios como antecedentes. Sus prototipos descartados
(η lineal, 10 semillas para el signo y otras grillas) no son salidas del barrido definitivo.
Los resultados reproducibles del script son los CSV y gráficos referidos en conclusiones.


## 10. Cierre en el repositorio

El código, los CSV, los gráficos y el modelo de Oja se incorporaron en `09da83f`.
El informe integrado y estas notas completan la documentación de las secciones 1.1–1.4.
La reproducción incluye preparación del entorno y los tres scripts; el README general y
la presentación del TP quedan para el cierre de ambos ejercicios. `TAREAS.md` se mantiene
como seguimiento local, excluido de los commits.
