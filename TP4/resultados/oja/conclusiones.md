# Ej. 1.3 · Índice de creadores con la regla de Oja

Script: `TP4/indice.py` (se corre desde `TP4/` con `python indice.py`, tarda unos 10 s).
Regenera los CSV, gráficos y el modelo de esta carpeta; los Markdown son informes manuales. Modelo: `models/oja.py`.

- **Datos:** las mismas 27 variables estandarizadas (z-score) del 1.1 y el 1.2 (se sacan las
  4 derivadas). 50 creadores.
- **Regla:** una neurona lineal `y = w·x`, `Δw = η · y · (x − y · w)`. Una época = una pasada
  por los 50 creadores en orden aleatorio. `w₀` aleatorio con norma 1.
- **Configuración final:** η(t) = 0,01 / (1 + t/50) (decrece con las iteraciones, τ = una
  época), 100 épocas, semilla 0. El signo de `w` se fija con el mismo criterio que en la PCA
  (el peso de mayor módulo queda positivo), sin mirar a PC1.

**Resumen:** la regla de Oja se aproxima a ‖w‖ = 1 y al autovector de PC1 (ángulo 0,03°, diferencia
máxima por peso 0,0002). El índice resultante **aproxima** la proyección sobre PC1: mismo
ranking en los 50 creadores, con valores ligeramente diferentes. Representa el eje "alcance masivo y caro" (índice bajo) contra
"audiencia chica pero profunda y comprometida" (índice alto). **No** sirve para elegir
automáticamente al creador de mayor valor para cualquier campaña: es un solo eje, inverso al
tamaño, y prácticamente ciego al tema del creador.

---

## Convergencia (`convergencia.png`, `convergencia_resumen.csv`)

Se mide, al final de cada época, ‖w‖ y el ángulo entre `w` y el autovector de PC1 (5 semillas).
La tabla muestra medias finales, salvo el máximo explícito. «Épocas hasta < 1°» es la media
del primer cruce entre semillas cuando las cinco lo alcanzan; no exige permanecer bajo 1°.
La corrida final con semilla 0 tiene un ángulo de 0,030°; la media de las cinco es 0,0308°.

| η | ‖w‖ final | ángulo final con PC1 | épocas hasta < 1° | observación |
|---|---|---|---|---|
| 0,0001 | 1,0003 | 1,6° (máx. 3,7°) | no en todas las semillas | converge pero demasiado lento |
| 0,001 | 1,0033 | 0,9° (máx. 1,2°) | 11 | buen compromiso con η fijo |
| 0,01 | 1,031 | 13° (máx. 17°) | nunca | oscila alrededor de PC1 |
| 0,02 | 1,070 | 21° (máx. 26°) | nunca | oscila más |
| 0,05 | — | — | — | **diverge** en las 5 semillas (≤ 3 épocas) |
| **0,01 / (1 + t/50)** | **1,0004** | **0,03°** | **8** | rápido al principio, fino al final |

Lecturas:

- **La norma se acerca a 1 en las configuraciones estables probadas**: con ‖w₀‖ = 0,01 crece y con
  ‖w₀‖ = 10 decrece (en la primera época ya baja a 3,4) y en ~15 épocas todas las curvas están cerca de 1. Con η fijo = 0,001 terminan en
  aproximadamente 1,0033, no exactamente en 1. Esto no prueba convergencia desde cualquier
  inicialización: por ejemplo, w₀ = 0 permanecería en cero.
  Es el término `− η y² w` de la regla el que hace de normalización implícita.
- **Con η fijo hay un compromiso.** η chico tarda (el ángulo baja como una exponencial lenta);
  η grande llega rápido pero no se queda quieto: cada muestra empuja a `w` en una dirección
  distinta y el vector oscila alrededor de PC1 con mayor ángulo final al aumentar η en los casos probados (13° con 0,01,
  21° con 0,02). También la norma queda sesgada por encima de 1 (1,03 y 1,07): el equilibrio
  estocástico no es exactamente el punto fijo de la regla promedio.
- **Divergencia.** El factor radial de cada paso es `1 − η y²`. Con la dirección PC1 y
  norma unitaria, la mayor proyección al cuadrado es 47,6: cerca de η = 0,021 ese factor
  puede volverse negativo para alguna muestra. Es una referencia de sobrecorrección,
  **no una cota necesaria ni suficiente de estabilidad** de toda la actualización: y depende
  también del peso actual y hay un término η y x. La evidencia experimental es que η = 0,02
  oscila sin divergir en estas corridas y η = 0,05 diverge en las cinco semillas.
- **η decreciente** resuelve el compromiso: arranca en 0,01 (converge en ~8 épocas) y termina
  en 0,0001 (con menor oscilación residual). Es la configuración final. El cronograma inverso
  cumple las condiciones de sumas de pasos usuales si se prolonga indefinidamente, pero
  eso no convierte una corrida finita en convergencia exacta.

## Pesos de Oja contra el autovector de PC1 (`pesos_vs_pc1.png`, `pesos.csv`)

Las 27 barras son indistinguibles: la diferencia máxima entre pesos es 0,0002. Es lo esperado,
porque el punto fijo estable de la regla de Oja es el autovector de la matriz de covarianza con
mayor autovalor, y sobre datos estandarizados esa matriz es la de correlación que usa la PCA.
Hay que tener en cuenta que **el signo es arbitrario**: según la semilla, Oja converge a +PC1 o
a −PC1 (en 10 semillas, 5 y 5). Por eso se fija con una convención y la comparación de ángulo
usa el valor absoluto del coseno.

Composición del vector (ordenado):

| signo | peso | variables |
|---|---|---|
| − | −0,25 a −0,24 | impresiones, video views, alcance, % de posts de tendencia, % de video, % formato corto, precio, followers |
| − | −0,22 a −0,18 | % de entretenimiento, likes/1000, alcance no seguidores, shares/1000 |
| ~0 | −0,07 a +0,03 | topic fútbol, topic maquillaje, posts mensuales, % patrocinado, topic arte, audiencia argentina |
| + | +0,10 a +0,13 | % showcase, topic perros |
| + | +0,18 a +0,26 | % review, watch time, % tutorial, comentarios/1000, visitas al perfil/1000, saves/1000, palabras por caption |

## Qué representa el índice (`indice.png`, `indice.csv`, `correlacion_indice_variables.csv`)

`índice = Z · w`. Su dirección coincide aproximadamente con PC1, que explica el 49,7 % de la varianza total.
La varianza del índice es 13,430, frente a λ₁ = 13,421: la pequeña diferencia procede sobre
todo de ‖w‖ = 1,00035. Para comparar varianza capturada por una dirección se usa
`Var(Z·w) / ‖w‖²`, evitando atribuir esa escala a información adicional.

- **Índice bajo (negativo)**: creadores masivos y caros, de formato corto, tendencias y
  entretenimiento. Correlación del índice con impresiones −0,92, con followers −0,87, con
  precio −0,87, con % de tendencias −0,90.
- **Índice alto (positivo)**: tendencia hacia menor tamaño y precio (por ejemplo, varios
  perfiles de 10–60 mil followers), captions extensos, tutoriales y reviews. Presentan
  mayor retención (watch time) y tasas de guardados,
  comentarios y visitas al perfil. Correlación con palabras por caption +0,94, saves +0,84,
  visitas al perfil +0,77, tutoriales +0,73.
- **Casi no mira el tema**: los pesos de los `topic_*` son relativamente pequeños frente a los de tamaño y profundidad
  (−0,07 fútbol, −0,07 maquillaje, −0,01 arte, +0,13 perros). El sesgo hacia perros es consistente con que en este dataset los creadores de perros son en promedio chicos y profundos (índice medio
  +2,8 contra −0,3 arte, −0,9 fútbol y −1,7 maquillaje).

Es decir, el índice es una **escala "masivo ↔ profundo"**, inversamente asociada al tamaño
y positivamente asociada a señales de interacción y retención. La escala logarítmica del
gráfico de seguidores facilita su lectura; no se ajustó una relación −log(tamaño) ni se
midió calidad o conversión directamente. Con el signo invertido sería un índice de alcance: igual de
válido, es la misma recta.

### ¿Elegir automáticamente al creador con el valor más alto para cualquier campaña?

**No.** Razones, con evidencia:

1. **Es un solo eje.** PC1 explica la mitad de la varianza; las otras 26 dimensiones (por
   ejemplo PC2, que separa arte/showcase de perros/entretenimiento/audiencia argentina) se
   pierden. Dos creadores con el mismo índice pueden ser muy distintos.
2. **Es ciego al tema.** El top 10 por índice tiene 7 creadores de perros, 2 de fútbol y 1 de
   maquillaje; no hay ninguno de arte. Para la campaña de materiales de arte el "mejor" sería
   `@mundoanimal.ar`, un creador de perros, con solo 6,73 % en `topic_art_pct`, según el dataset. El mejor
   de arte recién aparece en el puesto 15 (`@trazosyreels`).
3. **Premia una sola estrategia.** Índice alto = chico y profundo. Para una campaña que busca
   descubrimiento masivo (por ejemplo lanzar la ropa de fútbol) un índice alto no garantiza el alcance requerido: `@consejoscaninos` tiene 9.604 followers. Elegir por índice alto sería el error
   simétrico al que ya comete la agencia eligiendo por followers (índice bajo).
4. **Es relativo al dataset.** Los pesos salen de la correlación entre estos 50 creadores; si
   se recalcula el modelo tras cambiar los creadores o las variables, puede cambiar la recta.
   Para comparar meses hay que conservar una referencia y distinguir las versiones.

Lo que sí hace bien el índice: **resume en un número el eje principal de variación**, y sirve
como filtro o como coordenada dentro de una selección por tema (por ejemplo, entre los creadores
de perros, los de índice alto presentan mayores tasas de visitas al perfil y guardados, señales
que permiten proponer un piloto de consultas y adopciones sin afirmar conversiones observadas).

## Comparación con la proyección sobre PC1

| | Oja | PCA (PC1) |
|---|---|---|
| Vector | `w`, ‖w‖ = 1,0004 | autovector 1, ‖v‖ = 1 |
| Ángulo entre ambos | 0,03° | |
| Correlación de Pearson entre índices | 1,00000 | |
| Spearman (ranking) | 1,00000 | |
| Posiciones distintas en el ranking | 0 de 50 | |
| Cómo se obtiene | iterativo, online, una muestra por vez, sin armar la matriz de covarianza | autodescomposición de la matriz 27×27 |
| Qué devuelve | solo la primera componente | todas las componentes y los autovalores |

Con η fijo la coincidencia es menor: con η = 0,001 el ranking difiere en 4 posiciones (a lo sumo
1 lugar) y con η = 0,01 en 19 posiciones (a lo sumo 2 lugares), por la oscilación descripta
arriba. Es decir, la calidad del índice de Oja como sustituto de PC1 depende directamente del
esquema de η.

**Conclusión:** Oja es una forma iterativa de aproximar PC1 sin diagonalizar la covarianza, útil cuando los datos
llegan de a uno o son muchos, pero sobre 50 creadores y 27 variables no aporta nada que la PCA
no dé, y por sí sola no devuelve el resto de las componentes ni el espectro de varianza explicada. La interpretación
del índice es la de PC1: masivo contra profundo.


## Alcance y cierre del ejercicio 1

La regla implementada aprende una dirección de máxima varianza sobre datos centrados.
Con paso decreciente se aproxima a PC1; con paso fijo conserva fluctuaciones y puede
alterar el ranking. Las pruebas de norma y ángulo se complementan: tener norma cercana
a 1 por sí solo no prueba que se haya aprendido la dirección correcta.

El índice no mide rentabilidad y no sustituye tema, audiencia local ni presupuesto.
La decisión de reducción de dimensionalidad, la interpretación de Kohonen, las propuestas
para los cuatro clientes y la actualización mensual se desarrollan en el
[informe del ejercicio 1](../conclusiones_ejercicio1.md). Las recomendaciones se plantean
como hipótesis para un piloto, ya que no hay ventas ni adopciones observadas.
