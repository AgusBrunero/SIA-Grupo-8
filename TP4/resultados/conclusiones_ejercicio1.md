# Ejercicio 1 · Selección de creadores para campañas

Documento principal con todas las conclusiones: [README de TP4](../README.md).

## 1.1 Exploración y representación

Los 50 registros no tienen faltantes. Se excluyen `creator_id` y `name` del entrenamiento:
son identificadores, no características del perfil. De las 31 variables numéricas se retiran
`price_ars`, `organic_posts_pct`, `total_engagement_per_1000` y `estimated_argentina_reach`
porque se calculan a partir de otras. Quedan **27 variables**, compartidas por PCA, Kohonen
y Oja. Las derivadas siguen disponibles para interpretar los resultados.

Se aplica z-score con media y desvío poblacional por columna. Sin este paso, las diferencias
de escala entre seguidores, precios, porcentajes y tasas dominarían las distancias. Las
distribuciones de tamaño son asimétricas; se conserva la escala original antes de estandarizar
para representar también las diferencias absolutas de alcance. El logaritmo queda como
alternativa de sensibilidad, no como una prueba realizada. No se eliminan creadores grandes
por ser extremos: son perfiles relevantes para campañas masivas.

La estandarización no elimina la redundancia entre seguidores, alcance, impresiones, vistas
y precio. Tampoco elimina las restricciones composicionales: los cuatro temas suman 100 %,
al igual que los cuatro formatos de contenido. Esto explica dos autovalores nulos y obliga
a interpretar los pesos como relaciones dentro de este dataset, no como efectos independientes.
Se conservan estas variables por su significado comercial, admitiendo que los bloques
correlacionados tienen mayor peso conjunto en las distancias.

### Dimensionalidad y cargas

Fuente: [varianza](exploracion/pca_varianza.csv), [scree plot](exploracion/scree.png),
[cargas](exploracion/pca_cargas.csv) y [biplot](exploracion/biplot.png).
Las cargas indicadas son correlaciones variable–componente, no coeficientes del autovector.

| Componente | Varianza | Lectura de las cargas principales |
|---|---:|---|
| PC1 | 49,71 % | Profundidad: captions +0,94, guardados +0,84, visitas al perfil +0,77; frente a impresiones −0,92, vistas −0,91 y alcance −0,91. |
| PC2 | 9,19 % | Showcase +0,66 y arte +0,63 frente a audiencia argentina −0,57, perros −0,41 y entretenimiento −0,34. |
| PC3 | 7,00 % | Patrocinio +0,73 y fútbol +0,66 frente a perros −0,46 y frecuencia mensual −0,36. |
| PC4 | 5,83 % | Maquillaje +0,64 y review +0,36 frente a fútbol −0,59. |
| PC5 | 4,88 % | Frecuencia mensual +0,57 y maquillaje +0,39 frente a audiencia argentina −0,56 y arte −0,47. |
| PC6 | 4,51 % | Maquillaje +0,47 y patrocinio +0,36 frente a showcase −0,36 y perros −0,34. |

**Decisión:** reducir a dos componentes para visualizar, pero entrenar Kohonen y Oja con las
27 variables estandarizadas. PC1–PC2 retienen solo **58,89 %**: una cercanía en el biplot no
garantiza cercanía en todas las variables. Seis componentes superan el autovalor 1 y retienen
81,12 %; serían una reducción razonable si hiciera falta comprimir. Con 50 × 27 datos no hay
una necesidad computacional que justifique perder el 18,88 % restante en el agrupamiento.
PC1 sola tampoco alcanza para decidir campañas: pierde diferencias temáticas visibles desde PC2–PC4.

## 1.2 Kohonen: elección e interpretación

Fuente: [barridos](kohonen/barridos.csv), [hit map](kohonen/hit_map.png),
[matriz U](kohonen/u_matrix.png), [planos](kohonen/component_planes.png),
[perfiles](kohonen/perfiles_neuronas.csv) y [grilla en PCA](kohonen/grilla_en_pca.png).

Se adopta una grilla hexagonal **5 × 5**, inicialización con muestras, 500 épocas,
η lineal 0,5 → 0,01 y radio lineal 3 → 1. El barrido cambia un parámetro por vez con cinco
semillas; no es una búsqueda exhaustiva de interacciones. Los errores se calculan sobre los
mismos datos de entrenamiento y describen representación y topología, no rendimiento comercial.

| Variante | Error de cuantización medio | Error topográfico medio | Neuronas sin asignaciones, media |
|---|---:|---:|---:|
| k = 3 | 3,643 | 0 | 0,2 de 9 |
| k = 5, configuración elegida | 3,016 | 0 | 3,4 de 25 |
| k = 8 | 2,255 | 0 | 32,2 de 64 |
| k = 5, radio final 0 | 2,010 | 0,212 | 1,6 de 25 |

Una grilla mayor reduce el error pero introduce muchas celdas vacías para solo 50 creadores.
El radio final 0 mejora el ajuste local a costa de perder continuidad entre primeras y segundas
BMU. Se prioriza el compromiso de k = 5 y radio final 1 para interpretar regiones vecinas.
Las medias de error varían poco con η inicial (3,016–3,062), inicialización (3,016 frente a
3,036) o 50–1000 épocas (3,016–3,039). Las 500 épocas son una elección conservadora dentro
de esa meseta; no se afirma que sean el único valor adecuado. Radio inicial 3 permite ordenar
la grilla primero y refinarla después.

Entre las cinco semillas de la configuración elegida se toma la de menor error de
cuantización: **EC = 3,0015, ET = 0 y 3 neuronas vacías**. Una neurona vacía tiene pesos y
puede actuar como transición; no equivale a un grupo de creadores observado.

### Regiones de la grilla

Las coordenadas son `(fila, columna)`, desde 0, con fila 0 arriba. Su orientación corresponde
a este modelo; otra semilla puede rotar o reflejar el mapa sin cambiar su interpretación.
Los nombres de usuario no determinan el tema: por ejemplo, `@vidaperruna` tiene tema dominante
fútbol según sus porcentajes, y `@reelargentino`, maquillaje.

| Región | Perfil y ejemplos |
|---|---|
| Arriba izquierda, (0,0) | Fútbol de descubrimiento, alto alcance a no seguidores y tendencias: `@tacticavisual`, `@vidaperruna`, `@makeupconsejos`. |
| Arriba, columnas 1–3 | Predominio de arte/showcase, con tamaño decreciente hacia la derecha: `@pincelcotidiano`, `@artealpaso`, `@canvasycafe`. |
| Arriba derecha, (0,4), (1,3–4) | Review y tutoriales, guardados y comentarios; mezcla fútbol, arte y maquillaje: `@futbolentrepases`, `@arteentusmanos`, `@brochasycolor`. |
| Centro izquierda, (2,0–1) | Maquillaje de gran escala y precio alto: `@mateconideas`, `@brillourbano`, `@maquillajereal`. |
| Centro derecha, (2,3–4) | Tutoriales/reviews y retención: `@trazosyreels`, `@pasiondepotrero`, `@reelargentino`. |
| Abajo izquierda, (4,0–1) | Entretenimiento masivo, mezcla temática: `@golconmate`, `@tribunacreativa`, `@crearenreels`, `@galeriacotidiana`. |
| Abajo derecha, (3,3–4), (4,3–4) | Perros, profundidad y menor tamaño: `@guiadehuellas`, `@huellasyconsejos`, `@consejoscaninos`. |

En los planos de componentes, las columnas siguen aproximadamente PC1 (masividad a la
izquierda, profundidad a la derecha); las filas siguen PC2 (arte/showcase arriba,
perros/entretenimiento y audiencia argentina abajo). Coincide con el biplot, pero el SOM
usa también las dimensiones que ese plano omite. Los grupos mixtos muestran que agrupar
por todas las variables no equivale a clasificar exclusivamente por tema.

### Matriz U

Cada celda muestra la distancia euclídea media entre su vector de 27 pesos estandarizados
y los de sus vecinas adyacentes. Un valor alto señala una transición marcada de perfiles;
uno bajo, continuidad. No mide calidad, alcance ni rentabilidad del creador.

En el modelo guardado los valores van de **1,18 a 2,63**. La zona (3,4), U = 1,18, contiene
cinco creadores de perros similares en profundidad; (0,4), U = 1,37, mantiene continuidad
con sus vecinas. En cambio (4,1), U = 2,63, y (4,3), U = 2,56, muestran transiciones más
marcadas entre entretenimiento y perfiles de mayor profundidad. Son fronteras relativas,
no una demostración de un número exacto de clusters ni un umbral automático de separación.

## 1.3 Oja

El desarrollo completo está en [conclusiones de Oja](oja/conclusiones.md) y el proceso en
la [bitácora](oja/bitacora.md). La configuración η(t) = 0,01/(1 + t/50), 100 épocas y semilla 0
produce norma 1,00035 y ángulo cercano a 0,03° con PC1. El ranking coincide en los 50 registros,
aunque los valores de las proyecciones no son exactamente iguales. El índice describe el
eje masividad–profundidad; no es una estimación de ventas ni un orden universal de calidad.

## 1.4 Decisiones para las campañas

Se asumen campañas en Argentina y, para los clientes comerciales, prioridad en ventas.
Los guardados, visitas al perfil, tutoriales y reviews se toman como señales de consideración,
no como conversiones observadas. La propuesta es una **preselección para un piloto**: no hay
ventas, adopciones, margen, presupuesto ni solapamiento de audiencias para calcular ROI u
optimizar una cartera. Se filtra por afinidad temática, se usa el SOM para localizar perfiles
y se consulta Oja como coordenada de estilo, verificando luego las variables originales.

| Cliente | Preselección | Evidencia en los datos |
|---|---|---|
| Refugios de perros | `@huellasyconsejos` y `@consejoscaninos` | Perros 64,49 % y 82,00 %; audiencia argentina 94,43 % y 86,63 %; visitas al perfil/1000 23,71 y 22,68; ambos USD 75 e índice 6,05 y 6,47, celda (3,4). Prioridad en afinidad local y consultas con bajo costo. |
| Ropa de fútbol | `@futbolentrepases` y `@pasiondepotrero` | Fútbol 49,98 % y 49,88 %; tutoriales 44,20 % y 43,19 %; guardados/1000 19,84 y 19,25; USD 171,46 y 330,11, índice 3,86 y 4,08. Piloto de demostración de uso y consideración. |
| Belleza | `@reelargentino` y `@brochasycolor` | Maquillaje 59,62 % y 66,62 %; tutoriales 57,92 % y reviews 35,79 %, respectivamente; visitas al perfil/1000 24,66 y 19,76; USD 172,53 y 273,56, índice 4,21 y 3,13. |
| Materiales de arte | `@arteentusmanos` y `@colorenmovimiento` | Arte 48,28 % y 68,49 %; tutoriales 36,11 % y 53,78 %; visitas al perfil/1000 16,77 y 18,87; USD 205,54 y 561,97. El segundo tiene más audiencia argentina (72,08 % frente a 43,40 %) aunque menor índice (1,25 frente a 2,28). |

Los importes son `price_usd` del dataset, sin asumir frecuencia de contratación. No se suman
audiencias ni se atribuyen resultados futuros a estas parejas.

### Fútbol: alcance masivo, descubrimiento o profundidad

Para el objetivo de ventas se prioriza **profundidad educativa** en el piloto: demostraciones
de prendas en entrenamiento, explicación y review, coherentes con las regiones derechas.
La mayor afinidad temática de otros perfiles sigue siendo relevante y debe verificarse antes
de contratar; índice alto por sí solo no resuelve esa elección.

Si el objetivo pasa a conocimiento de marca, `@golconmate` (4,0) ofrece 745.025 seguidores,
268.518 de alcance y 77,46 % de audiencia argentina, por USD 3.251,20. Para descubrimiento,
`@tacticavisual` (0,0) tiene 71,52 % de alcance a no seguidores, pero solo 29,52 % de audiencia
argentina y cuesta USD 5.334,65: esa ventaja de descubrimiento no basta para preferirlo en
una campaña local. Las tres estrategias tienen utilidad bajo objetivos distintos; los datos no
demuestran que uno venda más. Se medirían visitas y ventas atribuidas en el piloto.

### Comparación con elegir por seguidores

| Tema | Mayor cantidad de seguidores | Seguidores | USD | Índice Oja |
|---|---|---:|---:|---:|
| Perros | `@crearenreels` | 551.553 | 3.813,64 | −5,60 |
| Fútbol | `@golconmate` | 745.025 | 3.251,20 | −5,49 |
| Belleza | `@mateconideas` | 775.589 | 6.310,69 | −5,25 |
| Arte | `@galeriacotidiana` | 381.324 | 4.866,57 | −5,96 |

Elegir solo por seguidores es simple y favorece escala; puede ser útil en notoriedad.
Sin filtrar por tema, elegiría `@mateconideas` para los cuatro clientes. Incluso filtrando,
los máximos de la tabla son perfiles masivos y más caros que los candidatos del piloto.
Seguidores no incorpora precio, audiencia local, afinidad ni formato y tampoco equivale a
alcance efectivo. La selección conjunta permite ajustar el perfil al objetivo, pero requiere
interpretar varios indicadores y mantener el modelo. No se puede afirmar superioridad en ROI
de ninguna estrategia con este dataset. Elegir el máximo Oja para todos repetiría el problema
de una regla única, esta vez favoreciendo profundidad.

### Mantenimiento

| Cambio | Acción |
|---|---|
| Campaña nueva, mismas variables y población | Mantener modelos y z-score; definir objetivo, tema, audiencia y presupuesto, localizar regiones y revisar candidatos. No hace falta reentrenar por cambiar de cliente. |
| Nuevos creadores con las mismas variables | Transformarlos con las medias/desvíos del entrenamiento y asignar BMU, proyecciones e índice. Revisar distancia a BMU y valores fuera del rango previo; si cambia la población, recalcular z-score y reentrenar PCA, SOM y Oja. El τ de Oja debe actualizarse si se quiere mantener un decaimiento de una época. |
| Variables nuevas o eliminadas | Revisar derivadas, unidades y faltantes; entrenar de nuevo todo el flujo con el nuevo orden y número de columnas. Cambian distancias, covarianza, componentes e índice. |
| Actualización mensual | Validar el CSV y conservar una copia con fecha. Para comparar con el mes anterior, aplicar primero el z-score y modelos anteriores; para actualizar la representación, recalcularlos y versionar ambos resultados. Revisar cambios de perfiles, BMU y rankings. |

Cada versión debe conservar dataset, orden de variables, media, desvío, pesos, hiperparámetros,
semilla y convención de signo. Los `.npz` actuales guardan pesos y variables, pero no medias
ni desvíos: para reproducir su escala hay que conservar el CSV original y reconstruir su
z-score con `datos.py`. Los scripts actuales recalculan el ajuste desde el CSV al ejecutarse;
no implementan todavía un servicio de inferencia o una actualización mensual automática.
Comparar índices recalculados sin distinguir versiones confunde cambios del creador con
cambios de la referencia. Una inversión de signo de PC1 tampoco debe interpretarse como
una inversión del perfil comercial.

## Reproducción

Desde `TP4/`, crear y activar un entorno e instalar las dependencias:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Luego ejecutar:

```bash
python exploracion.py
python agrupamiento.py
python indice.py
```

`agrupamiento.py final` e `indice.py final` regeneran solo los resultados finales;
los modos `barridos` y `convergencia` regeneran los experimentos correspondientes.
Las salidas se escriben en `resultados/exploracion/`, `resultados/kohonen/` y `resultados/oja/`.
El análisis completo de ambos ejercicios y las instrucciones de ejecución están reunidos
en el README principal. Este documento conserva el detalle del ejercicio 1 como respaldo.
