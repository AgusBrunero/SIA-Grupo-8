# Magnitud de la mutación (σ)

> Eje `sigma` · batch `20260906T233756Z` · 5 semillas · 800 generaciones

## Qué mide este eje

Barre σ, la **magnitud** de la perturbación gaussiana, con la carga de mutación fija. Es la otra mitad de la mutación: `pm` decide *cuántos* genes se tocan y σ *cuánto* se los mueve.

Como todos los genes viven en [0,1] y la perturbación se recorta a ese rango, σ grande no sólo explora más: además **satura** más genes contra los bordes, lo que sesga la distribución hacia 0 y 1.

La configuración exacta está en [`config.md`](config.md).

## Resultados — imagen `compleja`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | sigma=0.10 | 0.9133 ± 0.0023 | 0.9119–0.9144 | 22.11 | 405 | 3.78e-03 |
| 2 | sigma=0.20 | 0.9125 ± 0.0017 | 0.9122–0.9136 | 22.31 | 375 | 4.95e-03 |
| 3 | sigma=0.40 | 0.9108 ± 0.0017 | 0.9096–0.9118 | 22.75 | 397 | 9.11e-03 |
| 4 | sigma=0.05 | 0.9080 ± 0.0053 | 0.9061–0.9093 | 23.45 | 434 | 2.56e-03 |
| 5 | sigma=0.02 | 0.8970 ± 0.0072 | 0.8945–0.9027 | 26.26 | 564 | 1.74e-03 |

**No hay un ganador claro.** sigma=0.10 tiene la media más alta, pero su rango intercuartil se solapa con el de sigma=0.20: con 5 semillas la diferencia está dentro del ruido. Lo que **sí** se puede afirmar es que le gana a **sigma=0.02**, que queda separado (0.8970).

- **Rango del eje**: 0.0163 de fitness entre el mejor y el peor (4.1 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 375; el más lento, en la 564.
- **Diversidad**: 2 de 5 variantes colapsan por debajo del 1% de su diversidad inicial; la primera es **sigma=0.02** en la generación 31.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **sigma=0.02** (0.0001); cuanto más chica, más se parecen entre sí todos los individuos.

## Resultados — imagen `detallada`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | sigma=0.10 | 0.8781 ± 0.0050 | 0.8731–0.8826 | 31.07 | 466 | 4.42e-03 |
| 2 | sigma=0.20 | 0.8749 ± 0.0028 | 0.8742–0.8743 | 31.91 | 478 | 5.95e-03 |
| 3 | sigma=0.05 | 0.8724 ± 0.0052 | 0.8708–0.8772 | 32.54 | 554 | 2.65e-03 |
| 4 | sigma=0.40 | 0.8653 ± 0.0040 | 0.8620–0.8695 | 34.36 | 475 | 1.05e-02 |
| 5 | sigma=0.02 | 0.8522 ± 0.0069 | 0.8497–0.8552 | 37.69 | 575 | 1.48e-03 |

**No hay un ganador claro.** sigma=0.10 tiene la media más alta, pero su rango intercuartil se solapa con el de sigma=0.20: con 5 semillas la diferencia está dentro del ruido. Lo que **sí** se puede afirmar es que le gana a **sigma=0.02**, que queda separado (0.8522).

- **Rango del eje**: 0.0259 de fitness entre el mejor y el peor (6.6 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 466; el más lento, en la 575.
- **Diversidad**: 2 de 5 variantes colapsan por debajo del 1% de su diversidad inicial; la primera es **sigma=0.02** en la generación 34.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **sigma=0.10** (0.0001); cuanto más chica, más se parecen entre sí todos los individuos.

## Resultados — imagen `plana`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | sigma=0.20 | 0.9503 ± 0.0032 | 0.9495–0.9516 | 12.68 | 501 | 7.27e-03 |
| 2 | sigma=0.40 | 0.9413 ± 0.0040 | 0.9391–0.9454 | 14.98 | 575 | 1.10e-02 |
| 3 | sigma=0.10 | 0.9396 ± 0.0049 | 0.9393–0.9433 | 15.40 | 551 | 4.99e-03 |
| 4 | sigma=0.05 | 0.9322 ± 0.0073 | 0.9262–0.9400 | 17.29 | 628 | 3.07e-03 |
| 5 | sigma=0.02 | 0.8995 ± 0.0040 | 0.8969–0.9005 | 25.63 | 649 | 1.65e-03 |

**sigma=0.20** gana con el rango intercuartil **separado** del segundo (sigma=0.40): la diferencia no se explica por el ruido entre semillas.

- **Rango del eje**: 0.0508 de fitness entre el mejor y el peor (13.0 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 501; el más lento, en la 649.
- **Diversidad**: 1 de 5 variantes colapsan por debajo del 1% de su diversidad inicial; la primera es **sigma=0.02** en la generación 28.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **sigma=0.20** (0.0001); cuanto más chica, más se parecen entre sí todos los individuos.

## Qué mirar

- σ chico es ajuste fino: convergencia suave pero lenta. σ grande es exploración: avance rápido al principio y ruido al final. Mirá si eso se ve como un cruce de curvas.
- Este eje explica por qué la mutación **no uniforme** puede tener sentido aunque en [`07-mutacion`](../07-mutacion/informe.md) no gane: lo que la no uniforme hace es moverse por este eje a lo largo de la corrida, empezando con σ grande y terminando con σ chico.
- Un σ óptimo intermedio y un valle ancho significan que el parámetro no es crítico; un óptimo agudo significa que sí, y entonces la mutación no uniforme debería ganar.
- En las imágenes resultado: σ grande produce colores saturados por el clamp a [0,1]. Es visible.

## Trampas y advertencias

- σ se aplica igual a coordenadas y a color. Un σ óptimo distinto para cada tipo de gen es una mejora posible que no implementamos.
- El clamp a [0,1] hace que el efecto de σ no sea simétrico cerca de los bordes del dominio.

## Figuras

Sin título ni texto adentro del PNG, fondo transparente: el rótulo lo pone la slide.

| Archivo | Qué muestra | Cómo se lee |
|---|---|---|
| `convergencia_<target>.png` | Mejor fitness acumulado por generación, media entre semillas con banda ±σ | Pendiente = velocidad; altura final = calidad; ancho de la banda = cuánto depende de la suerte |
| `diversidad_<target>.png` | Diversidad genética por generación, escala log | Cuándo y cuán rápido colapsa la población |
| `boxplot_final_<target>.png` | Fitness final por variante, con los puntos de cada semilla | **Si las cajas se solapan, no hay ganador** |
| `mejor_vs_promedio_<target>.png` | Mejor (sólido) y promedio (guionado) de la población | Cuando el promedio alcanza al mejor, la población convergió |
| `fitness_vs_evaluaciones_<target>.png` | Mejor fitness contra evaluaciones de fitness | Comparación a presupuesto de cómputo igualado, no a generaciones iguales |

## Imágenes resultado

El mejor individuo de cada variante, renderizado a 320px con la primera semilla del barrido (`seed=1`). El target original está como `<target>__TARGET.png`.

| Target | Variante | Fitness | Archivo |
|---|---|---|---|
| compleja | sigma=0.02 | 0.8945 | `imagenes/compleja__sigma0.02.png` |
| compleja | sigma=0.05 | 0.9061 | `imagenes/compleja__sigma0.05.png` |
| compleja | sigma=0.10 | 0.9119 | `imagenes/compleja__sigma0.10.png` |
| compleja | sigma=0.20 | 0.9136 | `imagenes/compleja__sigma0.20.png` |
| compleja | sigma=0.40 | 0.9096 | `imagenes/compleja__sigma0.40.png` |
| detallada | sigma=0.02 | 0.8409 | `imagenes/detallada__sigma0.02.png` |
| detallada | sigma=0.05 | 0.8772 | `imagenes/detallada__sigma0.05.png` |
| detallada | sigma=0.10 | 0.8721 | `imagenes/detallada__sigma0.10.png` |
| detallada | sigma=0.20 | 0.8742 | `imagenes/detallada__sigma0.20.png` |
| detallada | sigma=0.40 | 0.8614 | `imagenes/detallada__sigma0.40.png` |
| plana | sigma=0.02 | 0.9005 | `imagenes/plana__sigma0.02.png` |
| plana | sigma=0.05 | 0.9400 | `imagenes/plana__sigma0.05.png` |
| plana | sigma=0.10 | 0.9435 | `imagenes/plana__sigma0.10.png` |
| plana | sigma=0.20 | 0.9495 | `imagenes/plana__sigma0.20.png` |
| plana | sigma=0.40 | 0.9463 | `imagenes/plana__sigma0.40.png` |

## Datos

- `datos/sigma.csv` — una fila por generación, variante y semilla
- `datos/resumen.csv` — una fila por variante y target, con mediana e IQR
