# Selección de padres

> Eje `seleccion` · batch `20260906T233756Z` · 5 semillas · 800 generaciones

## Qué mide este eje

Compara los siete métodos de selección de padres con todo lo demás fijo. Lo que está en juego es la **presión de selección**: cuánto favorece el método a los mejores individuos. Mucha presión converge rápido pero se estanca en un óptimo local; poca presión explora más pero avanza lento.

Los siete se dividen en tres familias por lo que usan para decidir: los que miran el **valor** del fitness (ruleta, universal, Boltzmann), los que miran sólo el **orden** (ranking, elite) y los que hacen **comparaciones locales** (los dos torneos). Esa distinción explica el comportamiento mejor que el nombre del método.

La configuración exacta está en [`config.md`](config.md).

## Resultados — imagen `compleja`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | torneo det | 0.9133 ± 0.0023 | 0.9119–0.9144 | 22.11 | 405 | 3.78e-03 |
| 2 | ranking | 0.9122 ± 0.0025 | 0.9112–0.9131 | 22.38 | 432 | 6.07e-03 |
| 3 | torneo prob | 0.9102 ± 0.0020 | 0.9089–0.9113 | 22.90 | 448 | 6.55e-03 |
| 4 | ruleta | 0.9086 ± 0.0021 | 0.9090–0.9096 | 23.30 | 450 | 9.15e-03 |
| 5 | elite | 0.9086 ± 0.0024 | 0.9063–0.9103 | 23.31 | 424 | 7.64e-03 |
| 6 | boltzmann | 0.9076 ± 0.0020 | 0.9062–0.9089 | 23.55 | 439 | 7.96e-03 |
| 7 | universal | 0.9074 ± 0.0034 | 0.9058–0.9109 | 23.60 | 440 | 8.30e-03 |

**No hay un ganador claro.** torneo det tiene la media más alta, pero su rango intercuartil se solapa con el de ranking: con 5 semillas la diferencia está dentro del ruido. Lo que **sí** se puede afirmar es que le gana a **universal**, que queda separado (0.9074).

- **Rango del eje**: 0.0059 de fitness entre el mejor y el peor (1.5 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 405; el más lento, en la 450.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **torneo det** (0.0001); cuanto más chica, más se parecen entre sí todos los individuos.

## Resultados — imagen `detallada`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | torneo det | 0.8781 ± 0.0050 | 0.8731–0.8826 | 31.07 | 466 | 4.42e-03 |
| 2 | torneo prob | 0.8717 ± 0.0027 | 0.8717–0.8737 | 32.72 | 498 | 7.30e-03 |
| 3 | ranking | 0.8708 ± 0.0032 | 0.8699–0.8734 | 32.94 | 508 | 6.78e-03 |
| 4 | universal | 0.8696 ± 0.0019 | 0.8688–0.8711 | 33.26 | 524 | 7.52e-03 |
| 5 | boltzmann | 0.8689 ± 0.0021 | 0.8671–0.8703 | 33.43 | 508 | 9.54e-03 |
| 6 | ruleta | 0.8684 ± 0.0018 | 0.8677–0.8704 | 33.55 | 547 | 9.74e-03 |
| 7 | elite | 0.8684 ± 0.0027 | 0.8668–0.8675 | 33.56 | 515 | 8.08e-03 |

**No hay un ganador claro.** torneo det tiene la media más alta, pero su rango intercuartil se solapa con el de torneo prob: con 5 semillas la diferencia está dentro del ruido. Lo que **sí** se puede afirmar es que le gana a **elite**, que queda separado (0.8684).

- **Rango del eje**: 0.0098 de fitness entre el mejor y el peor (2.5 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 466; el más lento, en la 547.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **torneo det** (0.0001); cuanto más chica, más se parecen entre sí todos los individuos.

## Resultados — imagen `plana`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | ranking | 0.9409 ± 0.0023 | 0.9400–0.9430 | 15.07 | 602 | 8.17e-03 |
| 2 | torneo det | 0.9396 ± 0.0049 | 0.9393–0.9433 | 15.40 | 551 | 4.99e-03 |
| 3 | universal | 0.9364 ± 0.0054 | 0.9337–0.9393 | 16.23 | 588 | 7.86e-03 |
| 4 | boltzmann | 0.9340 ± 0.0037 | 0.9301–0.9376 | 16.83 | 623 | 9.60e-03 |
| 5 | elite | 0.9335 ± 0.0032 | 0.9306–0.9337 | 16.95 | 606 | 8.90e-03 |
| 6 | ruleta | 0.9333 ± 0.0031 | 0.9322–0.9359 | 17.01 | 634 | 9.72e-03 |
| 7 | torneo prob | 0.9323 ± 0.0030 | 0.9319–0.9345 | 17.25 | 619 | 9.08e-03 |

**No hay un ganador claro.** ranking tiene la media más alta, pero su rango intercuartil se solapa con el de torneo det: con 5 semillas la diferencia está dentro del ruido. Lo que **sí** se puede afirmar es que le gana a **torneo prob**, que queda separado (0.9323).

- **Rango del eje**: 0.0086 de fitness entre el mejor y el peor (2.2 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 551; el más lento, en la 634.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **torneo det** (0.0001); cuanto más chica, más se parecen entre sí todos los individuos.

## Qué mirar

- En `convergencia`, la **pendiente inicial**: los de presión alta despegan primero.
- En `diversidad`, cuál colapsa antes. Presión alta = diversidad que cae rápido, y ése es el mecanismo de la convergencia prematura (**causa 1** de las tres que enumera la cátedra; las otras dos están en `08-tasa-mutacion` y `10-poblacion`).
- Los métodos por valor (ruleta, universal, Boltzmann) pierden presión cuando la población converge, porque todos los fitness se parecen y las probabilidades se vuelven casi uniformes. Los métodos por orden (ranking) y los torneos no: mantienen la misma presión aunque las diferencias sean mínimas. Mirá si eso se nota en la parte final de la curva.
- `elite` es el caso de borde a señalar: cuando K = N devuelve la población entera, o sea presión **cero**. Es consecuencia directa de la fórmula n(i) = ⌈(K−i)/N⌉, no un bug.
- En `boxplot_final`, si las cajas se solapan: si los siete caen en un rango angosto, la conclusión honesta es que **no se distinguen**, y eso también hay que decirlo.

## Trampas y advertencias

- La selección de sobrevivientes está fija en `elite`: acá sólo varía la de padres.
- Los parámetros de los torneos (M y Th) están fijos en sus valores por defecto. El efecto de moverlos se mide aparte, en [`02-presion-seleccion`](../02-presion-seleccion/informe.md), que es el experimento controlado de presión.

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
| compleja | boltzmann | 0.9109 | `imagenes/compleja__boltzmann.png` |
| compleja | elite | 0.9123 | `imagenes/compleja__elite.png` |
| compleja | ranking | 0.9163 | `imagenes/compleja__ranking.png` |
| compleja | ruleta | 0.9104 | `imagenes/compleja__ruleta.png` |
| compleja | torneo det | 0.9119 | `imagenes/compleja__torneo-det.png` |
| compleja | torneo prob | 0.9135 | `imagenes/compleja__torneo-prob.png` |
| compleja | universal | 0.9109 | `imagenes/compleja__universal.png` |
| detallada | boltzmann | 0.8718 | `imagenes/detallada__boltzmann.png` |
| detallada | elite | 0.8668 | `imagenes/detallada__elite.png` |
| detallada | ranking | 0.8655 | `imagenes/detallada__ranking.png` |
| detallada | ruleta | 0.8704 | `imagenes/detallada__ruleta.png` |
| detallada | torneo det | 0.8721 | `imagenes/detallada__torneo-det.png` |
| detallada | torneo prob | 0.8737 | `imagenes/detallada__torneo-prob.png` |
| detallada | universal | 0.8667 | `imagenes/detallada__universal.png` |
| plana | boltzmann | 0.9376 | `imagenes/plana__boltzmann.png` |
| plana | elite | 0.9333 | `imagenes/plana__elite.png` |
| plana | ranking | 0.9430 | `imagenes/plana__ranking.png` |
| plana | ruleta | 0.9322 | `imagenes/plana__ruleta.png` |
| plana | torneo det | 0.9435 | `imagenes/plana__torneo-det.png` |
| plana | torneo prob | 0.9345 | `imagenes/plana__torneo-prob.png` |
| plana | universal | 0.9374 | `imagenes/plana__universal.png` |

## Datos

- `datos/seleccion.csv` — una fila por generación, variante y semilla
- `datos/resumen.csv` — una fila por variante y target, con mediana e IQR
