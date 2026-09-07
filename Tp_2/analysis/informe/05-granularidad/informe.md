# Granularidad de la cruza

> Eje `granularidad` · batch `20260906T233756Z` · 5 semillas · 800 generaciones

## Qué mide este eje

El experimento conceptualmente más interesante del TP. Con la misma cruza uniforme, cambia sólo la **unidad de corte**: un componente suelto (puede partir un triángulo al medio, mezclando coordenadas de uno con color de otro) o un triángulo entero.

La hipótesis es teórica y se enuncia **antes** de medir: como el orden de la lista define el z-order, la posición en el cromosoma tiene significado real. Cortar adentro de un triángulo destruye una unidad semántica. En el vocabulario de la cátedra, el triángulo es el **gen** y los 10 valores son su alelo; cortar por componente es cortar por debajo del gen.

La configuración exacta está en [`config.md`](config.md).

## Resultados — imagen `compleja`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | corte por triangulo | 0.9151 ± 0.0020 | 0.9136–0.9168 | 21.65 | 378 | 3.87e-03 |
| 2 | corte por componente | 0.9143 ± 0.0014 | 0.9141–0.9154 | 21.84 | 351 | 4.03e-03 |

**Las 2 variantes son indistinguibles a esta escala**: ni siquiera el primero se separa del último (0.9151 contra 0.9143, con rangos que se solapan). El resultado es un empate, y como tal hay que presentarlo.

- **Rango del eje**: 0.0008 de fitness entre el mejor y el peor (0.2 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 351; el más lento, en la 378.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **corte por triangulo** (0.0001); cuanto más chica, más se parecen entre sí todos los individuos.

## Resultados — imagen `detallada`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | corte por triangulo | 0.8802 ± 0.0021 | 0.8795–0.8801 | 30.56 | 439 | 4.94e-03 |
| 2 | corte por componente | 0.8774 ± 0.0039 | 0.8760–0.8814 | 31.28 | 444 | 5.20e-03 |

**Las 2 variantes son indistinguibles a esta escala**: ni siquiera el primero se separa del último (0.8802 contra 0.8774, con rangos que se solapan). El resultado es un empate, y como tal hay que presentarlo.

- **Rango del eje**: 0.0028 de fitness entre el mejor y el peor (0.7 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 439; el más lento, en la 444.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **corte por triangulo** (0.0001); cuanto más chica, más se parecen entre sí todos los individuos.

## Resultados — imagen `plana`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | corte por componente | 0.9539 ± 0.0061 | 0.9514–0.9565 | 11.75 | 524 | 5.47e-03 |
| 2 | corte por triangulo | 0.9507 ± 0.0066 | 0.9474–0.9573 | 12.56 | 493 | 4.91e-03 |

**Las 2 variantes son indistinguibles a esta escala**: ni siquiera el primero se separa del último (0.9539 contra 0.9507, con rangos que se solapan). El resultado es un empate, y como tal hay que presentarlo.

- **Rango del eje**: 0.0032 de fitness entre el mejor y el peor (0.8 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 493; el más lento, en la 524.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **corte por triangulo** (0.0001); cuanto más chica, más se parecen entre sí todos los individuos.

## Qué mirar

- Es un eje de **dos variantes**: el boxplot es el gráfico decisivo, no la curva.
- Si las cajas se solapan, la hipótesis **no queda demostrada** aunque la media favorezca al corte por triángulo. Decirlo así es más fuerte que forzar una conclusión.
- El efecto debería crecer con la cantidad de triángulos: con más triángulos hay más que romper. Vale contrastarlo con [`11-triangulos`](../11-triangulos/informe.md).

## Trampas y advertencias

- Con dos variantes el poder estadístico es el que dan las semillas. Un empate no refuta la hipótesis: dice que el experimento no alcanza para decidirla.
- El efecto de la granularidad interactúa con el método de cruza: acá se mide sólo con `uniform`, que es donde debería notarse más (es la que rompe más bloques).

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
| compleja | corte por componente | 0.9116 | `imagenes/compleja__corte-por-componente.png` |
| compleja | corte por triangulo | 0.9136 | `imagenes/compleja__corte-por-triangulo.png` |
| detallada | corte por componente | 0.8761 | `imagenes/detallada__corte-por-componente.png` |
| detallada | corte por triangulo | 0.8800 | `imagenes/detallada__corte-por-triangulo.png` |
| plana | corte por componente | 0.9514 | `imagenes/plana__corte-por-componente.png` |
| plana | corte por triangulo | 0.9573 | `imagenes/plana__corte-por-triangulo.png` |

## Datos

- `datos/granularidad.csv` — una fila por generación, variante y semilla
- `datos/resumen.csv` — una fila por variante y target, con mediana e IQR
