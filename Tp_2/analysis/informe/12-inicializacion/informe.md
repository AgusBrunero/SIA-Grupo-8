# Inicialización de la población

> Eje `inicializacion` · batch `20260906T233756Z` · 5 semillas · 800 generaciones

## Qué mide este eje

Compara arrancar de ruido uniforme contra arrancar de una grilla donde cada triángulo toma el **color promedio que el target tiene en esa celda**.

Es el eje con más contenido conceptual del trabajo, y conviene plantearlo así: la grilla es **conocimiento del problema inyectado antes del algoritmo**, no expresado dentro de él. Funciona, pero el mérito no es del AG — es del preproceso. La alternativa honesta es meter la misma información *adentro*: en la función de aptitud, o en un operador de mutación guiado.

La configuración exacta está en [`config.md`](config.md).

## Resultados — imagen `compleja`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | grilla informada | 0.9180 ± 0.0009 | 0.9173–0.9180 | 20.92 | 387 | 2.27e-03 |
| 2 | al azar | 0.9133 ± 0.0023 | 0.9119–0.9144 | 22.11 | 405 | 3.78e-03 |

**grilla informada** gana con el rango intercuartil **separado** del segundo (al azar): la diferencia no se explica por el ruido entre semillas.

- **Rango del eje**: 0.0047 de fitness entre el mejor y el peor (1.2 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 387; el más lento, en la 405.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **al azar** (0.0001); cuanto más chica, más se parecen entre sí todos los individuos.

## Resultados — imagen `detallada`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | grilla informada | 0.8842 ± 0.0037 | 0.8823–0.8851 | 29.52 | 433 | 3.21e-03 |
| 2 | al azar | 0.8781 ± 0.0050 | 0.8731–0.8826 | 31.07 | 466 | 4.42e-03 |

**Las 2 variantes son indistinguibles a esta escala**: ni siquiera el primero se separa del último (0.8842 contra 0.8781, con rangos que se solapan). El resultado es un empate, y como tal hay que presentarlo.

- **Rango del eje**: 0.0061 de fitness entre el mejor y el peor (1.6 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 433; el más lento, en la 466.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **al azar** (0.0001); cuanto más chica, más se parecen entre sí todos los individuos.

## Resultados — imagen `plana`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | grilla informada | 0.9659 ± 0.0030 | 0.9636–0.9663 | 8.71 | 398 | 3.23e-03 |
| 2 | al azar | 0.9396 ± 0.0049 | 0.9393–0.9433 | 15.40 | 551 | 4.99e-03 |

**grilla informada** gana con el rango intercuartil **separado** del segundo (al azar): la diferencia no se explica por el ruido entre semillas.

- **Rango del eje**: 0.0262 de fitness entre el mejor y el peor (6.7 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 398; el más lento, en la 551.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **grilla informada** (0.0001); cuanto más chica, más se parecen entre sí todos los individuos.

## Qué mirar

- Mirá si la ventaja es sólo un **arranque más alto que se diluye**, o si se sostiene hasta el final. Que las curvas no se crucen es el resultado fuerte; que se crucen sería el resultado interesante.
- Mirá también el **desvío entre semillas**: una inicialización informada suele reducir la varianza, no sólo subir la media. Eso se ve en el ancho de la banda y en el alto de las cajas.
- En `diversidad`, la grilla arranca con menos diversidad (los colores ya están cerca del target). Si igual llega más lejos, es un contraejemplo útil a «más diversidad es mejor».
- Las imágenes de la generación 1 son la mejor ilustración: comparar el punto de partida de las dos vale más que cualquier número.

## Trampas y advertencias

- La base de **todos** los demás ejes usa `random` justamente para no contaminar sus mediciones con esta ventaja.
- Los vértices de la grilla siguen siendo aleatorios: si no, todos los individuos arrancarían idénticos y no habría diversidad para evolucionar.
- La ventaja de la grilla crece con la cantidad de triángulos (grilla más fina). Con 50 triángulos se ve una parte del efecto; el cruce completo está en [`15-caso-final`](../15-caso-final/informe.md).

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
| compleja | al azar | 0.9119 | `imagenes/compleja__al-azar.png` |
| compleja | grilla informada | 0.9196 | `imagenes/compleja__grilla-informada.png` |
| detallada | al azar | 0.8721 | `imagenes/detallada__al-azar.png` |
| detallada | grilla informada | 0.8851 | `imagenes/detallada__grilla-informada.png` |
| plana | al azar | 0.9435 | `imagenes/plana__al-azar.png` |
| plana | grilla informada | 0.9715 | `imagenes/plana__grilla-informada.png` |

## Datos

- `datos/inicializacion.csv` — una fila por generación, variante y semilla
- `datos/resumen.csv` — una fila por variante y target, con mediana e IQR
