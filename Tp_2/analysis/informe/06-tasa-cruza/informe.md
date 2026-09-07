# Probabilidad de cruza

> Eje `tasa_cruza` · batch `20260906T233756Z` · 5 semillas · 800 generaciones

## Qué mide este eje

Barre la probabilidad de cruza `pc`. Los dos extremos son los interesantes:

- **pc = 0**: no hay recombinación. El algoritmo queda reducido a mutación más selección, o sea una búsqueda local paralela. **Es el sanity check del enunciado**: si el fitness igual mejora, la mutación sola funciona; y la distancia hasta pc>0 es **cuánto aporta realmente la recombinación**, que es la pregunta de fondo de por qué usar un AG y no un hill climbing.
- **pc = 1**: todos los padres se cruzan siempre, no queda ningún individuo que pase intacto a la etapa de mutación.

La configuración exacta está en [`config.md`](config.md).

## Resultados — imagen `compleja`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | pc=0.50 | 0.9135 ± 0.0015 | 0.9126–0.9137 | 22.07 | 391 | 3.99e-03 |
| 2 | pc=0.85 | 0.9133 ± 0.0023 | 0.9119–0.9144 | 22.11 | 405 | 3.78e-03 |
| 3 | pc=1.00 | 0.9120 ± 0.0024 | 0.9103–0.9140 | 22.43 | 393 | 4.23e-03 |
| 4 | pc=0.25 | 0.9102 ± 0.0023 | 0.9100–0.9112 | 22.90 | 408 | 3.66e-03 |
| 5 | pc=0.00 (solo mutacion) | 0.9085 ± 0.0029 | 0.9057–0.9103 | 23.34 | 458 | 3.70e-03 |

**No hay un ganador claro.** pc=0.50 tiene la media más alta, pero su rango intercuartil se solapa con el de pc=0.85: con 5 semillas la diferencia está dentro del ruido. Lo que **sí** se puede afirmar es que le gana a **pc=0.00 (solo mutacion)**, que queda separado (0.9085).

- **Rango del eje**: 0.0050 de fitness entre el mejor y el peor (1.3 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 391; el más lento, en la 458.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **pc=0.25** (0.0001); cuanto más chica, más se parecen entre sí todos los individuos.

## Resultados — imagen `detallada`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | pc=0.85 | 0.8781 ± 0.0050 | 0.8731–0.8826 | 31.07 | 466 | 4.42e-03 |
| 2 | pc=1.00 | 0.8754 ± 0.0013 | 0.8744–0.8763 | 31.77 | 480 | 4.93e-03 |
| 3 | pc=0.50 | 0.8729 ± 0.0042 | 0.8718–0.8756 | 32.42 | 517 | 5.11e-03 |
| 4 | pc=0.25 | 0.8725 ± 0.0056 | 0.8692–0.8740 | 32.52 | 533 | 4.51e-03 |
| 5 | pc=0.00 (solo mutacion) | 0.8645 ± 0.0028 | 0.8631–0.8650 | 34.56 | 555 | 5.14e-03 |

**No hay un ganador claro.** pc=0.85 tiene la media más alta, pero su rango intercuartil se solapa con el de pc=1.00: con 5 semillas la diferencia está dentro del ruido. Lo que **sí** se puede afirmar es que le gana a **pc=0.00 (solo mutacion)**, que queda separado (0.8645).

- **Rango del eje**: 0.0137 de fitness entre el mejor y el peor (3.5 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 466; el más lento, en la 555.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **pc=0.85** (0.0001); cuanto más chica, más se parecen entre sí todos los individuos.

## Resultados — imagen `plana`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | pc=1.00 | 0.9506 ± 0.0048 | 0.9466–0.9525 | 12.60 | 584 | 5.22e-03 |
| 2 | pc=0.50 | 0.9403 ± 0.0039 | 0.9382–0.9391 | 15.23 | 548 | 5.20e-03 |
| 3 | pc=0.85 | 0.9396 ± 0.0049 | 0.9393–0.9433 | 15.40 | 551 | 4.99e-03 |
| 4 | pc=0.25 | 0.9374 ± 0.0051 | 0.9353–0.9412 | 15.96 | 615 | 4.60e-03 |
| 5 | pc=0.00 (solo mutacion) | 0.9319 ± 0.0036 | 0.9292–0.9324 | 17.36 | 650 | 4.79e-03 |

**pc=1.00** gana con el rango intercuartil **separado** del segundo (pc=0.50): la diferencia no se explica por el ruido entre semillas.

- **Rango del eje**: 0.0187 de fitness entre el mejor y el peor (4.8 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 548; el más lento, en la 650.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **pc=0.85** (0.0001); cuanto más chica, más se parecen entre sí todos los individuos.

## Qué mirar

- **La diferencia entre pc=0 y el mejor pc es el aporte neto de la cruza.** Si es chica, el trabajo pesado lo hace la mutación, y conviene decirlo con el número en la mano en vez de asumir que la recombinación es esencial.
- En `diversidad`, pc=0 debería mantener más o menos la misma diversidad que el resto: la cruza redistribuye material genético pero no crea material nuevo.
- Si la curva es plana entre pc=0.25 y pc=1, `pc` no es un parámetro crítico y no vale la pena afinarlo.

## Trampas y advertencias

- Con pc=0 los hijos son copias de los padres y **igual pasan por mutación**: no es una corrida sin cambios.
- `pc` se aplica por par de padres, no por individuo.

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
| compleja | pc=0.00 (solo mutacion) | 0.9057 | `imagenes/compleja__pc0.00-solo-mutacion.png` |
| compleja | pc=0.25 | 0.9112 | `imagenes/compleja__pc0.25.png` |
| compleja | pc=0.50 | 0.9129 | `imagenes/compleja__pc0.50.png` |
| compleja | pc=0.85 | 0.9119 | `imagenes/compleja__pc0.85.png` |
| compleja | pc=1.00 | 0.9131 | `imagenes/compleja__pc1.00.png` |
| detallada | pc=0.00 (solo mutacion) | 0.8631 | `imagenes/detallada__pc0.00-solo-mutacion.png` |
| detallada | pc=0.25 | 0.8670 | `imagenes/detallada__pc0.25.png` |
| detallada | pc=0.50 | 0.8653 | `imagenes/detallada__pc0.50.png` |
| detallada | pc=0.85 | 0.8721 | `imagenes/detallada__pc0.85.png` |
| detallada | pc=1.00 | 0.8744 | `imagenes/detallada__pc1.00.png` |
| plana | pc=0.00 (solo mutacion) | 0.9324 | `imagenes/plana__pc0.00-solo-mutacion.png` |
| plana | pc=0.25 | 0.9379 | `imagenes/plana__pc0.25.png` |
| plana | pc=0.50 | 0.9391 | `imagenes/plana__pc0.50.png` |
| plana | pc=0.85 | 0.9435 | `imagenes/plana__pc0.85.png` |
| plana | pc=1.00 | 0.9447 | `imagenes/plana__pc1.00.png` |

## Datos

- `datos/tasa_cruza.csv` — una fila por generación, variante y semilla
- `datos/resumen.csv` — una fila por variante y target, con mediana e IQR
