# Supervivencia y brecha generacional

> Eje `supervivencia` · batch `20260906T233756Z` · 5 semillas · 800 generaciones

## Qué mide este eje

Cruza dos cosas: la **estrategia de supervivencia** (aditiva vs. exclusiva) y el **tamaño de la descendencia K**, o sea la brecha generacional G = K/N.

Aditiva: compiten los N padres con los K hijos y sobreviven N del pool N+K.
Exclusiva: con K > N se seleccionan N de los K hijos; con K ≤ N pasan los K hijos más N−K padres.

Con K = N/2 la brecha es 0.5 y la mitad de la población sobrevive intacta; con K = 2N se generan el doble de hijos que lugares hay.

La configuración exacta está en [`config.md`](config.md).

## Resultados — imagen `compleja`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | aditiva K=2N | 0.9159 ± 0.0017 | 0.9154–0.9167 | 21.44 | 322 | 2.54e-03 |
| 2 | exclusiva K=2N | 0.9153 ± 0.0019 | 0.9136–0.9167 | 21.59 | 316 | 7.96e-03 |
| 3 | aditiva K=N | 0.9133 ± 0.0023 | 0.9119–0.9144 | 22.11 | 405 | 3.78e-03 |
| 4 | exclusiva K=N | 0.9086 ± 0.0025 | 0.9077–0.9087 | 23.31 | 384 | 1.25e-02 |
| 5 | exclusiva K=N/2 | 0.9082 ± 0.0025 | 0.9056–0.9108 | 23.41 | 457 | 5.51e-03 |
| 6 | aditiva K=N/2 | 0.9072 ± 0.0024 | 0.9073–0.9084 | 23.65 | 459 | 5.27e-03 |

**No hay un ganador claro.** aditiva K=2N tiene la media más alta, pero su rango intercuartil se solapa con el de exclusiva K=2N: con 5 semillas la diferencia está dentro del ruido. Lo que **sí** se puede afirmar es que le gana a **aditiva K=N/2**, que queda separado (0.9072).

- **Rango del eje**: 0.0087 de fitness entre el mejor y el peor (2.2 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 316; el más lento, en la 459.
- **Diversidad**: 1 de 6 variantes colapsan por debajo del 1% de su diversidad inicial; la primera es **aditiva K=2N** en la generación 794.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **aditiva K=2N** (0.0001); cuanto más chica, más se parecen entre sí todos los individuos.

#### A presupuesto de cómputo igualado

Las variantes de este eje **no gastan lo mismo**: al final del barrido, la más cara hizo 95,802 evaluaciones de fitness y la más barata 23,991. Comparadas todas a las **23,991 evaluaciones** que alcanza la más barata:

| # | Variante | Fitness a presupuesto igualado |
|---|---|---|
| 1 | exclusiva K=N/2 | 0.9082 |
| 2 | aditiva K=N/2 | 0.9072 |
| 3 | aditiva K=N | 0.9040 |
| 4 | exclusiva K=N | 0.9001 |
| 5 | aditiva K=2N | 0.8990 |
| 6 | exclusiva K=2N | 0.8987 |

**El orden se da vuelta.** Por generación gana *aditiva K=2N*; por evaluación gasta gana *exclusiva K=N/2*. La ventaja de *aditiva K=2N* no venía del método sino de **haber consumido más cómputo**. Es el gráfico `fitness_vs_evaluaciones` el que hay que llevar a la presentación, no el de convergencia.

## Resultados — imagen `detallada`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | exclusiva K=2N | 0.8844 ± 0.0033 | 0.8822–0.8847 | 29.48 | 468 | 8.64e-03 |
| 2 | aditiva K=2N | 0.8831 ± 0.0047 | 0.8810–0.8869 | 29.81 | 412 | 3.16e-03 |
| 3 | aditiva K=N | 0.8781 ± 0.0050 | 0.8731–0.8826 | 31.07 | 466 | 4.42e-03 |
| 4 | exclusiva K=N | 0.8708 ± 0.0020 | 0.8700–0.8717 | 32.95 | 478 | 1.32e-02 |
| 5 | exclusiva K=N/2 | 0.8663 ± 0.0044 | 0.8630–0.8690 | 34.08 | 515 | 5.59e-03 |
| 6 | aditiva K=N/2 | 0.8659 ± 0.0052 | 0.8653–0.8682 | 34.21 | 523 | 5.85e-03 |

**No hay un ganador claro.** exclusiva K=2N tiene la media más alta, pero su rango intercuartil se solapa con el de aditiva K=2N: con 5 semillas la diferencia está dentro del ruido. Lo que **sí** se puede afirmar es que le gana a **aditiva K=N/2**, que queda separado (0.8659).

- **Rango del eje**: 0.0185 de fitness entre el mejor y el peor (4.7 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 412; el más lento, en la 523.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **aditiva K=2N** (0.0001); cuanto más chica, más se parecen entre sí todos los individuos.

#### A presupuesto de cómputo igualado

Las variantes de este eje **no gastan lo mismo**: al final del barrido, la más cara hizo 95,802 evaluaciones de fitness y la más barata 23,991. Comparadas todas a las **23,991 evaluaciones** que alcanza la más barata:

| # | Variante | Fitness a presupuesto igualado |
|---|---|---|
| 1 | exclusiva K=N/2 | 0.8663 |
| 2 | aditiva K=N/2 | 0.8659 |
| 3 | aditiva K=N | 0.8654 |
| 4 | exclusiva K=N | 0.8581 |
| 5 | aditiva K=2N | 0.8570 |
| 6 | exclusiva K=2N | 0.8539 |

**El orden se da vuelta.** Por generación gana *exclusiva K=2N*; por evaluación gasta gana *exclusiva K=N/2*. La ventaja de *exclusiva K=2N* no venía del método sino de **haber consumido más cómputo**. Es el gráfico `fitness_vs_evaluaciones` el que hay que llevar a la presentación, no el de convergencia.

## Resultados — imagen `plana`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | exclusiva K=2N | 0.9567 ± 0.0019 | 0.9553–0.9573 | 11.04 | 529 | 9.58e-03 |
| 2 | aditiva K=2N | 0.9534 ± 0.0037 | 0.9503–0.9538 | 11.89 | 480 | 4.39e-03 |
| 3 | exclusiva K=N | 0.9416 ± 0.0083 | 0.9358–0.9447 | 14.88 | 519 | 1.48e-02 |
| 4 | aditiva K=N | 0.9396 ± 0.0049 | 0.9393–0.9433 | 15.40 | 551 | 4.99e-03 |
| 5 | aditiva K=N/2 | 0.9307 ± 0.0044 | 0.9280–0.9323 | 17.66 | 626 | 6.51e-03 |
| 6 | exclusiva K=N/2 | 0.9286 ± 0.0038 | 0.9266–0.9286 | 18.20 | 632 | 7.22e-03 |

**exclusiva K=2N** gana con el rango intercuartil **separado** del segundo (aditiva K=2N): la diferencia no se explica por el ruido entre semillas.

- **Rango del eje**: 0.0281 de fitness entre el mejor y el peor (7.2 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 480; el más lento, en la 632.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **aditiva K=2N** (0.0001); cuanto más chica, más se parecen entre sí todos los individuos.

#### A presupuesto de cómputo igualado

Las variantes de este eje **no gastan lo mismo**: al final del barrido, la más cara hizo 95,802 evaluaciones de fitness y la más barata 23,991. Comparadas todas a las **23,991 evaluaciones** que alcanza la más barata:

| # | Variante | Fitness a presupuesto igualado |
|---|---|---|
| 1 | aditiva K=N/2 | 0.9307 |
| 2 | exclusiva K=N/2 | 0.9286 |
| 3 | exclusiva K=N | 0.9236 |
| 4 | aditiva K=N | 0.9175 |
| 5 | aditiva K=2N | 0.9138 |
| 6 | exclusiva K=2N | 0.9134 |

**El orden se da vuelta.** Por generación gana *exclusiva K=2N*; por evaluación gasta gana *aditiva K=N/2*. La ventaja de *exclusiva K=2N* no venía del método sino de **haber consumido más cómputo**. Es el gráfico `fitness_vs_evaluaciones` el que hay que llevar a la presentación, no el de convergencia.

## Qué mirar

- **Agrupá las variantes por K, no por estrategia.** Si todas las de un mismo K quedan juntas, lo que decide es la brecha generacional y no la estrategia — y ésa es la conclusión.
- `fitness_vs_evaluaciones` es acá **el gráfico decisivo**: una generación con K=2N cuesta el doble de evaluaciones que una con K=N. Si la ventaja de K=2N desaparece al graficar contra evaluaciones, no había ventaja: había más presupuesto.
- Con supervivencia exclusiva el mejor de la población **puede empeorar** de una generación a la otra, porque los hijos desplazan a los padres. Por eso la curva usa el mejor acumulado; la línea punteada de `mejor_vs_promedio` muestra el efecto.
- Con K = N/2 y exclusiva, la mitad de la población pasa sin competir: mirá si eso frena la convergencia o si actúa como elitismo encubierto.

## Trampas y advertencias

- La combinación elite + K=N + exclusiva deja al algoritmo **sin ninguna presión de selección** (elite con k=N devuelve todo, y exclusiva reemplaza todo): degenera en caminata aleatoria. Por eso la base usa torneo en los padres.
- Los nombres aditiva/exclusiva siguen las láminas 45 y 46 del deck; están citadas textualmente en el docstring de `ga/replacement.py`.

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
| compleja | aditiva K=2N | 0.9186 | `imagenes/compleja__aditiva-k2n.png` |
| compleja | aditiva K=N | 0.9119 | `imagenes/compleja__aditiva-kn.png` |
| compleja | aditiva K=N/2 | 0.9084 | `imagenes/compleja__aditiva-kn-sobre-2.png` |
| compleja | exclusiva K=2N | 0.9182 | `imagenes/compleja__exclusiva-k2n.png` |
| compleja | exclusiva K=N | 0.9051 | `imagenes/compleja__exclusiva-kn.png` |
| compleja | exclusiva K=N/2 | 0.9108 | `imagenes/compleja__exclusiva-kn-sobre-2.png` |
| detallada | aditiva K=2N | 0.8750 | `imagenes/detallada__aditiva-k2n.png` |
| detallada | aditiva K=N | 0.8721 | `imagenes/detallada__aditiva-kn.png` |
| detallada | aditiva K=N/2 | 0.8653 | `imagenes/detallada__aditiva-kn-sobre-2.png` |
| detallada | exclusiva K=2N | 0.8822 | `imagenes/detallada__exclusiva-k2n.png` |
| detallada | exclusiva K=N | 0.8711 | `imagenes/detallada__exclusiva-kn.png` |
| detallada | exclusiva K=N/2 | 0.8690 | `imagenes/detallada__exclusiva-kn-sobre-2.png` |
| plana | aditiva K=2N | 0.9529 | `imagenes/plana__aditiva-k2n.png` |
| plana | aditiva K=N | 0.9435 | `imagenes/plana__aditiva-kn.png` |
| plana | aditiva K=N/2 | 0.9251 | `imagenes/plana__aditiva-kn-sobre-2.png` |
| plana | exclusiva K=2N | 0.9547 | `imagenes/plana__exclusiva-k2n.png` |
| plana | exclusiva K=N | 0.9409 | `imagenes/plana__exclusiva-kn.png` |
| plana | exclusiva K=N/2 | 0.9271 | `imagenes/plana__exclusiva-kn-sobre-2.png` |

## Datos

- `datos/supervivencia.csv` — una fila por generación, variante y semilla
- `datos/resumen.csv` — una fila por variante y target, con mediana e IQR
